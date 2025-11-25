import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import math

import sys
from model.backbone.patch_embed import PatchEmbed
from thirdparty.DepthAnythingV2.depth_anything_v2.dpt import DPTHead

MODEL_CONFIGS = {
    'vitl': {'encoder': 'vit_large_patch16_224', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitb': {'encoder': 'vit_base_patch16_224', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vits': {'encoder': 'vit_small_patch16_224', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitt': {'encoder': 'vit_tiny_patch16_224', 'features': 32, 'out_channels': [24, 48, 96, 192]}
}

class VisionTransformer(nn.Module):
    def __init__(self, model_name, input_dim, patch_size=16):
        super(VisionTransformer, self).__init__()
        model = timm.create_model(
            MODEL_CONFIGS[model_name]['encoder'],
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        )
        self.intermediate_layer_idx = {
            'vitt': [2, 5, 8, 11],
            'vits': [2, 5, 8, 11],
            'vitb': [2, 5, 8, 11], 
            'vitl': [4, 11, 17, 23], 
            'vitg': [9, 19, 29, 39]
        }
        self.idx = self.intermediate_layer_idx[model_name]
        self.blks = model.blocks
        self.embed_dim = model.embed_dim
        self.input_dim = input_dim
        self.img_size = (224, 224)
        self.patch_size = patch_size
        self.output_dim = MODEL_CONFIGS[model_name]['features']
        self.pos_embed = nn.Parameter(torch.zeros(1, 64, self.embed_dim))
        self.patch_embed = PatchEmbed(img_size=self.img_size, patch_size=self.patch_size, in_chans=input_dim, embed_dim=self.embed_dim)
        self.dpt_head = DPTHead(self.embed_dim, MODEL_CONFIGS[model_name]['features'], out_channels=MODEL_CONFIGS[model_name]['out_channels'])

    def interpolate_pos_encoding(self, x, h, w):
        previous_dtype = x.dtype
        npatch = x.shape[1]
        N = self.pos_embed.shape[1]
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        sqrt_N = math.sqrt(N)
        sx, sy = float(w0) / sqrt_N, float(h0) / sqrt_N
        pos_embed = nn.functional.interpolate(
            pos_embed.reshape(1, int(sqrt_N), int(sqrt_N), dim).permute(0, 3, 1, 2),
            scale_factor=(sy, sx),
            mode="bicubic",
            antialias=False
        )
        assert int(w0) == pos_embed.shape[-1]
        assert int(h0) == pos_embed.shape[-2]
        pos_embed = pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return pos_embed.to(previous_dtype)

    def forward(self, x):
        B, nc, h, w = x.shape
        x = self.patch_embed(x)
        x = x + self.interpolate_pos_encoding(x, h, w)
        outputs = []
        for i in range(len(self.blks)):
            x = self.blks[i](x)
            if i in self.idx:
                outputs.append([x])

        patch_h, patch_w = h // self.patch_size, w // self.patch_size
        out, path_1, path_2, path_3, path_4 = self.dpt_head.forward(outputs, patch_h, patch_w, return_intermediate=True)
        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}  # path_1 is 1/2; path_2 is 1/4

class ViTModulator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.act1 = nn.GELU()
        self.fc1 = nn.Linear(dim, dim)
        
        self.act2 = nn.GELU()
        self.fc2 = nn.Linear(dim, dim)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, scale, shift):
        """
        x: (B, N, C)
        scale, shift: (B, 1, C) -> Broadcasting됨
        """
        shortcut = x
        
        x = self.norm1(x)
        x = self.act1(x)
        x = self.fc1(x)
        
        x = x * (scale + 1) + shift
        
        x = self.act2(x)
        x = self.fc2(x)
        
        return shortcut + x * self.gamma

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class CrossAttentionAdapter(nn.Module):
    def __init__(self, dim, context_dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        
        # Project context if dimensions don't match
        self.context_proj = nn.Linear(context_dim, dim) if context_dim != dim else nn.Identity()

    def forward(self, x, context):
        """
        x: (B, N, C) - Image tokens
        context: (B, M, C_ctx) - Time embeddings (or other context)
        """
        shortcut = x
        x = self.norm(x)
        context = self.context_proj(context)
        
        # MultiheadAttention expects (B, N, C) if batch_first=True
        # query=x, key=context, value=context
        x, _ = self.cross_attn(x, context, context)
        
        return shortcut + x * self.gamma


class VisionTransformerDFM(nn.Module):
    def __init__(self, feature_dim=192, time_dim=128, num_modulators=4):
        super().__init__()
        self.time_dim = time_dim
        self.feature_dim = feature_dim 
        
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_dim),
            nn.Linear(time_dim, time_dim * 2),
            nn.GELU(),
            nn.Linear(time_dim * 2, time_dim),
        )
        
        self.block_time_mlp = nn.Sequential(
            nn.SiLU(), 
            nn.Linear(time_dim, 384 * 2) 
        )
        
        self.modulators = nn.ModuleList([
            ViTModulator(384) for _ in range(num_modulators)
        ])
        
        self.cross_attns = nn.ModuleList([
            CrossAttentionAdapter(dim=384, context_dim=time_dim) for _ in range(num_modulators)
        ])
    
    def forward(self, x, dfm_params=[]):
        t, funcs = dfm_params
        
        b = t.shape[0]
        time_emb = self.time_mlp(t) 
        
        style = self.block_time_mlp(time_emb)
        
        style = style.unsqueeze(1) 
        
        scale, shift = style.chunk(2, dim=-1) 

        B, nc, h, w = x.shape
        
        x = funcs.patch_embed(x)
        x = x + funcs.interpolate_pos_encoding(x, h, w)
        
        outputs = []
        modulator_idx = 0 
        
        for i in range(len(funcs.blks)):
            x = funcs.blks[i](x)
            
            if i in funcs.idx:
                x_modulated = self.modulators[modulator_idx](x, scale, shift)
                
                # Apply Cross-Attention
                # x_modulated: (B, N, C)
                # time_emb: (B, time_dim) -> (B, 1, time_dim)
                x_modulated = self.cross_attns[modulator_idx](x_modulated, time_emb.unsqueeze(1))
                
                outputs.append([x_modulated])
                modulator_idx += 1

        patch_h, patch_w = h // funcs.patch_size, w // funcs.patch_size
        
        out, path_1, path_2, path_3, path_4 = funcs.dpt_head.forward(outputs, patch_h, patch_w, return_intermediate=True)
        out = F.interpolate(out, (h, w), mode="bilinear", align_corners=True)
        
        return {'out': out, 'path_1':path_1, 'path_2':path_2, 'path_3':path_3, 'path_4':path_4}