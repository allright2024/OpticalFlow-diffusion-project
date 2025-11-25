import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import timm
from torch.cuda.amp import autocast

from model.backbone.twins import TwinsFeatureEncoder
from model.backbone.waftv2_dav2 import DepthAnythingFeature
from model.backbone.dinov3 import DinoV3Feature
from model.backbone.vit_v2 import VisionTransformer, MODEL_CONFIGS, VisionTransformerDFM
from utils.utils import coords_grid, Padder, bilinear_sampler

class ConvEE(nn.Module):
    def __init__(self, C_in, C_out):
        super().__init__()
        groups = 4
        self.conv1 = nn.Sequential(
            nn.GroupNorm(groups, C_in),  
            nn.GELU(),
            nn.Conv2d(C_in, C_in, 3, padding=1),
            nn.GroupNorm(groups, C_in))
        self.conv2 = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(C_in, C_in, 3, padding=1))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x, t_emb):
        scale, shift = t_emb
        x_res = x
        x = self.conv1(x)

        x = x * (scale + 1) + shift

        x = self.conv2(x)
        x_o = x * self.gamma

        return x_o

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """extract the appropriate  t  index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def ste_round(x):
    return torch.round(x) - x.detach() + x

def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)

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

class resconv(nn.Module):
    def __init__(self, inp, oup, k=3, s=1):
        super(resconv, self).__init__()
        self.conv = nn.Sequential(
            nn.GELU(),
            nn.Conv2d(inp, oup, kernel_size=k, stride=s, padding=k//2, bias=True),
            nn.GELU(),
            nn.Conv2d(oup, oup, kernel_size=3, stride=1, padding=1, bias=True),
        )
        if inp != oup or s != 1:
            self.skip_conv = nn.Conv2d(inp, oup, kernel_size=1, stride=s, padding=0, bias=True)
        else:
            self.skip_conv = nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.skip_conv(x)

class ResNet18Deconv(nn.Module):
    def __init__(self, inp, oup):
        super(ResNet18Deconv, self).__init__()
        self.feature_dims = [64, 128, 256, 512]
        self.ds1 = resconv(inp, 64, k=7, s=2)
        self.conv1 = resconv(64, 64, k=3, s=1)
        self.conv2 = resconv(64, 128, k=3, s=2)
        self.conv3 = resconv(128, 256, k=3, s=2)
        self.conv4 = resconv(256, 512, k=3, s=2)
        self.up_4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_3 = resconv(256, 256, k=3, s=1)
        self.up_3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_2 = resconv(128, 128, k=3, s=1)
        self.up_2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0, bias=True)
        self.proj_1 = resconv(64, oup, k=3, s=1)

    def forward(self, x):
        out_1 = self.ds1(x)
        out_1 = self.conv1(out_1)
        out_2 = self.conv2(out_1)
        out_3 = self.conv3(out_2)
        out_4 = self.conv4(out_3)
        out_3 = self.proj_3(out_3 + self.up_4(out_4))
        out_2 = self.proj_2(out_2 + self.up_3(out_3))
        out_1 = self.proj_1(out_1 + self.up_2(out_2))
        return [out_1, out_2, out_3, out_4]

class UpSampleMask4(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up_sample_mask = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=4 * 9, kernel_size=1, stride=1)
        )

    def forward(self, data):
        """
        :param data:  B, C, H, W
        :return:  batch, 8*8*9, H, W
        """
        mask = self.up_sample_mask(data)  # B, 64*6, H, W
        return mask

class UpSampleMask8(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up_sample_mask = nn.Sequential(
            nn.Conv2d(in_channels=dim, out_channels=256, kernel_size=3, padding=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=64 * 9, kernel_size=1, stride=1)
        )

    def forward(self, data):
        """
        :param data:  B, C, H, W
        :return:  batch, 64*9, H, W
        """
        mask = self.up_sample_mask(data)
        return mask

class WAFTv2_FlowDiffuser_TwoStage(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.args = args
    
        
        if args.feature_encoder == 'twins':
            self.encoder = TwinsFeatureEncoder(frozen=True)
            self.factor = 64
        elif args.feature_encoder == 'dav2':
            self.encoder = DepthAnythingFeature(model_name="vits", pretrained=True, lvl=-3)
            self.factor = 112
        elif args.feature_encoder == 'dinov3':
            self.encoder = DinoV3Feature(model_name="vits", lvl=-3)
            self.factor = 16
        else:
            raise ValueError(f"Unknown feature encoder: {args.feature_encoder}")

        self.pretrain_dim = self.encoder.output_dim
        self.fnet = ResNet18Deconv(3, self.pretrain_dim)
        self.iter_dim = MODEL_CONFIGS[args.iterative_module]['features']
        self.um4 = UpSampleMask4(self.iter_dim)
        self.um8 = UpSampleMask8(self.iter_dim)
        
        self.refine_net = VisionTransformer(args.iterative_module, self.iter_dim, patch_size=8)
        
        self.time_dim = 128
        self.refine_net_dfm = VisionTransformerDFM(feature_dim=self.iter_dim, time_dim=self.time_dim, num_modulators=4)


        self.fmap_conv = nn.Conv2d(self.pretrain_dim*2, self.iter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.hidden_conv = nn.Conv2d(self.iter_dim*2, self.iter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.warp_linear = nn.Conv2d(3*self.iter_dim+2, self.iter_dim, 1, 1, 0, bias=True)
        self.refine_transform = nn.Conv2d(self.iter_dim//2*3, self.iter_dim, 1, 1, 0, bias=True)
        
        self.upsample_weight_2x = nn.Sequential(
            nn.Conv2d(self.iter_dim, 2*self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.iter_dim, 4*9, 1, padding=0, bias=True)
        )
        
        self.upsample_weight_4x = nn.Sequential(
            nn.Conv2d(self.iter_dim, 2*self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.iter_dim, 16*9, 1, padding=0, bias=True) # 36 channels
        )
        
        self.flow_head = nn.Sequential(
            nn.Conv2d(self.iter_dim, 2*self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.iter_dim, 6, 1, padding=0, bias=True)
        )

        timesteps = 1000
        sampling_timesteps = 4
        self.recurr_itrs = 6  # Diffusion Iterations
        self.refine_iters = 6 # Refinement Iterations (args.iters)
        
        self.num_timesteps = int(timesteps)
        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        self.ddim_sampling_eta = 1.
        self.n_sc = 0.1
        self.scale = nn.Parameter(torch.ones(1) * 0.5, requires_grad=False)
        self.n_lambda = 0.2

        # [Schedules]
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas_cumprod', alphas_cumprod)
        self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

    def normalize_image(self, img):
        tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        return tf(img/255.0).contiguous()
    
    def upsample_data(self, flow, info, mask, factor=2):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, factor, factor, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(factor * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, factor*H, factor*W), up_info.reshape(N, C, factor*H, factor*W)

    def up_sample_flow8(self, flow, mask):
        B, _, H, W = flow.shape
        flow = torch.nn.functional.unfold(8 * flow, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
        flow = flow.reshape(B, 2, 9, 1, 1, H, W)
        mask = mask.reshape(B, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = torch.sum(flow * mask, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).contiguous()
        up_flow = up_flow.reshape(B, 2, H * 8, W * 8)
        return up_flow
    
    def _prepare_targets(self, flow_gt):
        noise = torch.randn(flow_gt.shape, device=flow_gt.device)
        b = flow_gt.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=flow_gt.device).long()
        
        x_start = flow_gt / self.norm_const_8x
        x_start = torch.clamp(x_start, min=-1.0, max=1.0) 
        x_start = x_start / self.scale
        x_t = self._q_sample(x_start=x_start, t=t, noise=noise)
        x_t = torch.clamp(x_t, min=-1, max=1)  
        x_t = x_t * self.n_sc  
        return x_t, noise, t
    
    def _q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def _train_dfm(self, fmap1_8x, fmap2_8x, flow_gt, net_8x, coords0_8x, coords1_8x):
        b, c, h, w = fmap1_8x.shape
        
        flow_gt_8x = F.interpolate(flow_gt, (h, w), mode='bilinear', align_corners=True) / 8.0
        x_t, noises, t = self._prepare_targets(flow_gt_8x)
        x_t = x_t * self.norm_const_8x
        coords1_8x = coords1_8x + x_t.float()

        flow_predictions = []
        info_predictions = []

        for ii in range(self.recurr_itrs):
            t_ii = (t - t / self.recurr_itrs * ii).int()
            
            flow_8x = coords1_8x - coords0_8x
            coords2 = (coords0_8x + flow_8x).detach()
            warp_8x = bilinear_sampler(fmap2_8x, coords2.permute(0, 2, 3, 1))
            
            refine_inp = self.warp_linear(torch.cat([fmap1_8x, warp_8x, net_8x, flow_8x], dim=1))
            
            refine_outs = self.refine_net_dfm(refine_inp, dfm_params=[t_ii, self.refine_net])
            
            net_8x = self.refine_transform(torch.cat([refine_outs['out'], net_8x], dim=1))
            
            flow_update = self.flow_head(net_8x)
            weight_update = .125 * self.um8(net_8x)
            
            coords1_8x = coords1_8x + flow_update[:, :2]
            info_8x = flow_update[:, 2:]
            
            curr_flow_8x = coords1_8x - coords0_8x
            
            flow_full, info_full = self.upsample_data(curr_flow_8x, info_8x, weight_update, factor=8)

            flow_predictions.append(flow_full)
            info_predictions.append(info_full)

        return flow_predictions, coords1_8x, net_8x, info_predictions

    @torch.no_grad()
    def _ddim_sample(self, fmap1_8x, fmap2_8x, net_8x, coords0_8x, coords1_8x):
        b, c, h, w = fmap1_8x.shape
        shape = (b, 2, h, w)
        
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        x_in = torch.randn(shape, device=fmap1_8x.device)
        
        flow_s = []
        pred_s = None # [net, weight, coords]
        
        for i_ddim, time_s in enumerate(time_pairs):
            time, time_next = time_s
            time_cond = torch.full((b,), time, device=fmap1_8x.device, dtype=torch.long)
            
            x_pred, inner_flow_s, pred_s = self._model_predictions_8x(x_in, time_cond, net_8x, fmap1_8x, fmap2_8x, coords0_8x, coords1_8x, i_ddim, pred_s)
            flow_s.extend(inner_flow_s) # Visualization용
            
            # DDIM Step
            if time_next < 0: continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            x_pred = x_pred * self.scale
            x_pred = torch.clamp(x_pred, min=-1*self.scale, max=self.scale)
            eps = (1 / (1 - alpha).sqrt()) * (x_in - alpha.sqrt() * x_pred)
            x_in = alpha_next.sqrt() * x_pred + (1 - alpha_next).sqrt() * eps
            
        net_8x, _, coords1_8x = pred_s
        return coords1_8x, net_8x, flow_s
    
    def up_sample_flow4(self, flow, mask):
        B, _, H, W = flow.shape
        flow = torch.nn.functional.unfold(4 * flow, kernel_size=[3, 3], stride=[1, 1], padding=[1, 1])
        flow = flow.reshape(B, 2, 9, 1, 1, H, W)
        mask = mask.reshape(B, 1, 9, 4, 4, H, W)
        mask = torch.softmax(mask, dim=2)
        up_flow = torch.sum(flow * mask, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3).contiguous()
        up_flow = up_flow.reshape(B, 2, H * 4, W * 4)
        return up_flow


    def _model_predictions_8x(self, x, t, net, fmap1, fmap2, coords0, coords1, i_ddim, pred_last=None):
        x_flow = torch.clamp(x, min=-1, max=1)
        x_flow = x_flow * self.n_sc
        x_flow = x_flow * self.norm_const_8x

        if pred_last:
            net, _, coords1 = pred_last
            x_flow = x_flow * self.n_lambda

        coords1 = coords1 + x_flow.float()

        flow_s = []
        for ii in range(self.recurr_itrs):
            t_ii = (t - (t - 0) / self.recurr_itrs * ii).int()

            flow_8x = coords1 - coords0
            coords2 = (coords0 + flow_8x).detach()
            warp_8x = bilinear_sampler(fmap2, coords2.permute(0, 2, 3, 1))
            
            refine_inp = self.warp_linear(torch.cat([fmap1, warp_8x, net, flow_8x], dim=1))

            with autocast(enabled=self.args.mixed_precision):
                itr = ii
                dfm_params = [t_ii, self.refine_net]
                refine_outs = self.refine_net_dfm(refine_inp, dfm_params=dfm_params)
                
                net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))
                
                flow_update = self.flow_head(net)
                up_mask = .125 * self.um8(net)
                
                delta_flow = flow_update[:, :2]

            coords1 = coords1 + delta_flow

            flow = coords1 - coords0
            flow_up = self.up_sample_flow8(flow, up_mask)

            flow_s.append(flow_up)

        flow = coords1 - coords0 
        x_pred = flow / self.norm_const_8x

        return x_pred, flow_s, [net, up_mask, coords1]

    def _refine_stage(self, fmap1_2x, fmap2_2x, net_2x, coords0_2x, coords1_2x):
        flow_predictions = []
        info_predictions = []
        
        for itr in range(self.refine_iters):
            flow_2x = coords1_2x - coords0_2x
            
            coords2 = (coords0_2x + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            
            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net_2x, flow_2x], dim=1))
            
            refine_outs = self.refine_net(refine_inp)
            
            net_2x = self.refine_transform(torch.cat([refine_outs['out'], net_2x], dim=1))
            
            flow_update = self.flow_head(net_2x)
            weight_update = .25 * self.upsample_weight_2x(net_2x)
            
            coords1_2x = coords1_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            
            curr_flow_2x = coords1_2x - coords0_2x
            flow_up, info_up = self.upsample_data(curr_flow_2x, info_2x, weight_update, factor=2)
            
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)
            
        return flow_predictions, info_predictions

    def forward(self, image1, image2, iters=None, flow_gt=None):
        if iters is None: iters = self.args.iters
        
        image1 = self.normalize_image(image1)
        image2 = self.normalize_image(image2)
        padder = Padder(image1.shape, factor=self.factor)
        image1 = padder.pad(image1)
        image2 = padder.pad(image2)
        N, _, H, W = image1.shape
        
        fmap1_pretrain = self.encoder(image1)
        fmap2_pretrain = self.encoder(image2)
        fmap1_img = self.fnet(image1)[0]
        fmap2_img = self.fnet(image2)[0]
        
        fmap1_2x = self.fmap_conv(torch.cat([fmap1_pretrain, fmap1_img], dim=1))
        fmap2_2x = self.fmap_conv(torch.cat([fmap2_pretrain, fmap2_img], dim=1))
        
        fmap1_4x = F.interpolate(fmap1_2x, scale_factor=0.5, mode='area')
        fmap2_4x = F.interpolate(fmap2_2x, scale_factor=0.5, mode='area')

        fmap1_8x = F.interpolate(fmap1_4x, scale_factor=0.5, mode='area')
        fmap2_8x = F.interpolate(fmap2_4x, scale_factor=0.5, mode='area')
        
        net_init = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        net_4x = F.avg_pool2d(net_init, 2, 2)
        net_8x = F.avg_pool2d(net_4x, 2, 2)

        coords0_8x = coords_grid(N, H//8, W//8, device=image1.device)
        coords1_8x = coords0_8x.clone()
        
        self.norm_const_8x = torch.as_tensor([W//8, H//8], dtype=torch.float, device=image1.device).view(1, 2, 1, 1)
        
        flow_list = []
        info_list = []

        if self.training:
            coords1_8x = coords1_8x.detach()
            flow_predictions_diff, coords1_8x, net_8x, info_predictions_diff = self._train_dfm(fmap1_8x, fmap2_8x, flow_gt, net_8x, coords0_8x, coords1_8x)
            flow_list.extend(flow_predictions_diff)
            info_list.extend(info_predictions_diff)
        else:
            coords1_8x, net_8x, _ = self._ddim_sample(fmap1_8x, fmap2_8x, net_8x, coords0_8x, coords1_8x)
        
        current_flow_8x = coords1_8x - coords0_8x
        flow_2x_init = F.interpolate(current_flow_8x * 4, scale_factor=4, mode='bilinear', align_corners=True)
        
        coords0_2x = coords_grid(N, H//2, W//2, device=image1.device)
        coords1_2x = coords0_2x + flow_2x_init
        
        net_2x = F.interpolate(net_8x, scale_factor=4, mode='bilinear', align_corners=True)
        
        flow_predictions_refine, info_predictions_refine = self._refine_stage(fmap1_2x, fmap2_2x, net_2x, coords0_2x, coords1_2x)
        flow_list.extend(flow_predictions_refine)
        info_list.extend(info_predictions_refine)
        
        final_output = {}
        
        for i in range(len(flow_list)):
            flow_list[i] = padder.unpad(flow_list[i])
            info_list[i] = padder.unpad(info_list[i])
            
        if flow_gt is not None:
            nf_predictions = []
            
            for i in range(len(flow_list)):
                pred_flow = flow_list[i]
                pred_info = info_list[i]

                weight = pred_info[:, :2]
                raw_b = pred_info[:, 2:]
                
                log_b = torch.zeros_like(raw_b)
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.args.var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.args.var_min, max=0)
                
                term2 = ((flow_gt - pred_flow).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                
                nf_predictions.append(nf_loss)
            
            output = {'flow': flow_list, 'info': info_list, 'nf': nf_predictions}
        
        else:
            output = {'flow': flow_list, 'info': info_list}
        
        return output
    