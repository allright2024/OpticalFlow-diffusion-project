import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision
import timm

from model.backbone.twins import TwinsFeatureEncoder
from model.backbone.waftv2_dav2 import DepthAnythingFeature
from model.backbone.dinov3 import DinoV3Feature
from model.backbone.vit import VisionTransformer, MODEL_CONFIGS, VisionTransformerDFM
from utils.utils import coords_grid, Padder, bilinear_sampler


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

class WAFTv2_FlowDiffuser_TwoStage(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.um4 = UpSampleMask4(128)
    
        
        if args.feature_encoder == 'twins':
            self.encoder = TwinsFeatureEncoder(frozen=True)
            self.factor = 32
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
        
        self.refine_net = VisionTransformer(args.iterative_module, self.iter_dim, patch_size=8)
        
        self.time_dim = 128
        self.refine_net_dfm = VisionTransformerDFM(feature_dim=self.iter_dim, time_dim=self.time_dim, num_modulators=4)

        self.fmap_conv = nn.Conv2d(self.pretrain_dim*2, self.iter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.hidden_conv = nn.Conv2d(self.iter_dim*2, self.iter_dim, kernel_size=1, stride=1, padding=0, bias=True)
        self.warp_linear = nn.Conv2d(3*self.iter_dim+2, self.iter_dim, 1, 1, 0, bias=True)
        self.refine_transform = nn.Conv2d(self.iter_dim//2*3, self.iter_dim, 1, 1, 0, bias=True)
        
        self.upsample_weight = nn.Sequential(
            nn.Conv2d(self.iter_dim, 2*self.iter_dim, 3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(2*self.iter_dim, 4*9, 1, padding=0, bias=True)
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
    
    def upsample_data(self, flow, info, mask):
        N, C, H, W = info.shape
        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, C, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        
        return up_flow.reshape(N, 2, 2*H, 2*W), up_info.reshape(N, C, 2*H, 2*W)
    
    def _prepare_targets(self, flow_gt):
        noise = torch.randn(flow_gt.shape, device=flow_gt.device)
        t = torch.randint(0, self.num_timesteps, (1,), device=flow_gt.device).long()

        x_start = flow_gt / self.norm_const_4x
        x_start = x_start * self.scale 
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

    # =========================================================
    # Stage 1: Diffusion Process (at 1/4 Resolution)
    # =========================================================
    def _train_dfm(self, fmap1_4x, fmap2_4x, flow_gt, net_4x, coords0_4x, coords1_4x):
        b, c, h, w = fmap1_4x.shape
        
        flow_gt_4x = F.interpolate(flow_gt, (h, w), mode='bilinear', align_corners=True) / 4.0
        x_t, noises, t = self._prepare_targets(flow_gt_4x)
        x_t = x_t * self.norm_const_4x
        coords1_4x = coords1_4x + x_t.float()

        flow_predictions = []
        info_predictions = []

        for ii in range(self.recurr_itrs):
            t_ii = (t - t / self.recurr_itrs * ii).int()
            
            flow_4x = coords1_4x - coords0_4x
            coords2 = (coords0_4x + flow_4x).detach()
            warp_4x = bilinear_sampler(fmap2_4x, coords2.permute(0, 2, 3, 1))
            
            refine_inp = self.warp_linear(torch.cat([fmap1_4x, warp_4x, net_4x, flow_4x], dim=1))
    
            refine_outs = self.refine_net_dfm(refine_inp, dfm_params=[t_ii, self.refine_net])
            
            net_4x = self.refine_transform(torch.cat([refine_outs['out'], net_4x], dim=1))
            
            flow_update = self.flow_head(net_4x)
            weight_update = .25 * self.upsample_weight(net_4x)
            
            coords1_4x = coords1_4x + flow_update[:, :2]
            info_4x = flow_update[:, 2:]
            
            curr_flow_4x = coords1_4x - coords0_4x
            
            
            
            flow_up_4x, info_up_4x = self.upsample_data(curr_flow_4x, info_4x, weight_update) # 1/4 -> 1/2
            
            flow_full = F.interpolate(flow_up_4x * 2, scale_factor=2, mode='bilinear', align_corners=True)
            info_full = F.interpolate(info_up_4x, scale_factor=2, mode='bilinear', align_corners=True)
            
            flow_predictions.append(flow_full)
            info_predictions.append(info_full)

        return flow_predictions, coords1_4x, net_4x, info_predictions

    @torch.no_grad()
    def _ddim_sample(self, fmap1_4x, fmap2_4x, net_4x, coords0_4x, coords1_4x):
        b, c, h, w = fmap1_4x.shape
        shape = (b, 2, h, w)
        
        times = torch.linspace(-1, self.num_timesteps - 1, steps=self.sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))
        
        x_in = torch.randn(shape, device=fmap1_4x.device)
        
        flow_s = []
        pred_s = None # [net, weight, coords]
        
        for i_ddim, time_s in enumerate(time_pairs):
            time, time_next = time_s
            time_cond = torch.full((b,), time, device=fmap1_4x.device, dtype=torch.long)
            
            x_pred, inner_flow_s, pred_s = self._model_predictions_4x(x_in, time_cond, net_4x, fmap1_4x, fmap2_4x, coords0_4x, i_ddim, pred_s)
            flow_s.extend(inner_flow_s) # Visualization용
            
            # DDIM Step
            if time_next < 0: continue
            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            x_pred = x_pred * self.scale
            x_pred = torch.clamp(x_pred, min=-1*self.scale, max=self.scale)
            eps = (1 / (1 - alpha).sqrt()) * (x_in - alpha.sqrt() * x_pred)
            x_in = alpha_next.sqrt() * x_pred + (1 - alpha_next).sqrt() * eps
            
        net_4x, _, coords1_4x = pred_s
        return coords1_4x, net_4x, flow_s
    
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


    def _model_predictions_4x(self, x, t, net, inp8, coords0, coords1, i_ddim, pred_last=None, t_next=None):
        x_flow = torch.clamp(x, min=-1, max=1)
        x_flow = x_flow * self.n_sc
        x_flow = x_flow * self.norm_const_4x

        if pred_last:
            net, _, coords1 = pred_last
            x_flow = x_flow * self.n_lambda

        coords1 = coords1 + x_flow.float()

        flow_s = []
        for ii in range(self.recurr_itrs):
            t_ii = (t - (t - 0) / self.recurr_itrs * ii).int()

            corr = self.corr_fn(coords1)
            flow = coords1 - coords0

            with autocast(enabled=self.args.mixed_precision):
                itr = ii
                first_step = False if itr != 0 else True
                dfm_params = [t_ii, self.refine_net]
                net, delta_flow = self.refine_net_dfm(net, inp8, corr, flow, itr, first_step=first_step, dfm_params=dfm_params)
                up_mask = self.um4(net)

            coords1 = coords1 + delta_flow

            flow = coords1 - coords0
            flow_up = self.up_sample_flow4(flow, up_mask)

            flow_s.append(flow_up)

        flow = coords1 - coords0 
        x_pred = flow / self.norm_const_4x

        return x_pred, flow_s, [net, up_mask, coords1] 
    # =========================================================
    # Stage 2: Refinement Process (at 1/2 Resolution)
    # =========================================================
    def _refine_stage(self, fmap1_2x, fmap2_2x, net_2x, coords0_2x, coords1_2x):
        flow_predictions = []
        info_predictions = []
        
        # Standard WAFTv2 Iterative Refinement
        for itr in range(self.refine_iters):
            flow_2x = coords1_2x - coords0_2x
            
            # Warp
            coords2 = (coords0_2x + flow_2x).detach()
            warp_2x = bilinear_sampler(fmap2_2x, coords2.permute(0, 2, 3, 1))
            
            # ViT Input
            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net_2x, flow_2x], dim=1))
            
            # [Standard Call] No Time Injection
            refine_outs = self.refine_net(refine_inp)
            
            # Update Net
            net_2x = self.refine_transform(torch.cat([refine_outs['out'], net_2x], dim=1))
            
            # Predict
            flow_update = self.flow_head(net_2x)
            weight_update = .25 * self.upsample_weight(net_2x)
            
            coords1_2x = coords1_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            
            # Upsample (1/2 -> Full)
            curr_flow_2x = coords1_2x - coords0_2x
            flow_up, info_up = self.upsample_data(curr_flow_2x, info_2x, weight_update)
            
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
        
        # Init Net & Coords
        # Net for 4x
        net_init = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))
        net_4x = F.avg_pool2d(net_init, 2, 2)
        coords0_4x = coords_grid(N, H//4, W//4, device=image1.device)
        coords1_4x = coords0_4x.clone()
        
        # Norm Const for 4x
        self.norm_const_4x = torch.as_tensor([W//4, H//4], dtype=torch.float, device=image1.device).view(1, 2, 1, 1)
        
        flow_list = []
        info_list = []

        # =================================================
        # Stage 1: Diffusion (1/4 Res)
        # =================================================
        if self.training:
            coords1_4x = coords1_4x.detach()
            flow_predictions_diff, coords1_4x, net_4x, info_predictions_diff = self._train_dfm(fmap1_4x, fmap2_4x, flow_gt, net_4x, coords0_4x, coords1_4x)
            flow_list.extend(flow_predictions_diff)
            info_list.extend(info_predictions_diff)
        else:
            coords1_4x, net_4x, _ = self._ddim_sample(fmap1_4x, fmap2_4x, net_4x, coords0_4x, coords1_4x)
        
        # =================================================
        # Transition: 1/4 -> 1/2 Res (FlowDiffuser Logic)
        # =================================================
        # 1. Flow Upsample: (Coords_4x - Coords_4x_Init) * 2 -> Flow_2x
        current_flow_4x = coords1_4x - coords0_4x
        flow_2x_init = F.interpolate(current_flow_4x * 2, scale_factor=2, mode='bilinear', align_corners=True)
        
        # 2. Coords 2x Init
        coords0_2x = coords_grid(N, H//2, W//2, device=image1.device)
        coords1_2x = coords0_2x + flow_2x_init
        
        # 3. Net Upsample
        net_2x = F.interpolate(net_4x, scale_factor=2, mode='bilinear', align_corners=True)
        
        # =================================================
        # Stage 2: Refinement (1/2 Res)
        # =================================================
        # 여기서 coords1_2x는 Diffusion 결과로 초기화된 상태
        flow_predictions_refine, info_predictions_refine = self._refine_stage(fmap1_2x, fmap2_2x, net_2x, coords0_2x, coords1_2x)
        flow_list.extend(flow_predictions_refine)
        info_list.extend(info_predictions_refine)
        
        # =================================================
        # Post-processing (Loss or Output)
        # =================================================
        # ... (Unpadding and Output formatting logic same as original WAFTv2) ...
        
        # (For brevity, reusing the final formatting logic)
        final_output = {}
        
        # Unpad all predictions
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
    