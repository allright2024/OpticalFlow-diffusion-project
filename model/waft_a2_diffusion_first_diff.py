import numpy as np
import torch
import math
import torchvision
import torch.nn as nn
import torch.nn.functional as F

from model.backbone.twins import TwinsFeatureEncoder
from model.backbone.waftv2_dav2 import DepthAnythingFeature
from model.backbone.dinov3 import DinoV3Feature
from model.backbone.vit import VisionTransformer, MODEL_CONFIGS, VisionTransformerDFM

from utils.utils import coords_grid, Padder, bilinear_sampler

import timm


def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def extract(a, t, x_shape):
    """extract the appropriate t index for a batch of indices"""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

def cosine_beta_schedule(timesteps, s=0.008):
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


class WAFTv2Diffusion(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
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

        self.refine_iters = 6
        self.pretrain_dim = self.encoder.output_dim
        self.fnet = ResNet18Deconv(3, self.pretrain_dim)
        self.iter_dim = MODEL_CONFIGS[args.iterative_module]['features']
        self.refine_net = VisionTransformer(args.iterative_module, self.iter_dim, patch_size=8)
        self.refine_net_dfm = VisionTransformerDFM()
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
            nn.Conv2d(2*self.iter_dim, 6, 1, padding=0, bias=True) # flow(2) + info(4)
        )

        self.diffusion = True
        if self.diffusion:
            time_emb_dim = self.iter_dim

            timesteps = 1000
            sampling_timesteps = 4
            recurr_itrs = 6
            print(' -- denoise steps: %d \n' % sampling_timesteps)
            print(' -- recurrent iterations: %d \n' % recurr_itrs)

            self.ddim_n = sampling_timesteps
            self.recurr_itrs = recurr_itrs
            self.n_sc = 0.1
            self.scale = nn.Parameter(torch.ones(1) * 0.5, requires_grad=False)
            self.n_lambda = 0.2

            self.objective = 'pred_x0'
            betas = cosine_beta_schedule(timesteps)
            alphas = 1. - betas
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
            timesteps, = betas.shape
            self.num_timesteps = int(timesteps)

            self.sampling_timesteps = default(sampling_timesteps, timesteps)
            assert self.sampling_timesteps <= timesteps
            self.is_ddim_sampling = self.sampling_timesteps < timesteps
            self.ddim_sampling_eta = 1.
            self.self_condition = False

            self.register_buffer('betas', betas)
            self.register_buffer('alphas_cumprod', alphas_cumprod)
            self.register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)
            self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
            self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
            self.register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
            self.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
            self.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

            posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
            self.register_buffer('posterior_variance', posterior_variance)
            self.register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
            self.register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
            self.register_buffer('posterior_mean_coef2',
                                 (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

    def upsample_data(self, flow, info, mask):
        """ Upsample flow and info field """
        N, _, H, W = flow.shape

        mask = mask.view(N, 1, 9, 2, 2, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(2 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)
        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        up_flow = up_flow.reshape(N, 2, 2*H, 2*W)

        up_info = F.unfold(info, [3, 3], padding=1)
        up_info = up_info.view(N, 4, 9, 1, 1, H, W)
        up_info = torch.sum(mask * up_info, dim=2)
        up_info = up_info.permute(0, 1, 4, 2, 5, 3)
        up_info = up_info.reshape(N, 4, 2*H, 2*W)

        return up_flow, up_info

    def normalize_image(self, img):
        '''
        @img: (B,C,H,W) in range 0-255, RGB order
        '''
        tf = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225], inplace=False)
        return tf(img/255.0).contiguous()

    def forward(self, image1, image2, iters=None, flow_gt=None):
        """ Estimate optical flow between pair of frames """

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

        net = self.hidden_conv(torch.cat([fmap1_2x, fmap2_2x], dim=1))

        coords0 = coords_grid(N, H//2, W//2, device=image1.device)
        coords1 = coords_grid(N, H//2, W//2, device=image1.device)

        flow_list = []
        info_list = []

        if self.diffusion:
            self.device = image1.device
            h, w = fmap1_2x.shape[-2:]
            self.norm_const = torch.as_tensor([w, h], dtype=torch.float, device=self.device).view(1, 2, 1, 1)

            inp_data = (fmap1_2x, fmap2_2x)

            if self.training:
                coords1 = coords1.detach()
                flow_up_s, info_up_s, coords1, net = self._train_dfm(
                    fmap1_2x.shape, flow_gt, net, inp_data, coords0, coords1
                )
                flow_list = flow_up_s
                info_list = info_up_s
            else:
                coords1, net, flow_up_s, info_up_s = self._ddim_sample(
                    fmap1_2x.shape, net, inp_data, coords0, coords1
                )
                flow_list = flow_up_s
                info_list = info_up_s
        else:
            raise NotImplementedError("Diffusion이 False인 경우는 현재 구현되지 않았습니다.")

        flow_predictions_refine, info_predictions_refine = self._refine_stage(fmap1_2x, fmap2_2x, net, coords1, coords0)
        flow_list = flow_predictions_refine
        info_list = info_predictions_refine

        flow_predictions = [padder.unpad(flow) for flow in flow_list]
        info_predictions = [padder.unpad(info) for info in info_list]

        if flow_gt is not None:
            nf_predictions = []
            for i in range(len(info_predictions)):
                raw_b = info_predictions[i][:, 2:]
                log_b = torch.zeros_like(raw_b)
                weight = info_predictions[i][:, :2]
                log_b[:, 0] = torch.clamp(raw_b[:, 0], min=0, max=self.args.var_max)
                log_b[:, 1] = torch.clamp(raw_b[:, 1], min=self.args.var_min, max=0)
                term2 = ((flow_gt - flow_predictions[i]).abs().unsqueeze(2)) * (torch.exp(-log_b).unsqueeze(1))
                term1 = weight - math.log(2) - log_b
                nf_loss = torch.logsumexp(weight, dim=1, keepdim=True) - torch.logsumexp(term1.unsqueeze(1) - term2, dim=2)
                nf_predictions.append(nf_loss)
            output = {'flow': flow_predictions, 'info': info_predictions, 'nf': nf_predictions}
        else:
            output = {'flow': flow_predictions, 'info': info_predictions}

        return output


    def _prepare_targets(self, flow_gt):
        noise = torch.randn(flow_gt.shape, device=self.device)
        t = torch.randint(0, self.num_timesteps, (1,), device=self.device).long()
        x_start = flow_gt / self.norm_const
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

    def _train_dfm(self, feat_shape, flow_gt, net, inp_data, coords0, coords1):
        b, c, h, w = feat_shape
        if len(flow_gt.shape) == 3:
            flow_gt = flow_gt.unsqueeze(0)

        flow_gt_sp = F.interpolate(flow_gt, (h, w), mode='bilinear', align_corners=True) / 2.0

        x_t, noises, t = self._prepare_targets(flow_gt_sp)
        x_t = x_t * self.norm_const
        coords1 = coords1 + x_t.float()

        flow_up_s = []
        info_up_s = []
        fmap1_2x, fmap2_2x = inp_data

        for ii in range(self.recurr_itrs):
            t_ii = (t - t / self.recurr_itrs * ii).int()

            coords1 = coords1.detach()
            flow_2x = coords1 - coords0

            coords2 = coords1.permute(0, 2, 3, 1)
            warp_2x = bilinear_sampler(fmap2_2x, coords2)

            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))

            refine_outs = self.refine_net_dfm(refine_inp, dfm_params=[t_ii, self.refine_net])

            net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))

            flow_head_out = self.flow_head(net)
            delta_flow = flow_head_out[:, :2]
            info_2x = flow_head_out[:, 2:]

            coords1 = coords1 + delta_flow

            weight_update = .25 * self.upsample_weight(net)

            flow_up, info_up = self.upsample_data(coords1 - coords0, info_2x, weight_update)
            flow_up_s.append(flow_up)
            info_up_s.append(info_up)

        return flow_up_s, info_up_s, coords1, net

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
            weight_update = .25 * self.upsample_weight(net_2x)
            
            coords1_2x = coords1_2x + flow_update[:, :2]
            info_2x = flow_update[:, 2:]
            
            curr_flow_2x = coords1_2x - coords0_2x
            flow_up, info_up = self.upsample_data(curr_flow_2x, info_2x, weight_update)
            
            flow_predictions.append(flow_up)
            info_predictions.append(info_up)
            
        return flow_predictions, info_predictions

    @torch.no_grad()
    def _ddim_sample(self, feat_shape, net, inp_data, coords0, coords1_init, clip_denoised=True):
        batch, c, h, w = feat_shape
        shape = (batch, 2, h, w)
        total_timesteps, sampling_timesteps, eta, objective = self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        x_in = torch.randn(shape, device=self.device)

        flow_s = []
        info_s = []
        pred_s = None
        for i_ddim, time_s in enumerate(time_pairs):
            time, time_next = time_s
            time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
            t_next = torch.full((batch,), time_next, device=self.device, dtype=torch.long)

            x_pred, inner_flow_s, inner_info_s, pred_s = self._model_predictions(
                x_in, time_cond, net, inp_data, coords0, coords1_init, i_ddim, pred_s, t_next
            )
            flow_s = flow_s + inner_flow_s
            info_s = info_s + inner_info_s

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            x_t = x_in

            x_pred = x_pred * self.scale
            x_pred = torch.clamp(x_pred, min=-1 * self.scale, max=self.scale)

            eps = (1 / (1 - alpha).sqrt()) * (x_t - alpha.sqrt() * x_pred)
            x_next = alpha_next.sqrt() * x_pred + (1 - alpha_next).sqrt() * eps

            x_in = x_next

        net, _, coords1 = pred_s
        return coords1, net, flow_s, info_s

    def _model_predictions(self, x, t, net, inp_data, coords0, coords1, i_ddim, pred_last=None, t_next=None):

        x_flow = torch.clamp(x, min=-1, max=1)
        x_flow = x_flow * self.n_sc
        x_flow = x_flow * self.norm_const

        if pred_last:
            net, _, coords1 = pred_last
            x_flow = x_flow * self.n_lambda

        coords1 = coords1 + x_flow.float()

        flow_s = []
        info_s = []
        fmap1_2x, fmap2_2x = inp_data

        for ii in range(self.recurr_itrs):
            t_ii = (t - (t - 0) / self.recurr_itrs * ii).int()

            flow_2x = coords1 - coords0

            coords2 = coords1.permute(0, 2, 3, 1)
            warp_2x = bilinear_sampler(fmap2_2x, coords2)

            refine_inp = self.warp_linear(torch.cat([fmap1_2x, warp_2x, net, flow_2x], dim=1))

            refine_outs = self.refine_net(refine_inp, time_emb)

            net = self.refine_transform(torch.cat([refine_outs['out'], net], dim=1))

            flow_head_out = self.flow_head(net)
            delta_flow = flow_head_out[:, :2]
            info_2x = flow_head_out[:, 2:]

            coords1 = coords1 + delta_flow


            weight_update = .25 * self.upsample_weight(net)

            flow_up, info_up = self.upsample_data(coords1 - coords0, info_2x, weight_update)
            flow_s.append(flow_up)
            info_s.append(info_up)

        flow = coords1 - coords0
        x_pred = flow / self.norm_const

        return x_pred, flow_s, info_s, [net, weight_update, coords1]
