# -*- coding: utf-8 -*-
"""
CorrDiff SOTA Network (Production-Grade Final Version)
Optimized for Continental Scale (India Wide) on Multi-GPU H100 Systems
Features: Spatial Coordinates Injection, Aggressive CFG, PixelShuffle, SDPA, QDM, PhysicsGuide
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def _g(ch, mx=32):
    for g in range(min(mx, ch), 0, -1):
        if ch % g == 0: return g
    return 1

# ════════════════════════════════════════════════════════════════════════════
# SOTA UPSAMPLE / DOWNSAMPLE (Zero Bilinear Blurring)
# ════════════════════════════════════════════════════════════════════════════
class PixelShuffleUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels * (scale_factor ** 2), 3, padding=1)
        self.ps = nn.PixelShuffle(scale_factor)
        self.norm = nn.GroupNorm(_g(out_channels), out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.ps(self.conv(x))))

class StridedDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=scale_factor*2, stride=scale_factor, padding=scale_factor//2)
        self.norm = nn.GroupNorm(_g(out_channels), out_channels)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

# ════════════════════════════════════════════════════════════════════════════
# CORE ARCHITECTURE BLOCKS
# ════════════════════════════════════════════════════════════════════════════
class SpatialCoordinateInjection(nn.Module):
    """Injects static normalized lat/lon coordinates to force geographic awareness across India."""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        b, _, h, w = x.shape
        yg = torch.linspace(-1, 1, h, device=x.device).view(1, 1, h, 1).expand(b, 1, h, w)
        xg = torch.linspace(-1, 1, w, device=x.device).view(1, 1, 1, w).expand(b, 1, h, w)
        return torch.cat([x, yg, xg], dim=1)

class CoordConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=1, stride=1):
        super().__init__()
        self.coord_inject = SpatialCoordinateInjection()
        self.conv = nn.Conv2d(in_channels + 2, out_channels, kernel_size, padding=padding, stride=stride)

    def forward(self, x):
        return self.conv(self.coord_inject(x))

class SDPA_Attention(nn.Module):
    """H100 Native Flash Attention 2"""
    def __init__(self, ch, heads=8):
        super().__init__()
        self.heads = heads
        self.norm = nn.GroupNorm(_g(ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x)).view(B, 3, self.heads, C // self.heads, H * W)
        q, k, v = qkv.unbind(1)
        
        with torch.backends.cuda.sdp_kernel(enable_flash=True, enable_math=True, enable_mem_efficient=True):
            out = F.scaled_dot_product_attention(q.transpose(-1, -2), k.transpose(-1, -2), v.transpose(-1, -2))
            
        out = out.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(out)

class FourierFilter(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.complex_weight = nn.Parameter(torch.randn(channels, channels, 2) * 0.02)
        self.norm = nn.GroupNorm(_g(channels), channels)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_fft = torch.fft.rfft2(self.norm(x))
        weight = torch.view_as_complex(self.complex_weight)
        out_fft = torch.einsum('bchw,cd->bdhw', x_fft, weight)
        return x + self.proj(torch.fft.irfft2(out_fft, s=(H, W)))

class ResConv(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(_g(ic), ic), nn.SiLU(),
            nn.Conv2d(ic, oc, 3, padding=1),
            nn.GroupNorm(_g(oc), oc), nn.SiLU(),
            nn.Conv2d(oc, oc, 3, padding=1)
        )
        self.skip = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x):
        return self.skip(x) + self.net(x)

class FiLMResBlock(nn.Module):
    def __init__(self, ic, oc, ec, down=False, up=False, topo_channels=3, dropout=0.1):
        super().__init__()
        self.rs = StridedDownsample(ic, ic, 2) if down else (PixelShuffleUpsample(ic, ic, 2) if up else None)
        self.n1 = nn.GroupNorm(_g(ic), ic)
        self.c1 = nn.Conv2d(ic, oc, 3, padding=1)
        self.ep = nn.Linear(ec, oc)
        
        self.topo_proj = nn.Conv2d(topo_channels, oc * 2, 3, padding=1)
        nn.init.zeros_(self.topo_proj.weight); nn.init.zeros_(self.topo_proj.bias)
        
        self.n2 = nn.GroupNorm(_g(oc), oc)
        self.dropout = nn.Dropout(dropout)
        self.c2 = nn.Conv2d(oc, oc, 3, padding=1)
        self.sk = nn.Conv2d(ic, oc, 1) if ic != oc else nn.Identity()

    def forward(self, x, e, topo_res):
        if self.rs: x = self.rs(x)
        h = self.c1(F.silu(self.n1(x))) + self.ep(F.silu(e))[:, :, None, None]
        h_norm = self.n2(h)
        
        if topo_res is not None:
            gamma, beta = self.topo_proj(topo_res).chunk(2, dim=1)
            h_norm = h_norm * (1 + torch.tanh(gamma)) + beta
            
        return self.c2(F.silu(self.dropout(h_norm))) + self.sk(x)

# ════════════════════════════════════════════════════════════════════════════
# STAGE 1: REGRESSOR
# ════════════════════════════════════════════════════════════════════════════
class CorrDiffRegressor(nn.Module):
    def __init__(self, in_channels=2, out_channels=1, base_channels=64, channel_mult=(1, 2, 4), num_blocks=2, global_dim=2, topo_channels=3, d2m_channels=1, use_d2m=True):
        super().__init__()
        st, emb = base_channels, base_channels * 2
        self.use_d2m = use_d2m
        self.g_mlp = nn.Sequential(nn.Linear(global_dim, emb), nn.SiLU(), nn.Linear(emb, st)) if global_dim > 0 else None

        self.r_stem = nn.Sequential(
            PixelShuffleUpsample(in_channels, st, scale_factor=4),
            CoordConv2d(st, st, 3, padding=1)
        )
        self.t_stem = CoordConv2d(topo_channels, st, 3, padding=1)
        if use_d2m: self.d_stem = CoordConv2d(d2m_channels, st, 3, padding=1)

        self.r_enc, self.t_enc, self.r_dn, self.t_dn, self.sk_ch = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), []
        rc = tc = st

        for li, m in enumerate(channel_mult):
            oc = base_channels * m
            rb, tb = nn.ModuleList(), nn.ModuleList()
            for _ in range(num_blocks):
                rb.append(ResConv(rc, oc)); tb.append(ResConv(tc, oc))
                rc = tc = oc
            self.r_enc.append(rb); self.t_enc.append(tb)
            self.sk_ch.append(rc + tc)
            last = (li == len(channel_mult) - 1)
            self.r_dn.append(nn.Identity() if last else StridedDownsample(rc, rc, 2))
            self.t_dn.append(nn.Identity() if last else StridedDownsample(tc, tc, 2))

        bn = base_channels * channel_mult[-1]
        self.bn_proj = nn.Conv2d(rc + tc, bn, 1)
        self.bn_attn = SDPA_Attention(bn)
        self.bn_mid = ResConv(bn, bn)

        self.d_ups, self.d_blk = nn.ModuleList(), nn.ModuleList()
        dc = bn

        for li, m in reversed(list(enumerate(channel_mult))):
            oc = base_channels * m
            self.d_ups.append(nn.Identity() if li == len(channel_mult) - 1 else PixelShuffleUpsample(dc, dc, 2))
            blks = nn.ModuleList()
            ic2 = dc + self.sk_ch[li]
            for _ in range(num_blocks):
                blks.append(ResConv(ic2, oc)); ic2 = oc
            self.d_blk.append(blks); dc = oc

        self.out = nn.Sequential(nn.GroupNorm(_g(dc), dc), nn.SiLU(), nn.Conv2d(dc, out_channels, 3, padding=1))

    def forward(self, x, topo, global_features=None, d2m=None):
        r, t = self.r_stem(x), self.t_stem(topo)
        if self.use_d2m and d2m is not None: r = r + self.d_stem(d2m)
        if self.g_mlp and global_features is not None:
            gs = self.g_mlp(global_features)[:, :, None, None]
            r, t = r + gs, t + gs

        rs, ts = [], []
        for li in range(len(self.r_enc)):
            for rb, tb in zip(self.r_enc[li], self.t_enc[li]):
                r, t = rb(r), tb(t)
            rs.append(r); ts.append(t)
            r, t = self.r_dn[li](r), self.t_dn[li](t)

        d = self.bn_mid(self.bn_attn(self.bn_proj(torch.cat([r, t], 1))))

        for li, (up, blks) in enumerate(zip(self.d_ups, self.d_blk)):
            lv = len(self.r_enc) - 1 - li
            d = up(d)
            d = torch.cat([d, rs[lv], ts[lv]], 1)
            for b in blks: d = b(d)

        return self.out(d)

# ════════════════════════════════════════════════════════════════════════════
# STAGE 2: GENERATIVE UNET (CFG-Ready)
# ════════════════════════════════════════════════════════════════════════════
class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, base_channels=128, channel_mult=(1, 2, 2, 4), num_blocks=3, global_dim=2, topo_channels=3, use_d2m=True, d2m_channels=1, use_var_map=True, var_map_channels=1):
        super().__init__()
        ec = base_channels * 4
        self.use_d2m, self.use_var_map = use_d2m, use_var_map
        
        self.t_emb = nn.Sequential(nn.Linear(base_channels, ec), nn.SiLU(), nn.Linear(ec, ec))
        
        # Explicit NULL embeddings initialization for strict CFG stability
        self.null_global = nn.Parameter(torch.zeros(global_dim)) if global_dim else None
        self.g_mlp = nn.Sequential(nn.Linear(global_dim, ec), nn.SiLU()) if global_dim else None

        if use_d2m:
            self.d_stem = CoordConv2d(d2m_channels, base_channels, 3, padding=1)
            self.d_gate = nn.Parameter(torch.zeros(1))
        if use_var_map:
            self.v_stem = PixelShuffleUpsample(var_map_channels, base_channels, scale_factor=4)
            self.v_gate = nn.Parameter(torch.zeros(1))

        self.head = CoordConv2d(in_channels, base_channels, 3, padding=1)
        
        # ── Topography Pyramid ──
        self.topo_pyramid = nn.ModuleList([nn.Conv2d(topo_channels, topo_channels, 3, padding=1)])
        for _ in range(len(channel_mult)):
            self.topo_pyramid.append(StridedDownsample(topo_channels, topo_channels, 2))

        self.downs, self.ups, sk = nn.ModuleList(), nn.ModuleList(), []
        ch = base_channels

        for m in channel_mult:
            oc = base_channels * m
            for _ in range(num_blocks):
                self.downs.append(FiLMResBlock(ch, oc, ec, topo_channels=topo_channels))
                ch = oc; sk.append(ch)
            self.downs.append(FiLMResBlock(ch, ch, ec, down=True, topo_channels=topo_channels))
            sk.append(ch)

        self.m1 = FiLMResBlock(ch, ch, ec, topo_channels=topo_channels)
        self.ma = SDPA_Attention(ch)
        self.fft = FourierFilter(ch)
        self.m2 = FiLMResBlock(ch, ch, ec, topo_channels=topo_channels)

        for m in reversed(channel_mult):
            oc = base_channels * m
            self.ups.append(FiLMResBlock(ch + sk.pop(), oc, ec, up=True, topo_channels=topo_channels))
            ch = oc
            for _ in range(num_blocks):
                self.ups.append(FiLMResBlock(ch + sk.pop(), oc, ec, topo_channels=topo_channels))
                ch = oc

        self.out = nn.Sequential(nn.GroupNorm(_g(ch), ch), nn.SiLU(), nn.Conv2d(ch, out_channels, 3, padding=1))

    def _temb(self, t):
        half = self.t_emb[0].in_features // 2
        freq = torch.exp(torch.arange(half, device=t.device) * (-math.log(10000) / (half - 1)))
        e = t.unsqueeze(1) * freq.unsqueeze(0) * 2 * math.pi
        return self.t_emb(torch.cat([e.sin(), e.cos()], -1))

    def forward(self, x, t, topo, global_features=None, cfg_drop=None, d2m=None, var_map=None):
        emb = self._temb(t)
        
        if global_features is not None and self.g_mlp:
            gf = global_features.clone()
            if cfg_drop is not None and cfg_drop.any(): 
                gf[cfg_drop] = self.null_global
            emb = emb + self.g_mlp(gf)

        x_in, topo_in = x.clone(), topo.clone()
        if cfg_drop is not None and cfg_drop.any():
            x_in[cfg_drop, 1:] = 0.0 
            topo_in[cfg_drop] = 0.0

        t_pyr = [self.topo_pyramid[0](topo_in)]
        for i in range(1, len(self.topo_pyramid)): t_pyr.append(self.topo_pyramid[i](t_pyr[-1]))

        h = self.head(x_in)

        if self.use_d2m and d2m is not None:
            d_in = d2m.clone()
            if cfg_drop is not None and cfg_drop.any(): d_in[cfg_drop] = 0.0
            h = h + self.d_gate.tanh() * self.d_stem(d_in)

        if self.use_var_map and var_map is not None:
            v_in = var_map.clone()
            if cfg_drop is not None and cfg_drop.any(): v_in[cfg_drop] = 0.0
            h = h + self.v_gate.tanh() * self.v_stem(v_in)

        sk, pyr_idx = [h], 0
        for layer in self.downs:
            h = layer(h, emb, t_pyr[pyr_idx])
            sk.append(h)
            if layer.rs: pyr_idx += 1

        h = self.m1(h, emb, t_pyr[pyr_idx])
        h = self.ma(h)
        h = self.fft(h)
        h = self.m2(h, emb, t_pyr[pyr_idx])

        for layer in self.ups:
            if layer.rs: pyr_idx -= 1
            h = torch.cat([h, sk.pop()], 1)
            h = layer(h, emb, t_pyr[pyr_idx])

        return self.out(h)

# ════════════════════════════════════════════════════════════════════════════
# SOTA GENERATIVE WRAPPER
# ════════════════════════════════════════════════════════════════════════════
class FlowMatchingWrapper:
    def __init__(self, n_steps=10, cfg_scale=3.0, p_uncond=0.15):
        self.n_steps = n_steps
        self.cfg_scale = cfg_scale
        self.p_uncond = p_uncond 

    def get_train_sample(self, x1):
        B = x1.shape[0]
        x0 = torch.randn_like(x1)
        alpha = torch.distributions.Beta(1.5, 1.5).sample((B,)).to(x1.device)
        t = alpha.view(B, 1, 1, 1)
        
        x_t = (1 - t) * x0 + t * x1
        v_target = x1 - x0
        cfg_drop = torch.rand(B, device=x1.device) < self.p_uncond
        
        return x_t, alpha, v_target, cfg_drop

    @torch.no_grad()
    def sample(self, model, x_cond, topo, global_features=None, d2m=None, var_map=None, cfg_scale=None):
        cfg = cfg_scale or self.cfg_scale
        B, _, H, W = x_cond.shape
        x = torch.randn(B, 1, H, W, device=x_cond.device)
        dt = 1.0 / self.n_steps

        mask_cond = torch.zeros(B, dtype=torch.bool, device=x.device)
        mask_uncond = torch.ones(B, dtype=torch.bool, device=x.device)

        for i in range(self.n_steps):
            t_vec = torch.full((B,), i * dt, device=x.device)
            x_in = torch.cat([x, x_cond], dim=1)
            
            if cfg > 1.0:
                v_cond = model(x_in, t_vec, topo=topo, global_features=global_features, d2m=d2m, var_map=var_map, cfg_drop=mask_cond)
                v_uncond = model(x_in, t_vec, topo=topo, global_features=global_features, d2m=d2m, var_map=var_map, cfg_drop=mask_uncond)
                v = v_uncond + cfg * (v_cond - v_uncond)
            else:
                v = model(x_in, t_vec, topo=topo, global_features=global_features, d2m=d2m, var_map=var_map, cfg_drop=mask_cond)
                
            x = x + dt * v
            x = torch.clamp(x, min=-1.0, max=8.0) 

        return x

    @staticmethod
    def loss(v_pred, v_target):
        return F.mse_loss(v_pred, v_target)

# ════════════════════════════════════════════════════════════════════════════
# Embedded Physics Guide
# ════════════════════════════════════════════════════════════════════════════
class PhysicsGuide:
    DRY_THRESH_LOG = -10.0 

    @staticmethod
    def apply(pred_log, coarse_log, enforce_mass=True, enforce_dry=False):
        pred = pred_log.clamp(min=0.)
        if enforce_dry:
            dry_mask = coarse_log.mean(dim=[-2, -1], keepdim=True) < PhysicsGuide.DRY_THRESH_LOG
            pred = pred * (~dry_mask).float()
            
        if enforce_mass:
            pred_phys = torch.expm1(pred)
            coarse_phys = torch.expm1(coarse_log.clamp(0))
            
            coarse_up = coarse_phys.repeat_interleave(4, dim=-2).repeat_interleave(4, dim=-1)
            target_mean = coarse_up.mean(dim=[-2, -1], keepdim=True).clamp(1e-6)
            pred_mean = pred_phys.mean(dim=[-2, -1], keepdim=True).clamp(1e-6)
            
            scale = (target_mean / pred_mean).clamp(0.5, 3.0) 
            pred = torch.log1p((pred_phys * scale).clamp(0))
            
        return pred

# ════════════════════════════════════════════════════════════════════════════
# Quantile Delta Mapping (QDM)
# ════════════════════════════════════════════════════════════════════════════
class QDM:
    def __init__(self, n_quantiles=1000, clip_min=0.):
        self.n = n_quantiles
        self.clip = clip_min
        self.q = torch.linspace(0., 1., n_quantiles)
        self.cm = self.co = None
        self._fitted = False

    def fit(self, mp, op):
        mp_flat = mp.flatten().float().cpu()
        op_flat = op.flatten().float().cpu()
        max_elements = 15000000
        if op_flat.numel() > max_elements:
            indices  = torch.randperm(op_flat.numel())[:max_elements]
            op_sampled = op_flat[indices]; mp_sampled = mp_flat[indices]
        else:
            op_sampled = op_flat; mp_sampled = mp_flat
            
        self.cm = torch.quantile(mp_sampled, self.q)
        self.co = torch.quantile(op_sampled, self.q)
        self._fitted = True
        
        i95 = int(.95 * self.n); i99 = int(.99 * self.n)
        print(f"[QDM] P95 model={self.cm[i95]:.2f} obs={self.co[i95]:.2f} | P99 model={self.cm[i99]:.2f} obs={self.co[i99]:.2f}")

    @torch.no_grad()
    def apply(self, pred):
        assert self._fitted
        dev = pred.device
        cm  = self.cm.to(dev); co = self.co.to(dev)
        x   = pred.flatten().float()
        idx = torch.searchsorted(cm.contiguous(), x.contiguous()).clamp(0, self.n - 1)
        delta = (co[idx] - cm[idx]).clamp(-5.0, 7.0)
        return (x + delta).clamp(self.clip).reshape(pred.shape).to(pred.dtype)

    def save(self, p): torch.save({"cm": self.cm, "co": self.co, "n": self.n, "clip": self.clip}, p)

    @classmethod
    def load(cls, p):
        d = torch.load(p, map_location="cpu")
        q = cls(d["n"], d["clip"])
        q.cm, q.co, q._fitted = d["cm"], d["co"], True
        return q
