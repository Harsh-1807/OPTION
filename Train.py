# -*- coding: utf-8 -*-
"""
Unified Two-Stage CorrDiff Trainer — SOTA Production
──────────────────────────────────────────────────────
Stage 1  Regressor  : Charbonnier + Spectral + Extreme-Weighted MAE + Mass Conservation
Stage 2  Diffusion  : EDM residual + Hybrid SNR loss + BF16 + EMA + QDM + 4×H100 DDP

torchrun --nproc_per_node=4 train.py [--stage {all,regressor,unet}]
"""

import os, math, time, argparse, traceback, warnings
from copy import deepcopy
warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

from Dataset import UpscaleDataset
# FlowMatchingWrapper renamed from FlowMatching in the SOTA Network.py
from Network import CorrDiffRegressor, UNet, FlowMatchingWrapper as FlowMatching, PhysicsGuide, QDM

# ════════════════════════════════════════════════════════════════════════════
# PATHS
# ════════════════════════════════════════════════════════════════════════════
RF_PATH    = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/RF_1975to2023.nc"
ORO_PATH   = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/oro.nc"
D2M_PATH   = "/lustre/home/hpc/bipink/VIT_Pune_New/Harsh/Diffusion_Downscaling/data/era5_aligned_to_rf.nc"
CKPT_DIR   = "checkpoints/v13_unified/"
REG_BEST   = os.path.join(CKPT_DIR, "regressor_best.pth")
REG_LATEST = os.path.join(CKPT_DIR, "regressor_latest.pth")
UNT_BEST   = os.path.join(CKPT_DIR, "unet_best.pth")
UNT_LATEST = os.path.join(CKPT_DIR, "unet_latest.pth")
QDM_BEST   = os.path.join(CKPT_DIR, "qdm_best.pth")
QDM_LATEST = os.path.join(CKPT_DIR, "qdm_latest.pth")

# ════════════════════════════════════════════════════════════════════════════
# HYPER-PARAMETERS
# ════════════════════════════════════════════════════════════════════════════
# Stage 1 – Regressor
REG_EPOCHS   = 300
REG_BATCH    = 32
REG_LR       = 3e-4
REG_PATIENCE = 60

# Stage 2 – UNet
UNT_EPOCHS   = 1500
UNT_BATCH    = 16
UNT_LR       = 1e-4
UNT_MIN_LR   = 5e-6
UNT_ACCUM    = 2
UNT_PATIENCE = 200

# Shared
WEIGHT_DECAY = 1e-3
GRAD_CLIP    = 1.0
EMA_DECAY    = 0.9995
DS_FACTOR    = 4
T_COND       = 5
PRECIP_CH    = 0

# Architecture  (must match Network.py UNet __init__ params)
BASE_CH      = 256
CHANNEL_MULT = (1, 2, 2, 4)
NRB          = 3            # num_blocks in UNet (was num_res_blocks in old trainer — FIXED)
TOPO_CH      = 3
GLOBAL_DIM   = 2
UNET_IN_CH   = 1 + 1 + T_COND   # noisy_residual | mu | tc_frames

# EDM / Sampling
SIGMA_DATA   = 0.1925
EDM_STEPS    = 15
CFG_SCALE    = 1.5
P_CFG_DROP   = 0.10
N_ENS        = 2


# ════════════════════════════════════════════════════════════════════════════
# EDM NOISE SCHEDULE
# ════════════════════════════════════════════════════════════════════════════
def build_edm_schedule(n=EDM_STEPS, smin=0.002, sdata=SIGMA_DATA, rho=7.0):
    smax = 2.0 * sdata
    t    = torch.arange(n, dtype=torch.float32) / max(n - 1, 1)
    return (smax**(1/rho) + t*(smin**(1/rho) - smax**(1/rho)))**rho

_EDM = build_edm_schedule()


# ════════════════════════════════════════════════════════════════════════════
# EMA
# ════════════════════════════════════════════════════════════════════════════
class EMA:
    def __init__(self, model, decay=EMA_DECAY):
        self.decay  = decay
        self.shadow = {k: v.clone().detach().float() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model):
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.float(), alpha=1 - self.decay)

    def apply_to(self, model):
        model.load_state_dict({k: v.to(next(model.parameters()).device)
                               for k, v in self.shadow.items()})

    @staticmethod
    def save_state(model):  return deepcopy(model.state_dict())
    @staticmethod
    def restore(model, sd): model.load_state_dict(sd)


# ════════════════════════════════════════════════════════════════════════════
# DDP UTILITIES
# ════════════════════════════════════════════════════════════════════════════
def setup_ddp():
    rank  = int(os.environ.get("RANK",       0))
    ws    = int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    if ws > 1:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local)
    torch.backends.cudnn.benchmark         = True
    torch.backends.cuda.matmul.allow_tf32  = True
    torch.backends.cudnn.allow_tf32        = True
    dev = torch.device(f"cuda:{local}" if torch.cuda.is_available() else "cpu")
    return rank, ws, local, dev


def broadcast_scalar(val, dev, src=0):
    """Sync a Python int/float from rank 0 to all ranks."""
    t = torch.tensor(float(val), device=dev)
    if dist.is_initialized(): dist.broadcast(t, src=src)
    return type(val)(t.item())


# ════════════════════════════════════════════════════════════════════════════
# DATA / TOPO HELPERS
# ════════════════════════════════════════════════════════════════════════════
def expand_topo(topo_1ch: torch.Tensor) -> torch.Tensor:
    """[B,1,H,W] elevation → [B,3,H,W]  (elevation, slope, aspect) all globally normalised."""
    kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32,
                      device=topo_1ch.device).view(1,1,3,3)
    ky = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32,
                      device=topo_1ch.device).view(1,1,3,3)
    e      = topo_1ch.float()
    dx     = F.conv2d(e, kx, padding=1)
    dy     = F.conv2d(e, ky, padding=1)
    slope  = torch.sqrt(dx**2 + dy**2 + 1e-8)
    aspect = torch.atan2(dy, dx)
    gnorm  = lambda t, lo, hi: 2*(t - lo)/(hi - lo + 1e-8) - 1
    return torch.cat([gnorm(e, 0., 8600.), gnorm(slope, 0., 1.5), aspect / math.pi], dim=1)


def build_coarse_input(coarse: torch.Tensor, var_map: torch.Tensor) -> torch.Tensor:
    return torch.cat([coarse, F.adaptive_avg_pool2d(var_map, coarse.shape[-2:])], dim=1)


def get_batch_tensors(b: dict, dev: torch.device):
    """
    Returns raw tensors — augmentation and tp/xi construction happen after.
    tc_frames is already fine-res (repeat_interleave in Dataset.__getitem__).
    DO NOT re-interpolate here; bilinear upsampling would blur coarse pixels.
    """
    fp      = b["fine"].to(dev, non_blocking=True)[:, PRECIP_CH:PRECIP_CH+1]
    topo_1  = b["topo"].to(dev, non_blocking=True)           # [B,1,H,W]
    xi_raw  = b["coarse"].to(dev, non_blocking=True)         # [B,1,Hc,Wc]
    var_map = b["var_map"].to(dev, non_blocking=True)        # [B,1,H,W]
    d2m     = b["d2m"].to(dev, non_blocking=True) if "d2m" in b else None
    gf      = torch.stack([b["doy"], b["hour"]], 1).float().to(dev, non_blocking=True)
    tc      = b["tc_frames"].to(dev, non_blocking=True)      # [B,T_COND,H,W]
    return fp, topo_1, xi_raw, var_map, d2m, gf, tc


def random_flip(tensors_hw, tensors_hw_optional, dim):
    """In-place flip augmentation shared across all spatial tensors."""
    out  = [t.flip(dim).contiguous() for t in tensors_hw]
    outo = [t.flip(dim).contiguous() if t is not None else None for t in tensors_hw_optional]
    return out, outo


# ════════════════════════════════════════════════════════════════════════════
# LOSS — STAGE 1 : REGRESSOR
# ════════════════════════════════════════════════════════════════════════════
def regressor_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Charbonnier  — smooth L1, heavy-tail robust (beats MSE on log-precip)
    Spectral     — frequency-domain coherence (mesoscale pattern fidelity)
    Extreme-MAE  — intensity² weighting so model cares about heavy rain
    Mass         — total water volume conservation (physical prior)
    """
    p, t = pred.float(), target.float()

    # 1. Charbonnier
    charb = torch.sqrt((p - t)**2 + 1e-6).mean()

    # 2. Spectral (normalised L1 on amplitude spectrum)
    amp_p = torch.fft.rfft2(p).abs()
    amp_t = torch.fft.rfft2(t).abs()
    spec  = F.l1_loss(amp_p, amp_t) / (amp_t.mean() + 1e-8)

    # 3. Extreme-weighted MAE  (α=2 → exponential focus on heavy rain)
    w       = t.clamp(min=0.)**2 + 1.0
    extreme = (torch.abs(p - t) * w).mean()

    # 4. Mass conservation (physical space, not log-space)
    mass = torch.abs(
        torch.expm1(p.clamp(0)).mean([-1, -2]) -
        torch.expm1(t.clamp(0)).mean([-1, -2])
    ).mean()

    return charb + 0.08*spec + 0.40*extreme + 0.15*mass


# ════════════════════════════════════════════════════════════════════════════
# LOSS — STAGE 2 : DIFFUSION UNET
# ════════════════════════════════════════════════════════════════════════════
def _pcc_log_cosh(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Log-cosh PCC push — steeper gradient near r=1 to overcome the 0.85 plateau."""
    B   = pred.shape[0]
    p   = pred.float().view(B, -1);   t = target.float().view(B, -1)
    pc  = p - p.mean(1, keepdim=True); tc = t - t.mean(1, keepdim=True)
    r   = (pc*tc).sum(1) / (torch.sqrt((pc**2).sum(1)*(tc**2).sum(1)) + 1e-8)
    var_pen = torch.abs(
        torch.log((pc**2).sum(1) + 1e-6) - torch.log((tc**2).sum(1) + 1e-6)
    ).mean()
    return torch.log(torch.cosh((1. - r.mean()) * 5.)) + 0.1*var_pen


def hybrid_sigma_loss(pred: torch.Tensor, target: torch.Tensor,
                      sigma: torch.Tensor, epoch: int = 0) -> torch.Tensor:
    """
    SNR-weighted multi-objective loss for EDM residual diffusion.

    Components
    ──────────
    spatial   : intensity-weighted MAE  (SNR scaled)
    spectral  : frequency coherence
    pcc       : log-cosh PCC push       (warms up after epoch 10)
    dry       : false-positive drizzle penalty
    mass      : water volume conservation
    """
    p, t = pred.float(), target.float()
    snr  = SIGMA_DATA**2 / (sigma**2 + 1e-6)          # [B,1,1,1]
    snr_w = snr.clamp(max=5.)

    # 1. Intensity-weighted spatial MAE
    pw      = t.clamp(min=0.)**1.5 + 1.0
    spatial = (torch.abs(p - t) * pw * snr_w).mean()

    # 2. Spectral coherence
    spectral = torch.abs(torch.fft.rfft2(p).abs() - torch.fft.rfft2(t).abs()).mean()

    # 3. PCC (delayed warm-up prevents early mode collapse)
    lam_pcc = 0.0 if epoch < 10 else min(1.0 + epoch / 100.0, 6.0)
    t_var   = torch.var(t, dim=[-1, -2])
    mask    = (t_var > 1e-4).float().view(-1, 1, 1)
    pcc_l   = _pcc_log_cosh(p*mask, t*mask) if mask.sum() > 0 else p.new_zeros(1)

    # 4. Dry-pixel false-rain (asymmetric — only penalise false positives)
    dry_l = (torch.relu(p - 0.1) * (t <= 1e-4).float()).mean()

    # 5. Mass conservation
    mass_l = torch.abs(p.mean([-1, -2]) - t.mean([-1, -2])).mean()

    return spatial + 0.10*spectral + lam_pcc*pcc_l + 0.05*dry_l + 0.20*mass_l


# ════════════════════════════════════════════════════════════════════════════
# METRICS
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def _pcc(p: torch.Tensor, t: torch.Tensor) -> float:
    B = p.shape[0]
    p = p.float().view(B,-1); t = t.float().view(B,-1)
    pc = p - p.mean(1,keepdim=True); tc = t - t.mean(1,keepdim=True)
    return ((pc*tc).sum(1)/(torch.sqrt((pc**2).sum(1)*(tc**2).sum(1))+1e-8)).mean().item()

@torch.no_grad()
def raw_pcc(pred_log: torch.Tensor, target_log: torch.Tensor) -> float:
    return _pcc(torch.expm1(pred_log.float().clamp(0)),
                torch.expm1(target_log.float().clamp(0)))

@torch.no_grad()
def crps_ensemble(samp: torch.Tensor, tgt: torch.Tensor) -> float:
    N    = samp.shape[0]
    mae  = (samp - tgt.unsqueeze(0)).abs().mean(0)
    pair = (samp.unsqueeze(0) - samp.unsqueeze(1)).abs()
    return (mae - 0.5/N/(N-1) * pair.sum([0,1])).mean().item()

@torch.no_grad()
def fss(pred: torch.Tensor, tgt: torch.Tensor, thr=0.5, win=5) -> float:
    pb = (pred > thr).float(); tb = (tgt > thr).float()
    pf = F.avg_pool2d(pb, win, 1, win//2)
    tf = F.avg_pool2d(tb, win, 1, win//2)
    mse = ((pf-tf)**2).mean([-1,-2])
    ref = (pf**2).mean([-1,-2]) + (tf**2).mean([-1,-2])
    return (1. - mse/(ref+1e-8)).mean().item()


# ════════════════════════════════════════════════════════════════════════════
# DDIM SAMPLER  (bug-fixed: all c_in/c_out scoped inside loop)
# ════════════════════════════════════════════════════════════════════════════
@torch.no_grad()
def ddim_sample(model, mu, tc, tp, gf, d2m, var_map, sched, dev):
    B, _, H, W = mu.shape
    sigmas = sched.to(dev)
    x = torch.randn(B, 1, H, W, device=dev) * sigmas[0]

    for i, sig in enumerate(sigmas):
        sc     = sig.view(1, 1, 1, 1)
        c_in   = 1.0 / torch.sqrt(sc**2 + SIGMA_DATA**2)
        c_out  = sc * SIGMA_DATA / torch.sqrt(sc**2 + SIGMA_DATA**2)
        c_skip = SIGMA_DATA**2 / (sc**2 + SIGMA_DATA**2)
        c_n    = (sig.log() / 4).expand(B)

        x_in = torch.cat([c_in * x, mu, tc], dim=1)
        D    = model(x_in, c_n, topo=tp, global_features=gf, d2m=d2m, var_map=var_map)
        x0   = c_skip * x + c_out * D

        if i < len(sigmas) - 1:
            x = x0 + sigmas[i+1].view(1,1,1,1) * (x - x0) / sc.clamp(min=1e-8)
        else:
            x = x0
    return x


# ════════════════════════════════════════════════════════════════════════════
# DATA LOADER FACTORY
# ════════════════════════════════════════════════════════════════════════════
def make_loaders(ds, batch, ws, rank, trn, van):
    tr_sub = Subset(ds, range(0, trn))
    va_sub = Subset(ds, range(trn, trn + van))
    tr_smp = DistributedSampler(tr_sub, ws, rank, shuffle=True,  drop_last=True) if ws > 1 else None
    va_smp = DistributedSampler(va_sub, ws, rank, shuffle=False)                 if ws > 1 else None
    kw = dict(num_workers=8, pin_memory=True, persistent_workers=True, prefetch_factor=2)
    trl = DataLoader(tr_sub, batch, sampler=tr_smp, shuffle=(tr_smp is None), drop_last=True, **kw)
    val = DataLoader(va_sub, batch, sampler=va_smp, shuffle=False, **kw)
    return trl, val, tr_sub, va_sub, tr_smp


# ════════════════════════════════════════════════════════════════════════════
# STAGE 1 — REGRESSOR TRAINING
# ════════════════════════════════════════════════════════════════════════════
def train_regressor(rank, ws, local, dev):
    os.makedirs(CKPT_DIR, exist_ok=True)

    ds  = UpscaleDataset(RF_PATH, ORO_PATH, d2m_file=D2M_PATH,
                         split="train", normalize=True, device="cpu")
    n   = len(ds)
    trn = int(0.70*n); van = int(0.10*n)
    trl, val, _, _, tr_smp = make_loaders(ds, REG_BATCH, ws, rank, trn, van)

    model = CorrDiffRegressor(
        in_channels=2, out_channels=1, base_channels=64,
        channel_mult=(1, 2, 4), num_blocks=2,
        global_dim=GLOBAL_DIM, topo_channels=TOPO_CH,
        d2m_channels=1, use_d2m=True,
    ).to(dev)

    if ws > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local])
    raw = model.module if ws > 1 else model

    opt    = AdamW(model.parameters(), lr=REG_LR, weight_decay=WEIGHT_DECAY)
    scaler = GradScaler(device=dev.type)
    # Linear warm-up (5 ep) → cosine with restarts
    sched  = SequentialLR(opt,
                          schedulers=[LinearLR(opt, 0.1, 1.0, total_iters=5),
                                      CosineAnnealingWarmRestarts(opt, T_0=50, T_mult=2,
                                                                  eta_min=REG_LR*0.02)],
                          milestones=[5])

    start = 0; best_loss = float("inf"); no_imp = 0
    if os.path.exists(REG_LATEST) and rank == 0:
        ck = torch.load(REG_LATEST, map_location=dev)
        raw.load_state_dict({k.replace("module.", ""): v
                             for k, v in ck["model_state_dict"].items()})
        opt.load_state_dict(ck["opt"])
        start = ck["epoch"] + 1; best_loss = ck["best_loss"]; no_imp = ck["no_imp"]
        print(f"[Stage 1] Resumed ep={start}  best_loss={best_loss:.5f}")
    if ws > 1:
        dist.barrier()
        # Broadcast resume state to all ranks
        start     = broadcast_scalar(start,     dev)
        best_loss = broadcast_scalar(best_loss, dev)
        no_imp    = broadcast_scalar(no_imp,    dev)
        dist.broadcast(torch.tensor(0, device=dev), src=0)  # sync params via barrier above

    if rank == 0:
        p_ = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"[Stage 1] Regressor {p_:.1f}M | LR={REG_LR} | eff_batch={REG_BATCH*ws}")
        hdr = f"{'Ep':>5} | {'TrLoss':>8} | {'VaLoss':>8} | {'vPCC':>7} | {'LR':>9}"
        print(hdr); print("-"*len(hdr))

    for ep in range(start, REG_EPOCHS):
        if tr_smp: tr_smp.set_epoch(ep)
        model.train(); t0 = time.time(); sl = nb = 0.0

        for b in trl:
            fp, topo_1, xi_raw, var_map, d2m, gf, _ = get_batch_tensors(b, dev)

            # Spatial flip augmentation (applied consistently to all spatial tensors)
            spat    = [fp, topo_1, xi_raw, var_map]
            spat_op = [d2m]
            if torch.rand(1) < 0.5: spat, spat_op = random_flip(spat, spat_op, -1)
            if torch.rand(1) < 0.5: spat, spat_op = random_flip(spat, spat_op, -2)
            fp, topo_1, xi_raw, var_map = spat; d2m = spat_op[0]

            tp = expand_topo(topo_1)
            xi = build_coarse_input(xi_raw, var_map)

            with autocast(device_type=dev.type, dtype=torch.bfloat16):
                pred = model(xi, topo=tp, global_features=gf, d2m=d2m)
                loss = regressor_loss(pred, fp)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            sl += loss.item(); nb += 1

        sched.step()
        tr_t = torch.tensor(sl / max(nb, 1), device=dev)
        if ws > 1: dist.all_reduce(tr_t, op=dist.ReduceOp.SUM); tr_t /= ws
        tr_l = tr_t.item()

        # Validation
        model.eval(); vs = vn = ps = 0.0
        with torch.no_grad():
            for b in val:
                fp, topo_1, xi_raw, var_map, d2m, gf, _ = get_batch_tensors(b, dev)
                tp = expand_topo(topo_1); xi = build_coarse_input(xi_raw, var_map)
                with autocast(device_type=dev.type, dtype=torch.bfloat16):
                    pred = model(xi, topo=tp, global_features=gf, d2m=d2m)
                    lv   = regressor_loss(pred, fp)
                bs = fp.shape[0]
                vs += lv.item() * bs; vn += bs; ps += raw_pcc(pred, fp) * bs

        vm = torch.tensor([vs, vn, ps], device=dev)
        if ws > 1: dist.all_reduce(vm, op=dist.ReduceOp.SUM)
        va_l  = (vm[0] / vm[1].clamp(1)).item()
        va_pcc = (vm[2] / vm[1].clamp(1)).item()
        lr_now = opt.param_groups[0]["lr"]

        if rank == 0:
            star = " ★" if va_l < best_loss else ""
            print(f"{ep:>5} | {tr_l:>8.5f} | {va_l:>8.5f} | {va_pcc:>7.4f}"
                  f" | {lr_now:>9.2e}  [{time.time()-t0:.0f}s]{star}")
            ck = dict(model_state_dict=raw.state_dict(), opt=opt.state_dict(),
                      epoch=ep, best_loss=min(best_loss, va_l),
                      no_imp=no_imp if va_l >= best_loss else 0)
            torch.save(ck, REG_LATEST)
            if va_l < best_loss:
                best_loss = va_l; no_imp = 0
                torch.save(ck, REG_BEST)
                print(f"  ★ BEST val_loss={va_l:.5f}  R-PCC={va_pcc:.4f}")
            else:
                no_imp += 1

        no_imp = broadcast_scalar(no_imp, dev)
        if no_imp >= REG_PATIENCE:
            if rank == 0: print(f"\n[Stage 1] Early stop ep={ep+1}  best_loss={best_loss:.5f}")
            break

    if rank == 0: print(f"\n[Stage 1] Done → {REG_BEST}\n")
    if ws > 1: dist.barrier()


# ════════════════════════════════════════════════════════════════════════════
# LOAD FROZEN REGRESSOR
# ════════════════════════════════════════════════════════════════════════════
def load_regressor(path: str, dev: torch.device) -> CorrDiffRegressor:
    ck  = torch.load(path, map_location=dev)
    reg = CorrDiffRegressor(
        in_channels=2, out_channels=1, base_channels=64,
        channel_mult=(1, 2, 4), num_blocks=2,
        global_dim=GLOBAL_DIM, topo_channels=TOPO_CH,
        d2m_channels=ck.get("d2m_channels", 1),
        use_d2m=ck.get("use_d2m", True),
    ).to(dev)
    reg.load_state_dict({k.replace("module.", ""): v
                         for k, v in ck["model_state_dict"].items()})
    reg.eval()
    for p in reg.parameters(): p.requires_grad_(False)
    return reg


# ════════════════════════════════════════════════════════════════════════════
# STAGE 2 — DIFFUSION UNET
# ════════════════════════════════════════════════════════════════════════════
def train_unet(rank, ws, local, dev):
    os.makedirs(CKPT_DIR, exist_ok=True)

    ds  = UpscaleDataset(RF_PATH, ORO_PATH, d2m_file=D2M_PATH,
                         split="train", normalize=True, device="cpu")
    n   = len(ds)
    trn = int(0.70*n); van = int(0.10*n)
    trl, val, _, va_sub, tr_smp = make_loaders(ds, UNT_BATCH, ws, rank, trn, van)

    reg = load_regressor(REG_BEST, dev)
    if rank == 0: print(f"[Stage 2] Regressor frozen from {REG_BEST}")

    # num_blocks (not num_res_blocks), no temporal_frames param, no dropout param in UNet
    model = UNet(
        in_channels=UNET_IN_CH, out_channels=1, base_channels=BASE_CH,
        channel_mult=CHANNEL_MULT, num_blocks=NRB,
        global_dim=GLOBAL_DIM, topo_channels=TOPO_CH,
        use_d2m=True, d2m_channels=1,
        use_var_map=True, var_map_channels=1,
    ).to(dev)

    if ws > 1:
        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[local], find_unused_parameters=False)
    raw = model.module if ws > 1 else model

    ema    = EMA(raw)
    opt    = AdamW(model.parameters(), lr=UNT_LR, weight_decay=WEIGHT_DECAY, betas=(0.9, 0.999))
    scaler = GradScaler(device=dev.type)
    sched  = CosineAnnealingWarmRestarts(opt, T_0=100, T_mult=1, eta_min=UNT_MIN_LR)
    edm    = _EDM.clone()

    start = 0; best_pcc = -1.0; no_imp = 0
    if os.path.exists(UNT_BEST):
        ck = torch.load(UNT_BEST, map_location=dev)
        try:
            raw.load_state_dict({k.replace("module.", ""): v
                                 for k, v in ck["model_state_dict"].items()})
            opt.load_state_dict(ck["opt"])
            if "ema" in ck: ema.shadow = {k: v.to(dev) for k, v in ck["ema"].items()}
            start = ck["epoch"] + 1; best_pcc = ck["best_pcc"]; no_imp = ck["no_imp"]
            if rank == 0: print(f"[Stage 2] Resumed ep={start}  best_pcc={best_pcc:.4f}")
        except RuntimeError as e:
            if rank == 0: print(f"[Stage 2] Resume aborted (arch mismatch), fresh start: {e}")
    if ws > 1:
        start    = broadcast_scalar(start,    dev)
        best_pcc = broadcast_scalar(best_pcc, dev)
        no_imp   = broadcast_scalar(no_imp,   dev)

    if rank == 0:
        p_ = sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6
        print(f"[Stage 2] UNet {p_:.1f}M | IN={UNET_IN_CH} | eff_batch={UNT_BATCH*UNT_ACCUM*ws}")
        hdr = f"{'Ep':>5}|{'TrLoss':>9}|{'VaLoss':>8}|{'R-PCC':>7}|{'CRPS':>7}|{'FSS':>6}|{'LR':>9}"
        print(hdr); print("-"*len(hdr))

    for ep in range(start, UNT_EPOCHS):
        if tr_smp: tr_smp.set_epoch(ep)
        model.train(); t0 = time.time(); sl = nb = 0.0
        opt.zero_grad(set_to_none=True); _opt_steps = 0

        for step, b in enumerate(trl, 1):
            try:
                fp, topo_1, xi_raw, var_map, d2m, gf, tc = get_batch_tensors(b, dev)

                # Consistent flip augmentation across all spatial tensors
                spat    = [fp, topo_1, xi_raw, var_map, tc]
                spat_op = [d2m]
                if torch.rand(1) < 0.5: spat, spat_op = random_flip(spat, spat_op, -1)
                if torch.rand(1) < 0.5: spat, spat_op = random_flip(spat, spat_op, -2)
                fp, topo_1, xi_raw, var_map, tc = spat; d2m = spat_op[0]

                # Build derived tensors AFTER augmentation
                tp = expand_topo(topo_1)
                xi = build_coarse_input(xi_raw, var_map)

                with torch.no_grad():
                    with autocast(device_type=dev.type, dtype=torch.bfloat16):
                        mu = reg(xi, topo=tp, global_features=gf, d2m=d2m)
                mu = mu.float()

                residual = fp - mu

                # EDM noise level
                idx     = torch.randint(0, len(edm), (fp.shape[0],))
                sigma_t = edm[idx].to(dev).view(-1, 1, 1, 1)
                x_t     = residual + sigma_t * torch.randn_like(residual)

                # EDM preconditioning
                c_in   = 1.0 / torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                c_out  = sigma_t * SIGMA_DATA / torch.sqrt(sigma_t**2 + SIGMA_DATA**2)
                c_skip = SIGMA_DATA**2 / (sigma_t**2 + SIGMA_DATA**2)
                c_n    = (sigma_t.log() / 4).view(fp.shape[0])
                cfg_d  = torch.rand(fp.shape[0], device=dev) < P_CFG_DROP

                x_in = torch.cat([c_in * x_t, mu, tc], dim=1)

                with autocast(device_type=dev.type, dtype=torch.bfloat16):
                    D_pred  = model(x_in, c_n, topo=tp, global_features=gf,
                                    cfg_drop=cfg_d, d2m=d2m, var_map=var_map)
                    x0_pred = c_skip * x_t + c_out * D_pred
                    loss    = hybrid_sigma_loss(x0_pred, residual, sigma_t, ep) / UNT_ACCUM

                scaler.scale(loss).backward()

                if step % UNT_ACCUM == 0:
                    scaler.unscale_(opt)
                    nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                    if torch.isfinite(loss):
                        scaler.step(opt)
                        ema.update(raw)
                        _opt_steps += 1
                    else:
                        if rank == 0: print(f"[ep{ep} s{step}] NaN — skipped")
                    scaler.update()
                    opt.zero_grad(set_to_none=True)

                sl += loss.item() * UNT_ACCUM; nb += 1

            except Exception:
                if rank == 0: traceback.print_exc()
                raise

        sched.step()
        tr_t = torch.tensor(sl / max(nb, 1), device=dev)
        if ws > 1: dist.all_reduce(tr_t, op=dist.ReduceOp.SUM); tr_t /= ws
        tr_l   = tr_t.item()
        lr_now = opt.param_groups[0]["lr"]

        # ── Validation ────────────────────────────────────────────────────
        live = EMA.save_state(raw); ema.apply_to(raw); model.eval()
        do_heavy = (ep + 1) % 5 == 0
        vm = torch.zeros(5, device=dev)   # [loss_sum, n, pcc_sum, crps_sum, fss_sum]

        with torch.no_grad():
            for b in val:
                fp, topo_1, xi_raw, var_map, d2m, gf, tc = get_batch_tensors(b, dev)
                tp = expand_topo(topo_1); xi = build_coarse_input(xi_raw, var_map)

                with autocast(device_type=dev.type, dtype=torch.bfloat16):
                    mu = reg(xi, topo=tp, global_features=gf, d2m=d2m)
                mu = mu.float(); residual = fp - mu

                # Validation denoising loss (own scope — no c_in bleed-over)
                idx_v    = torch.randint(0, len(edm), (fp.shape[0],))
                sv       = edm[idx_v].to(dev).view(-1, 1, 1, 1)
                x_tv     = residual + sv * torch.randn_like(residual)
                ci_v     = 1.0 / torch.sqrt(sv**2 + SIGMA_DATA**2)
                co_v     = sv * SIGMA_DATA / torch.sqrt(sv**2 + SIGMA_DATA**2)
                cs_v     = SIGMA_DATA**2 / (sv**2 + SIGMA_DATA**2)
                cn_v     = (sv.log() / 4).view(fp.shape[0])

                with autocast(device_type=dev.type, dtype=torch.bfloat16):
                    Dv   = raw(torch.cat([ci_v*x_tv, mu, tc], 1), cn_v,
                               topo=tp, global_features=gf, d2m=d2m, var_map=var_map)
                    x0v  = cs_v*x_tv + co_v*Dv
                    lv   = hybrid_sigma_loss(x0v, residual, sv, ep)

                bs = fp.shape[0]
                vm[0] += lv * bs; vm[1] += bs

                if do_heavy:
                    samps = []
                    for _ in range(N_ENS):
                        s = ddim_sample(raw, mu, tc, tp, gf, d2m, var_map, edm, dev) + mu
                        s = PhysicsGuide.apply(s, xi_raw, enforce_mass=True, enforce_dry=True)
                        samps.append(s)
                    phys    = torch.stack([torch.expm1(s.clamp(0)) for s in samps])
                    best_s  = torch.log1p(phys.mean(0))
                    samp_t  = torch.stack(samps)
                    vm[2] += raw_pcc(best_s, fp) * bs
                    vm[3] += crps_ensemble(samp_t, fp) * bs
                    vm[4] += fss(best_s, fp) * bs

        EMA.restore(raw, live); model.train()
        if ws > 1 and dist.is_initialized(): dist.all_reduce(vm, op=dist.ReduceOp.SUM)
        va_l    = (vm[0] / vm[1].clamp(1)).item()
        is_best = False

        if rank == 0:
            ck_base = dict(model_state_dict=raw.state_dict(), opt=opt.state_dict(),
                           ema=ema.shadow, epoch=ep, best_pcc=best_pcc, no_imp=no_imp)
            torch.save(ck_base, UNT_LATEST)

            if do_heavy:
                rpcc = (vm[2] / vm[1].clamp(1)).item()
                crps = (vm[3] / vm[1].clamp(1)).item()
                fss_ = (vm[4] / vm[1].clamp(1)).item()
                star = " ★" if rpcc > best_pcc else ""
                print(f"{ep:>5}|{tr_l:>9.5f}|{va_l:>8.4f}|{rpcc:>7.4f}|{crps:>7.4f}"
                      f"|{fss_:>6.3f}|{lr_now:>9.2e}  [{time.time()-t0:.0f}s]{star}")
                if rpcc > best_pcc:
                    best_pcc = rpcc; no_imp = 0; is_best = True
                    ck_base["best_pcc"] = best_pcc
                    torch.save(ck_base, UNT_BEST)
                    if (ep + 1) % 50 == 0:
                        torch.save(ck_base,
                                   os.path.join(CKPT_DIR, f"unet_ep{ep+1:04d}_pcc{rpcc:.4f}.pth"))
                else:
                    no_imp += 2
            else:
                print(f"{ep:>5}|{tr_l:>9.5f}|{va_l:>8.4f}|-- fast epoch --|{lr_now:>9.2e}"
                      f"  [{time.time()-t0:.0f}s]")

        # ── Isolated QDM (Rank 0 only, barrier sync after) ────────────────
        if (ep + 1) % 5 == 0 or is_best:
            if rank == 0:
                print(f"\n[QDM] Calibrating on full val split...")
                qdm = QDM(n_quantiles=500)
                vqdl = DataLoader(va_sub, UNT_BATCH, shuffle=False,
                                  num_workers=4, pin_memory=True)
                ap, ao = [], []
                live2 = EMA.save_state(raw); ema.apply_to(raw); model.eval()

                with torch.no_grad():
                    for b in vqdl:
                        fp2, t12, xr2, vm2, d2_2, gf2, tc2 = get_batch_tensors(b, dev)
                        tp2 = expand_topo(t12); xi2 = build_coarse_input(xr2, vm2)
                        with autocast(device_type=dev.type, dtype=torch.bfloat16):
                            mu2 = reg(xi2, topo=tp2, global_features=gf2, d2m=d2_2)
                        mu2 = mu2.float()
                        s2  = ddim_sample(raw, mu2, tc2, tp2, gf2, d2_2, vm2, edm, dev) + mu2
                        s2  = PhysicsGuide.apply(s2, xr2, enforce_mass=True, enforce_dry=False)
                        ap.append(s2.cpu()); ao.append(fp2.cpu())

                EMA.restore(raw, live2); model.train()
                qdm.fit(torch.cat(ap), torch.cat(ao))
                qdm.save(QDM_LATEST)
                if is_best: qdm.save(QDM_BEST)
                print(f"[QDM] {'★ BEST ' if is_best else ''}saved → {QDM_LATEST}\n")

            if ws > 1 and dist.is_initialized(): dist.barrier()

        no_imp = broadcast_scalar(no_imp, dev)
        if no_imp >= UNT_PATIENCE:
            if rank == 0: print(f"\n[Stage 2] Early stop ep={ep+1}  best_pcc={best_pcc:.4f}")
            break

    if ws > 1 and dist.is_initialized(): dist.barrier()
    if rank == 0: print(f"\n[Stage 2] Done → {UNT_BEST}")


# ════════════════════════════════════════════════════════════════════════════
# INFERENCE (EMA weights, QDM calibration, ensemble)
# ════════════════════════════════════════════════════════════════════════════
class CorrDiffInference:
    def __init__(self, unet_ckpt, qdm_ckpt=None, device="cuda", cfg_scale=CFG_SCALE, n_ens=8):
        self.dev = torch.device(device); self.n_ens = n_ens
        self.reg  = load_regressor(REG_BEST, self.dev)
        ck        = torch.load(unet_ckpt, map_location=self.dev)
        self.unet = UNet(
            in_channels=ck.get("unet_in_channels", UNET_IN_CH), out_channels=1,
            base_channels=BASE_CH, channel_mult=CHANNEL_MULT, num_blocks=NRB,
            global_dim=GLOBAL_DIM, topo_channels=TOPO_CH,
            use_d2m=True,  d2m_channels=ck.get("d2m_channels", 1),
            use_var_map=True, var_map_channels=ck.get("var_map_channels", 1),
        ).to(self.dev)
        state = (ck["ema"] if "ema" in ck
                 else {k.replace("module.", ""): v for k, v in ck["model_state_dict"].items()})
        self.unet.load_state_dict({k: v.to(self.dev) for k, v in state.items()})
        self.unet.eval()
        self.edm = build_edm_schedule()
        self.qdm = QDM.load(qdm_ckpt) if qdm_ckpt and os.path.exists(qdm_ckpt) else None

    @torch.no_grad()
    def predict(self, coarse, var_map, topo_1, d2m, doy, hour, tc_frames):
        dev = self.dev
        fp_dummy = coarse  # shape [B,1,H,W] at coarse res
        gf  = torch.stack([doy, hour], 1).float().to(dev)
        tp  = expand_topo(topo_1.to(dev))
        xi  = build_coarse_input(coarse.to(dev), var_map.to(dev))
        tc  = tc_frames.to(dev)

        with autocast(device_type=dev.type, dtype=torch.bfloat16):
            mu = self.reg(xi, topo=tp, global_features=gf, d2m=d2m.to(dev) if d2m is not None else None)
        mu = mu.float()

        samps_phys = []
        for _ in range(self.n_ens):
            s = ddim_sample(self.unet, mu, tc, tp, gf,
                            d2m.to(dev) if d2m is not None else None,
                            var_map.to(dev), self.edm, dev) + mu
            s = PhysicsGuide.apply(s, coarse.to(dev), enforce_mass=True, enforce_dry=True)
            samps_phys.append(torch.expm1(s.clamp(0)))

        mean_phys = torch.stack(samps_phys).mean(0)
        out = torch.log1p(mean_phys)

        if self.qdm is not None:
            out = self.qdm.apply(out)
        return out


# ════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all",
                        choices=["all", "regressor", "unet"],
                        help="'all' runs Stage 1 then Stage 2 automatically")
    args = parser.parse_args()

    rank, ws, local, dev = setup_ddp()

    try:
        if args.stage in ("all", "regressor"):
            if args.stage == "regressor" or not os.path.exists(REG_BEST):
                train_regressor(rank, ws, local, dev)
            elif rank == 0:
                print(f"[Stage 1] Skipped — {REG_BEST} exists. Use --stage regressor to re-run.")

        if args.stage in ("all", "unet"):
            if not os.path.exists(REG_BEST):
                raise FileNotFoundError(
                    f"Regressor checkpoint missing: {REG_BEST}\n"
                    "Run:  torchrun --nproc_per_node=4 train.py --stage regressor")
            train_unet(rank, ws, local, dev)

    finally:
        if ws > 1 and dist.is_initialized():
            dist.destroy_process_group()
