"""
train_regressor.py
Stage-1 CorrDiff Regressor training script.
Usage:
    torchrun --standalone --nproc_per_node=4 train_regressor.py \
        --nc_file /path/to/RF.nc \
        --oro_file /path/to/oro.nc \
        --d2m_file /path/to/era5.nc
"""

import argparse
import os
import math
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, random_split
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import GradScaler, autocast

from Regressor import CorrDiffRegressor
from Dataset import UpscaleDataset   


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────

def get_args():
    p = argparse.ArgumentParser()
    # data
    p.add_argument("--nc_file",  required=True)
    p.add_argument("--oro_file", required=True)
    p.add_argument("--d2m_file", default=None)
    # training
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=4)   # per GPU
    p.add_argument("--lr",           type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--grad_clip",    type=float, default=1.0)
    p.add_argument("--val_split",    type=float, default=0.1)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--amp",          action="store_true", default=True)
    # model
    p.add_argument("--base_channels", type=int, default=64)
    p.add_argument("--num_blocks",    type=int, default=2)
    # checkpointing
    p.add_argument("--ckpt_dir",    default="checkpoints")
    p.add_argument("--resume",      default=None)
    p.add_argument("--save_every",  type=int, default=5)
    p.add_argument("--log_every",   type=int, default=50)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# DDP helpers
# ─────────────────────────────────────────────────────────────────────────────

def setup_ddp():
    dist.init_process_group("nccl")
    rank       = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return rank, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def is_main(rank):
    return rank == 0

def reduce_mean(tensor, rank):
    t = tensor.clone()
    dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return t / dist.get_world_size()


# ─────────────────────────────────────────────────────────────────────────────
# Loss
# ─────────────────────────────────────────────────────────────────────────────

def reg_loss(pred, target):
    """L1 + 0.5*MSE — robust to precip outliers."""
    return nn.functional.l1_loss(pred, target) + 0.5 * nn.functional.mse_loss(pred, target)


# ─────────────────────────────────────────────────────────────────────────────
# One epoch
# ─────────────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scaler, device, args, train=True):
    model.train(train)
    total_loss = torch.tensor(0.0, device=device)
    n_steps    = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for step, batch in enumerate(loader):
            fine    = batch["fine"].to(device, non_blocking=True)       # [B,1,H,W]
            coarse  = batch["coarse"].to(device, non_blocking=True)     # [B,1,H/4,W/4]
            topo    = batch["topo"].to(device, non_blocking=True)       # [B,1,H,W]
            var_map = batch["var_map"].to(device, non_blocking=True)    # [B,1,H/4,W/4]
            doy     = batch["doy"].to(device, non_blocking=True)        # [B]
            hour    = batch["hour"].to(device, non_blocking=True)       # [B]
            d2m     = batch["d2m"].to(device, non_blocking=True) if "d2m" in batch else None

            # stack coarse + var_map as model input (2 channels, coarse res)
            x_in = torch.cat([coarse, var_map], dim=1)                  # [B,2,H/4,W/4]

            # global features: [doy, hour]
            gf = torch.stack([doy, hour], dim=1)                        # [B,2]

            # topo: repeat to 3 channels (slope x, slope y, elevation)
            topo3 = _make_topo3(topo)                                   # [B,3,H,W]

            with autocast(enabled=args.amp):
                pred = model(x_in, topo3, global_features=gf, d2m=d2m) # [B,1,H,W]
                loss = reg_loss(pred, fine)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.detach()
            n_steps    += 1

            if train and is_main(dist.get_rank()) and (step + 1) % args.log_every == 0:
                avg = total_loss.item() / n_steps
                print(f"  step {step+1}/{len(loader)}  loss={avg:.5f}", flush=True)

    return (total_loss / max(n_steps, 1)).item()


def _make_topo3(topo):
    """Convert 1-channel topo to 3-channel [elev, grad_y, grad_x]."""
    e = topo
    gy = torch.zeros_like(e)
    gx = torch.zeros_like(e)
    gy[:, :, 1:-1, :] = (e[:, :, 2:, :] - e[:, :, :-2, :]) / 2
    gx[:, :, :, 1:-1] = (e[:, :, :, 2:] - e[:, :, :, :-2]) / 2
    return torch.cat([e, gy, gx], dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args       = get_args()
    rank, local_rank = setup_ddp()
    device     = torch.device(f"cuda:{local_rank}")
    main_proc  = is_main(rank)

    if main_proc:
        os.makedirs(args.ckpt_dir, exist_ok=True)
        print(f"[Config] {vars(args)}", flush=True)

    # ── dataset ──────────────────────────────────────────────────────────
    full_ds = UpscaleDataset(
        nc_file          = args.nc_file,
        oro_file         = args.oro_file,
        d2m_file         = args.d2m_file,
        downscale_factor = 4,
        normalize        = True,
        device           = "cpu",
    )

    n_val   = max(1, int(len(full_ds) * args.val_split))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(42))

    train_sampler = DistributedSampler(train_ds, shuffle=True)
    val_sampler   = DistributedSampler(val_ds,   shuffle=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              sampler=val_sampler,
                              num_workers=args.num_workers,
                              pin_memory=True, persistent_workers=True)

    # ── model ─────────────────────────────────────────────────────────────
    use_d2m = args.d2m_file is not None
    model = CorrDiffRegressor(
        in_channels   = 2,
        out_channels  = 1,
        base_channels = args.base_channels,
        channel_mult  = (1, 2, 4),
        num_blocks    = args.num_blocks,
        global_dim    = 2,
        topo_channels = 3,
        d2m_channels  = 1,
        use_d2m       = use_d2m,
    ).to(device)

    model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # ── optimizer / scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
    scaler    = GradScaler(enabled=args.amp)

    # ── resume ────────────────────────────────────────────────────────────
    start_epoch = 0
    best_val    = math.inf

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        start_epoch = ckpt["epoch"] + 1
        best_val    = ckpt.get("best_val", math.inf)
        if main_proc:
            print(f"[Resume] epoch {start_epoch}  best_val={best_val:.5f}", flush=True)

    # ── training loop ─────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        train_sampler.set_epoch(epoch)
        t0 = time.time()

        tr_loss = run_epoch(model, train_loader, optimizer, scaler,
                            device, args, train=True)
        vl_loss = run_epoch(model, val_loader,   optimizer, scaler,
                            device, args, train=False)

        # reduce across GPUs for clean logging
        tr_t = reduce_mean(torch.tensor(tr_loss, device=device), rank).item()
        vl_t = reduce_mean(torch.tensor(vl_loss, device=device), rank).item()

        scheduler.step()

        if main_proc:
            elapsed = time.time() - t0
            lr_now  = optimizer.param_groups[0]["lr"]
            print(f"Epoch {epoch+1}/{args.epochs}  "
                  f"train={tr_t:.5f}  val={vl_t:.5f}  "
                  f"lr={lr_now:.2e}  t={elapsed:.1f}s", flush=True)

            # save best
            if vl_t < best_val:
                best_val = vl_t
                torch.save({
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val": best_val,
                }, os.path.join(args.ckpt_dir, "best.pt"))
                print(f"  ↳ saved best.pt  val={best_val:.5f}", flush=True)

            # periodic save
            if (epoch + 1) % args.save_every == 0:
                torch.save({
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                    "best_val": best_val,
                }, os.path.join(args.ckpt_dir, f"epoch_{epoch+1:04d}.pt"))

    if main_proc:
        print("Training complete.", flush=True)

    cleanup_ddp()


if __name__ == "__main__":
