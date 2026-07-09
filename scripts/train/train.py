"""RCAR-VLA training (MT-10 trial run, time-budgeted)."""
import argparse
import time
from pathlib import Path

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from models.rcar_vla import RCARVLA
from data.datasets.metaworld_dataset import build_dataloader


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ── Model ──
    model = RCARVLA(
        action_dim=4, 
        horizon=args.horizon, 
        proprio_dim=args.proprio_dim, 
        ema_decay=args.ema_decay
    ).to(device)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[trainable] {n_train/1e6:.2f}M params")

    # ── Data ──
    dl = build_dataloader(
        data_root=args.data_root,
        hdf5_file=args.hdf5_file,
        task_set=args.tasks,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rcar_mode=True,
        rcar_seed=42,
        proprio_dim=args.proprio_dim,
    )
    print(f"[data] {len(dl)} batches/epoch, batch_size={args.batch_size}")

    # ── Optim ──
    optimizer = optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    total_epochs = args.phase1_epochs + args.phase2_epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_epochs)

    # ── Resume Logic ──
    start_epoch = 0
    if args.resume and Path(args.resume).exists():
        print(f"[resume] loading {args.resume}")
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        print(f"[resume] starting from epoch {start_epoch+1}")

    # bf16 autocast (no GradScaler needed — bf16 has same exp range as fp32)
    use_bf16 = torch.cuda.is_bf16_supported()
    print(f"[precision] bf16={use_bf16}")

    t_start = time.time()
    best_loss = float("inf")

    for epoch in range(start_epoch, total_epochs):
        in_phase1 = epoch < args.phase1_epochs
        lw_sc = 0.0 if in_phase1 else 0.5

        model.train()
        ep_loss = 0.0
        ep_cfm = 0.0
        ep_mode = 0.0
        ep_fail = 0.0
        ep_sc = 0.0
        n = 0

        pbar = tqdm(dl, desc=f"ep {epoch+1}/{total_epochs}")
        for batch in pbar:
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                     for k, v in batch.items()}
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                out = model(batch, compute_sc=not in_phase1, lw_sc=lw_sc)

            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()

            # EMA update on flow-matching teacher
            model.cfm.update_ema()

            ep_loss += out.loss.item(); ep_cfm += out.loss_cfm1.item()
            ep_mode += out.loss_mode.item(); ep_fail += out.loss_fail.item()
            ep_sc += out.loss_sc.item(); n += 1

            pbar.set_postfix({
                "loss": f"{out.loss.item():.3f}",
                "cfm": f"{out.loss_cfm1.item():.3f}",
                "mode": f"{out.loss_mode.item():.3f}",
                "fail": f"{out.loss_fail.item():.3f}",
                "sc": f"{out.loss_sc.item():.3f}",
            })

        scheduler.step()
        avg = ep_loss / max(n, 1)
        elapsed = time.time() - t_start
        print(f"[ep {epoch+1}] loss={avg:.4f} cfm={ep_cfm/n:.4f} mode={ep_mode/n:.4f} "
              f"fail={ep_fail/n:.4f} sc={ep_sc/n:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
              f"elapsed={elapsed/60:.1f}min")

        # Checkpoint
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "args": vars(args),
        }
        torch.save(ckpt, Path(args.out_dir) / "last.pt")
        if avg < best_loss:
            best_loss = avg
            torch.save(ckpt, Path(args.out_dir) / "best.pt")

        # Wall-clock budget guard
        if args.max_minutes > 0 and elapsed / 60 > args.max_minutes:
            print(f"[budget] hit {args.max_minutes}min — stopping early")
            break


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="/home/user/Desktop/vla_projects/tools/tc-pruner/data")
    p.add_argument("--hdf5_file", default="mt50_multiview_full.hdf5")
    p.add_argument("--tasks", default="mt10")
    p.add_argument("--out_dir", default="checkpoints/rcar_mt10_patched")
    p.add_argument("--proprio_dim", type=int, default=7)
    p.add_argument("--ema_decay", type=float, default=0.999)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--num_workers", type=int, default=8)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight_decay", type=float, default=0.05)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--phase1_epochs", type=int, default=10)
    p.add_argument("--phase2_epochs", type=int, default=10)
    p.add_argument("--max_minutes", type=int, default=0,
                   help="hard wall-clock budget; 0 disables")
    p.add_argument("--resume", default="", help="Path to checkpoint to resume from")
    args = p.parse_args()
    train(args)
