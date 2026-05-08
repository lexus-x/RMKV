#!/usr/bin/env python3
"""
KANFlow-VLA Training Script.

Trains the RWKV-GroupKAN + Consistency Flow Matching VLA on MetaWorld MT-50
with 10 expert demonstrations per task.

Usage:
    python -m kanflow_vla.train                           # defaults
    python -m kanflow_vla.train --config configs/metaworld.yaml
    python -m kanflow_vla.train --batch-size 64 --lr 5e-5  # overrides

Based on KAN-We-Flow (arXiv:2602.01115v2) training setup:
    - AdamW, lr=1e-4, batch=128, 3000 epochs
    - EMA decay=0.95
    - Mixed precision (bf16)
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

# Use the modern autocast API (torch 2.x+)
try:
    from torch.amp import autocast, GradScaler
except ImportError:
    from torch.cuda.amp import autocast, GradScaler

try:
    import yaml
except ImportError:
    yaml = None

try:
    import wandb
except ImportError:
    wandb = None

from kanflow_vla.model.kanflow_vla import KANFlowVLA
from kanflow_vla.losses import KANFlowLoss
from kanflow_vla.data.metaworld_dataset import MetaWorldDataset, build_dataloader
from kanflow_vla.eval_metaworld import (
    HAS_METAWORLD,
    MT10_TASKS,
    build_task_lang_ids,
    evaluate_task,
    load_task_tokenizer,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KANFlow-VLA Training")
    parser.add_argument("--config", type=str, default=None, help="Config YAML path")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints/kanflow_vla")
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--mixed-precision", type=str, default="bf16",
                        choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="kanflow-vla")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop if validation SR doesn't improve for N validation cycles (0=disabled)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="Run 10 steps with synthetic data for validation")
    parser.add_argument("--ablation", type=str, default=None,
                        choices=[None, "disable_groupkan", "disable_rwkv", "standard_transformer"],
                        help="Ablation mode for novelty proof")
    parser.add_argument("--domain-randomize", action="store_true",
                        help="Enable visual domain randomization (Phase 2)")
    parser.add_argument("--val-rollouts", type=int, default=1,
                        help="Number of rollouts per MT10 validation task")
    parser.add_argument("--val-max-steps", type=int, default=500,
                        help="Max steps per MT10 validation rollout")
    parser.add_argument("--sampler", type=str, default="balanced",
                        choices=["balanced", "uniform"],
                        help="Sampler mode: balanced (weighted) or uniform (plain shuffle)")
    parser.add_argument("--epoch-size", type=int, default=None,
                        help="Fixed epoch size for balanced sampler (default: dataset length)")
    return parser.parse_args()


def load_config(path: str | None) -> dict:
    """Load YAML config, or return empty dict if no config given."""
    if path is None or yaml is None:
        return {}
    with open(path) as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass


def inspect_resume_checkpoint(path: str) -> dict:
    """Inspect a checkpoint for compatibility-sensitive model settings."""
    ckpt = torch.load(path, map_location="cpu")
    state_dict = ckpt.get("model_state_dict", ckpt)
    proprio_weight = state_dict.get("proprio_mlp.0.weight")
    proprio_dim = proprio_weight.shape[1] if proprio_weight is not None else 15
    obs_length = max(1, proprio_dim // 15)
    has_reliability = any(k.startswith("reliability.") for k in state_dict)
    return {
        "obs_length": obs_length,
        "proprio_dim": proprio_dim,
        "has_reliability": has_reliability,
    }


def adapt_checkpoint_state_dict(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], list[tuple[str, tuple[int, ...], tuple[int, ...]]], list[tuple[str, tuple[int, ...], tuple[int, ...]]]]:
    """
    Filter checkpoint weights to the current model, adapting simple input-size drifts.

    Returns:
        filtered_state_dict, adapted_keys, skipped_mismatches
    """
    target_state = model.state_dict()
    filtered = {}
    adapted = []
    skipped = []

    for key, value in state_dict.items():
        target = target_state.get(key)
        if target is None:
            continue
        if target.shape == value.shape:
            filtered[key] = value
            continue
        if key == "proprio_mlp.0.weight" and target.shape[0] == value.shape[0]:
            old_in = value.shape[1]
            new_in = target.shape[1]
            if new_in > old_in and new_in % old_in == 0:
                repeat_factor = new_in // old_in
                filtered[key] = value.repeat(1, repeat_factor) / repeat_factor
                adapted.append((key, tuple(value.shape), tuple(target.shape)))
                continue
            if old_in > new_in and old_in % new_in == 0:
                chunk = old_in // new_in
                filtered[key] = value.view(value.shape[0], new_in, chunk).mean(dim=2)
                adapted.append((key, tuple(value.shape), tuple(target.shape)))
                continue
        skipped.append((key, tuple(value.shape), tuple(target.shape)))

    return filtered, adapted, skipped


def build_model(cfg: dict, device: torch.device) -> KANFlowVLA:
    """Build KANFlowVLA model from config."""
    model_cfg = cfg.get("model", {})
    unet_cfg = model_cfg.get("unet", {})

    obs_length = cfg.get("data", {}).get("obs_length", 2)

    model = KANFlowVLA(
        action_dim=model_cfg.get("action_dim", 4),
        horizon=model_cfg.get("horizon", 4),
        d_model=model_cfg.get("d_model", 256),
        proprio_dim=model_cfg.get("proprio_dim", 15) * obs_length,
        unet_base_dim=unet_cfg.get("base_dim", 128),
        num_groups=unet_cfg.get("num_groups", 4),
        num_knots=unet_cfg.get("num_knots", 8),
        num_segments=model_cfg.get("cfm", {}).get("num_segments", 2),
        ema_decay=model_cfg.get("cfm", {}).get("ema_decay", 0.95),
        delta_t=model_cfg.get("cfm", {}).get("delta_t", 0.01),
        lambda_acr=model_cfg.get("cfm", {}).get("lambda_acr", 1.0),
        alpha_consistency=model_cfg.get("cfm", {}).get("alpha_consistency", 1.0),
        inference_t=model_cfg.get("cfm", {}).get("inference_t", 0.0),
        vision_config=model_cfg.get("vision", {}),
        language_config=model_cfg.get("language", {}),
        fusion_config=model_cfg.get("fusion", {}),
        freeze_encoder=model_cfg.get("freeze_encoder", True),
    )

    return model.to(device)


def build_optimizer(model: KANFlowVLA, cfg: dict) -> torch.optim.Optimizer:
    """Build AdamW optimizer with per-module LR groups."""
    optim_cfg = cfg.get("optim", {})
    base_lr = optim_cfg.get("base_lr", 1e-4)
    weight_decay = optim_cfg.get("weight_decay", 0.05)
    betas = tuple(optim_cfg.get("betas", [0.9, 0.999]))
    eps = optim_cfg.get("eps", 1e-8)

    param_groups = model.get_param_groups(base_lr, weight_decay)

    optimizer = torch.optim.AdamW(
        param_groups,
        lr=base_lr,
        betas=betas,
        eps=eps,
    )

    return optimizer


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    steps_per_epoch: int,
    num_epochs: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Build cosine LR scheduler with warmup."""
    sched_cfg = cfg.get("optim", {}).get("schedule", {})
    warmup_steps = sched_cfg.get("warmup_steps", 500)
    min_lr = sched_cfg.get("min_lr", 1e-6)
    total_steps = steps_per_epoch * num_epochs

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        import math
        return max(min_lr / optimizer.defaults["lr"],
                    0.5 * (1.0 + math.cos(math.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: KANFlowVLA,
    dataloader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    loss_fn: KANFlowLoss,
    scaler: GradScaler | None,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    global_step: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> tuple[float, int]:
    """Train for one epoch. Returns (avg_loss, updated_global_step)."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        images = batch["images"].to(device)
        proprio = batch["proprio"].to(device)
        actions = batch["actions"].to(device)
        lang_ids = batch["lang_ids"].to(device)
        # RCAR labels (optional — only present when rcar_mode=True)
        mode_labels = batch["mode_label"].to(device) if "mode_label" in batch else None
        failure_labels = batch["failure_label"].to(device) if "failure_label" in batch else None
        progress_labels = batch["progress_label"].to(device) if "progress_label" in batch else None
        # NOTE: Do NOT squeeze images/proprio here.
        # The model's _encode_condition handles (B, T, V, C, H, W) → (B, V, C, H, W) correctly.

        optimizer.zero_grad()

        def _compute_loss(output):
            return loss_fn(
                {"loss": output.loss, "cfm_loss": output.cfm_loss,
                 "acr_loss": output.acr_loss, "velocity_loss": output.velocity_loss},
                predicted_actions=output.actions,
                expert_actions=actions,
                mode_logits=output.mode_logits,
                failure_logits=output.failure_logits,
                progress_pred=output.progress,
                mode_labels=mode_labels,
                failure_labels=failure_labels,
                progress_labels=progress_labels,
            )

        if use_amp:
            with autocast(device_type="cuda", dtype=amp_dtype):
                output = model(images, lang_ids, proprio, expert_actions=actions)
                loss_dict = _compute_loss(output)
                loss = loss_dict["total_loss"]

            if scaler is not None:
                scaler.scale(loss).backward()
                if args.grad_clip > 0:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if args.grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
        else:
            output = model(images, lang_ids, proprio, expert_actions=actions)
            loss_dict = _compute_loss(output)
            loss = loss_dict["total_loss"]
            loss.backward()
            if args.grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

        # Update EMA teacher
        model.update_ema()

        # Update scheduler
        if scheduler is not None:
            scheduler.step()

        total_loss += loss.item()
        num_batches += 1
        global_step += 1

        # Logging
        if global_step % args.log_every == 0:
            from kanflow_vla.metrics import accuracy, mae
            # Show LR from main trainable group (unet), not frozen vision
            lr = next(
                (g["lr"] for g in optimizer.param_groups if g.get("name") == "unet"),
                optimizer.param_groups[-1]["lr"],
            )
            # Compute auxiliary RCAR metrics if heads are active
            mode_acc_str = ""
            progress_mae_str = ""
            if output.mode_logits is not None and mode_labels is not None:
                m_acc = accuracy(output.mode_logits.detach(), mode_labels)
                mode_acc_str = f"  mode_acc={m_acc:.2f}"
            if output.progress is not None and progress_labels is not None:
                p_mae = mae(output.progress.detach(), progress_labels.float().to(device))
                progress_mae_str = f"  prog_mae={p_mae:.3f}"
            print(
                f"  [Step {global_step:6d} | Epoch {epoch:4d}] "
                f"loss={loss.item():.4f}  "
                f"cfm={output.cfm_loss.item():.4f}  "
                f"acr={output.acr_loss.item():.4f}  "
                f"lr={lr:.2e}"
                f"{mode_acc_str}{progress_mae_str}"
            )

            if args.wandb and wandb is not None:
                wandb.log({
                    "train/loss": loss.item(),
                    "train/cfm_loss": output.cfm_loss.item(),
                    "train/acr_loss": output.acr_loss.item(),
                    "train/velocity_loss": output.velocity_loss.item(),
                    "train/lr": lr,
                    "train/epoch": epoch,
                    "train/step": global_step,
                })

        if args.smoke_test and batch_idx >= 10:
            break

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss, global_step


def save_checkpoint(
    model: KANFlowVLA,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    loss: float,
    path: str,
):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "epoch": epoch,
        "global_step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "loss": loss,
    }, path)
    print(f"  Saved checkpoint: {path}")


def run_mt10_validation(
    model: KANFlowVLA,
    device: torch.device,
    views: list[str],
    task_lang_ids: dict[str, torch.Tensor],
    args: argparse.Namespace,
    obs_length: int = 2,
) -> float:
    """Run a lightweight MT10 validation pass and return mean success rate."""
    val_results = []
    for task in MT10_TASKS:
        result = evaluate_task(
            model=model,
            task_name=task,
            seed=0,
            views=views,
            num_rollouts=args.val_rollouts,
            max_steps=args.val_max_steps,
            num_inference_steps=1,
            device=device,
            save_video=False,
            lang_ids=task_lang_ids.get(task),
            obs_length=obs_length,
        )
        if result is None:
            continue
        val_results.append(result["mean_success"])

    return sum(val_results) / len(val_results) if val_results else 0.0


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Apply CLI overrides
    cfg.setdefault("training", {})
    cfg["training"]["batch_size"] = args.batch_size
    cfg["training"]["num_epochs"] = args.epochs
    cfg.setdefault("optim", {})
    cfg["optim"]["base_lr"] = args.lr
    cfg.setdefault("data", {})
    cfg.setdefault("model", {})
    cfg["model"]["ablation"] = args.ablation
    cfg["model"]["domain_randomize"] = args.domain_randomize

    if args.resume:
        resume_info = inspect_resume_checkpoint(args.resume)
        cfg_obs_length = cfg.get("data", {}).get("obs_length", 2)
        resume_obs_length = resume_info["obs_length"]
        if resume_obs_length != cfg_obs_length:
            print(
                f"[Resume] Overriding data.obs_length from {cfg_obs_length} "
                f"to {resume_obs_length} to match checkpoint proprio_dim="
                f"{resume_info['proprio_dim']}."
            )
            cfg["data"]["obs_length"] = resume_obs_length

    # Setup
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*65}")
    print(f"KANFlow-VLA Training")
    print(f"{'='*65}")
    print(f"Device: {device}")
    print(f"Mixed precision: {args.mixed_precision}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")

    # W&B init
    if args.wandb and wandb is not None:
        wandb.init(
            project=args.wandb_project,
            name=args.run_name or cfg.get("run_name"),
            config=cfg,
        )

    # Build model
    model = build_model(cfg, device)

    loss_cfg = cfg.get("loss", {})
    if (
        loss_cfg.get("lambda_mode", 0.0) == 0.0
        and loss_cfg.get("lambda_failure", 0.0) == 0.0
        and loss_cfg.get("lambda_progress", 0.0) == 0.0
    ):
        for param in model.reliability.parameters():
            param.requires_grad = False
        print("[Setup] RCAR losses disabled; reliability heads frozen.")

    # Build data loader
    data_cfg = cfg.get("data", {})
    data_root = args.data_root or data_cfg.get(
        "data_root", "/home/user/Desktop/vla_projects/tc-pruner/data"
    )

    if args.smoke_test:
        from pathlib import Path
        hdf5_path = Path(data_root) / data_cfg.get("hdf5_file", "mt50_multiview_full.hdf5")
        data_source = f"real HDF5 ({hdf5_path.name})" if hdf5_path.exists() else "synthetic (HDF5 not found)"
        args.checkpoint_dir = "./outputs/smoke_test_checkpoints"
        os.makedirs(args.checkpoint_dir, exist_ok=True)
        print(f"\n[SMOKE TEST MODE] Data source: {data_source}\n")
        print(f"[SMOKE TEST MODE] Checkpoints will be written to: {args.checkpoint_dir}\n")

    dataloader = build_dataloader(
        data_root=data_root,
        hdf5_file=data_cfg.get("hdf5_file", "mt50_multiview_full.hdf5"),
        task_set=data_cfg.get("task_set", "mt50"),
        batch_size=args.batch_size,
        num_workers=0 if args.smoke_test else args.num_workers,
        img_size=data_cfg.get("img_size", 224),
        horizon=cfg.get("model", {}).get("horizon", 4),
        obs_length=data_cfg.get("obs_length", 2),
        max_demos_per_task=data_cfg.get("max_demos_per_task", 10),
        augment=not args.smoke_test,
        domain_randomize=args.domain_randomize,
        views=data_cfg.get("views", ["image_corner2", "image_gripperPOV"]),
        tokenizer_model=cfg.get("model", {}).get("language", {}).get(
            "name", "HuggingFaceTB/SmolLM-135M"
        ),
        sampler_mode=args.sampler,
        epoch_size=args.epoch_size,
        rcar_mode=cfg.get("data", {}).get("rcar_mode", False),
        rcar_seed=args.seed,
    )

    # Build optimizer, scheduler, loss
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(
        optimizer, cfg,
        steps_per_epoch=len(dataloader),
        num_epochs=args.epochs,
    )
    loss_fn = KANFlowLoss(
        gripper_weight_mult=loss_cfg.get("gripper_weight_mult", 3.0),
        lambda_mode=loss_cfg.get("lambda_mode", 0.0),
        lambda_failure=loss_cfg.get("lambda_failure", 0.0),
        lambda_progress=loss_cfg.get("lambda_progress", 0.0),
    )

    # Mixed precision setup
    use_amp = args.mixed_precision != "fp32" and device.type == "cuda"
    amp_dtype = torch.bfloat16 if args.mixed_precision == "bf16" else torch.float16
    scaler = GradScaler("cuda") if (use_amp and args.mixed_precision == "fp16") else None

    # Resume
    global_step = 0
    start_epoch = 0
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        ckpt_state_dict = ckpt.get("model_state_dict", ckpt)
        filtered_state_dict, adapted_keys, skipped_mismatches = adapt_checkpoint_state_dict(
            model, ckpt_state_dict
        )
        incompatible = model.load_state_dict(filtered_state_dict, strict=False)
        if adapted_keys:
            print(f"[Resume] Adapted {len(adapted_keys)} checkpoint tensor(s) for shape compatibility.")
        if skipped_mismatches:
            first_key, old_shape, new_shape = skipped_mismatches[0]
            print(
                f"[Resume] Skipped {len(skipped_mismatches)} mismatched tensor(s); "
                f"first: {first_key} {old_shape} -> {new_shape}"
            )
        if incompatible.missing_keys:
            print(
                f"[Resume] Missing {len(incompatible.missing_keys)} model key(s); "
                f"first: {incompatible.missing_keys[0]}"
            )
        if incompatible.unexpected_keys:
            print(
                f"[Resume] Ignoring {len(incompatible.unexpected_keys)} unexpected checkpoint key(s); "
                f"first: {incompatible.unexpected_keys[0]}"
            )

        optimizer_loaded = False
        if ckpt.get("optimizer_state_dict"):
            try:
                optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                optimizer_loaded = True
            except ValueError as exc:
                print(
                    "[Resume] Optimizer state incompatible with current model; "
                    f"starting with a fresh optimizer. ({exc})"
                )
        if ckpt.get("scheduler_state_dict") and scheduler and optimizer_loaded:
            try:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            except Exception as exc:
                print(
                    "[Resume] Scheduler state incompatible; "
                    f"starting with a fresh scheduler. ({exc})"
                )
        start_epoch = ckpt["epoch"] + 1
        global_step = ckpt["global_step"]
        print(f"Resumed from epoch {start_epoch}, step {global_step}")

    views = data_cfg.get("views", ["image_corner2", "image_gripperPOV"])
    val_tokenizer = load_task_tokenizer(
        cfg.get("model", {}).get("language", {}).get(
            "name", "HuggingFaceTB/SmolLM-135M"
        )
    )
    val_task_lang_ids = build_task_lang_ids(
        MT10_TASKS, tokenizer=val_tokenizer, device=device
    )

    # ── Training Loop ──
    print(f"\nStarting training from epoch {start_epoch}...\n", flush=True)
    best_loss = float("inf")
    best_val_sr = -1.0
    epochs_without_improvement = 0

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        avg_loss, global_step = train_one_epoch(
            model, dataloader, optimizer, scheduler, loss_fn,
            scaler, device, epoch, args, global_step,
            use_amp, amp_dtype,
        )
        elapsed = time.time() - t0

        print(f"Epoch {epoch:4d}/{args.epochs} | "
              f"avg_loss={avg_loss:.4f} | best={best_loss:.4f} | {elapsed:.1f}s",
              flush=True)

        # Save checkpoint at interval
        if (epoch + 1) % args.save_every == 0:
            ckpt_path = os.path.join(
                args.checkpoint_dir, f"epoch_{epoch:04d}.pt"
            )
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, avg_loss, ckpt_path,
            )

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, global_step, avg_loss,
                os.path.join(args.checkpoint_dir, "best_loss.pt"),
            )
            if best_val_sr < 0.0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, avg_loss,
                    os.path.join(args.checkpoint_dir, "best.pt"),
                )

        # Save best + early stopping tracking via validation
        if (
            args.eval_every > 0
            and not args.smoke_test
            and HAS_METAWORLD
            and (epoch + 1) % args.eval_every == 0
        ):
            print(f"\n[Validation] Running MT10 evaluation at epoch {epoch}...")
            was_training = model.training
            model.eval()
            try:
                current_val_sr = run_mt10_validation(
                    model=model,
                    device=device,
                    views=views,
                    task_lang_ids=val_task_lang_ids,
                    args=args,
                    obs_length=data_cfg.get("obs_length", 2),
                )
            finally:
                if was_training:
                    model.train()

            print(f"[Validation] MT10 Success Rate: {current_val_sr:.1%}", flush=True)
            if args.wandb and wandb is not None:
                wandb.log({
                    "val/mt10_sr": current_val_sr,
                    "val/epoch": epoch,
                    "val/step": global_step,
                })

            if current_val_sr > best_val_sr:
                best_val_sr = current_val_sr
                epochs_without_improvement = 0
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, avg_loss,
                    os.path.join(args.checkpoint_dir, "best_eval.pt"),
                )
                print(f"  [Validation] New best validation SR: {best_val_sr:.1%}! Saved best_eval.pt")
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step, avg_loss,
                    os.path.join(args.checkpoint_dir, "best.pt"),
                )
            else:
                epochs_without_improvement += 1
        elif args.eval_every > 0 and (epoch + 1) % args.eval_every == 0 and not HAS_METAWORLD:
            print("[Validation] MetaWorld not available; skipping rollout validation.")

        # Early stopping based on validation cycles
        if (args.early_stop_patience > 0
                and epochs_without_improvement >= args.early_stop_patience):
            print(f"\n[EARLY STOP] No validation SR improvement for {args.early_stop_patience} validation cycles. "
                  f"Best validation SR: {best_val_sr:.1%}", flush=True)
            break

        if args.smoke_test:
            print("\n[SMOKE TEST] Completed successfully!")
            break

    print(
        f"\nTraining complete. Best loss: {best_loss:.4f} | "
        f"Best validation SR: {best_val_sr:.1%}" if best_val_sr >= 0.0
        else f"\nTraining complete. Best loss: {best_loss:.4f}",
        flush=True,
    )

    if args.wandb and wandb is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
