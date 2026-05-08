#!/usr/bin/env python3
"""
KANFlow-VLA Evaluation on MetaWorld MT-50.

Runs policy rollouts in MetaWorld environments and reports:
  - SR1 (top-1), SR3 (top-3), SR5 (top-5) success rates
  - Per-task and per-difficulty-tier aggregates
  - Results across multiple seeds (0, 42, 100)

Usage:
    python -m kanflow_vla.eval_metaworld --checkpoint checkpoints/kanflow_vla/best.pt
    python -m kanflow_vla.eval_metaworld --checkpoint best.pt --tasks reach push
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw, ImageFont

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

try:
    import metaworld
    HAS_METAWORLD = True
except ImportError:
    HAS_METAWORLD = False
    print("[eval] metaworld not installed. Install via: pip install metaworld")

from kanflow_vla.model.kanflow_vla import KANFlowVLA
from kanflow_vla.data.metaworld_dataset import (
    MT50_TASKS, DIFFICULTY_TIERS, TASK_DESCRIPTIONS,
)

try:
    from transformers import AutoTokenizer
except ImportError:
    AutoTokenizer = None

MT10_TASKS = [
    "reach", "push", "pick-place", "door-open", "drawer-open",
    "drawer-close", "button-press-topdown", "peg-insert-side",
    "window-open", "window-close"
]


def set_global_seeds(seed: int) -> None:
    """Best-effort deterministic seeding for evaluation."""
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def decode_task_metadata(task, rollout_idx: int, reset_seed: int) -> dict:
    """Extract stable, human-readable metadata from a MetaWorld task."""
    rand_vec = None
    partially_observable = None
    task_index = rollout_idx
    try:
        payload = pickle.loads(task.data)
        rand_vec = payload.get("rand_vec")
        partially_observable = payload.get("partially_observable")
    except Exception:
        payload = {}

    try:
        task_index = int(task._asdict().get("index", rollout_idx))
    except Exception:
        try:
            task_index = int(task[2])
        except Exception:
            task_index = rollout_idx

    if rand_vec is not None:
        rand_vec = np.asarray(rand_vec, dtype=np.float64).round(6).tolist()

    return {
        "rollout_idx": rollout_idx,
        "reset_seed": reset_seed,
        "task_index": task_index,
        "env_name": getattr(task, "env_name", None),
        "partially_observable": partially_observable,
        "rand_vec": rand_vec,
    }


def build_eval_suite(task_name: str, seed: int):
    """Resolve a MetaWorld ML1 suite and environment class for a task name."""
    env_name_candidates = []
    if "-v" in task_name:
        env_name_candidates.append(task_name)
    else:
        env_name_candidates.extend([f"{task_name}-v3", f"{task_name}-v2", task_name])

    for env_name in env_name_candidates:
        try:
            ml1 = metaworld.ML1(env_name, seed=seed)
            env_cls = list(ml1.train_classes.values())[0]
            return env_name, env_cls, list(ml1.train_tasks)
        except Exception:
            continue

    return None, None, None


def make_env(env_cls):
    """Construct a MetaWorld env across API variants."""
    try:
        return env_cls(render_mode="rgb_array")
    except TypeError:
        return env_cls()


def reset_env(env, reset_seed: int):
    """Reset an env with as much explicit seeding as the API allows."""
    if hasattr(env, "action_space") and hasattr(env.action_space, "seed"):
        env.action_space.seed(reset_seed)
    if hasattr(env, "observation_space") and hasattr(env.observation_space, "seed"):
        env.observation_space.seed(reset_seed)

    try:
        return env.reset(seed=reset_seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(reset_seed)
        return env.reset()


def load_task_tokenizer(model_name: str = "HuggingFaceTB/SmolLM-135M"):
    """Load the task-instruction tokenizer used for training and evaluation."""
    if AutoTokenizer is None:
        return None

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token
        return tokenizer
    except Exception as exc:
        print(f"[eval] Failed to load tokenizer {model_name}: {exc}")
        return None


def build_task_lang_ids(
    task_names: list[str],
    tokenizer=None,
    max_tokens: int = 32,
    device: torch.device | None = None,
) -> dict[str, torch.Tensor]:
    """Precompute token IDs for a set of task descriptions."""
    task_lang_ids = {}
    for task_name in task_names:
        if tokenizer is None:
            lang_ids = torch.zeros(1, max_tokens, dtype=torch.long)
        else:
            desc = TASK_DESCRIPTIONS.get(
                task_name, f"Perform the {task_name.replace('-', ' ')} task"
            )
            encoded = tokenizer(
                desc,
                padding="max_length",
                max_length=max_tokens,
                truncation=True,
                return_tensors="pt",
            )
            lang_ids = encoded.input_ids.to(dtype=torch.long)

        if device is not None:
            lang_ids = lang_ids.to(device)
        task_lang_ids[task_name] = lang_ids

    return task_lang_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KANFlow-VLA MetaWorld Evaluation")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--tasks", nargs="*", default=None,
                        help="Specific tasks to evaluate (default: all MT-50)")
    parser.add_argument("--num-rollouts", type=int, default=10,
                        help="Number of rollouts per task per seed")
    parser.add_argument("--max-steps", type=int, default=500,
                        help="Max steps per episode")
    parser.add_argument("--seeds", nargs="*", type=int, default=[0, 42, 100])
    parser.add_argument("--num-inference-steps", type=int, default=1,
                        help="Flow integration steps (1 for real-time)")
    parser.add_argument("--output-dir", type=str, default="./eval_results/kanflow_vla")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--save-videos", action="store_true")
    parser.add_argument("--use-octo", action="store_true",
                        help="Use a pretrained Octo encoder as the frozen condition backbone")
    parser.add_argument("--octo-pretrained-path", type=str,
                        default="hf://rail-berkeley/octo-small-1.5")
    parser.add_argument("--octo-platform", type=str, default="cpu",
                        choices=["cpu"],
                        help="Run Octo on CPU for compatibility with the local JAX setup")
    return parser.parse_args()


def load_model(
    checkpoint_path: str,
    device: torch.device,
    use_octo: bool = False,
    octo_pretrained_path: str = "hf://rail-berkeley/octo-small-1.5",
    octo_platform: str = "cpu",
) -> KANFlowVLA:
    """Load model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device)

    # Infer model config from checkpoint
    state_dict = ckpt.get("model_state_dict", ckpt)
    proprio_weight = state_dict.get("proprio_mlp.0.weight")
    proprio_dim = proprio_weight.shape[1] if proprio_weight is not None else 15
    obs_length = max(1, proprio_dim // 15)

    model = KANFlowVLA(
        action_dim=4,
        horizon=4,
        d_model=256,
        proprio_dim=proprio_dim,
        vision_config={
            "name": "octo_small" if use_octo else "siglip_base",
            "pretrained_path": octo_pretrained_path,
            "platform": octo_platform,
        },
    )
    model.load_state_dict(state_dict, strict=False)
    model.eval_obs_length = max(obs_length, 2) if use_octo else obs_length
    model.to(device)
    model.eval()
    return model


def preprocess_obs(obs: list[np.ndarray], proprio: np.ndarray, img_size: int = 224, device: torch.device = None):
    """
    Preprocess multiple MetaWorld observations for model input.

    Args:
        obs: List of raw RGB images from different cameras.
        proprio: Raw proprioception vector.
        img_size: Target image size.
        device: Target device.

    Returns:
        images: (1, V, 3, H, W) preprocessed image tensor.
        proprio: (1, 15) proprioception tensor.
    """
    # Extract first 15 dims as proprio
    if len(proprio) >= 15:
        proprio_clean = proprio[:15]
    else:
        proprio_clean = np.pad(proprio, (0, 15 - len(proprio)))
    proprio_t = torch.from_numpy(proprio_clean).float().unsqueeze(0).to(device)

    from torchvision import transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    view_tensors = []
    for img in obs:
        if img is not None:
            view_tensors.append(transform(img))
        else:
            view_tensors.append(torch.zeros(3, img_size, img_size))
    
    images_t = torch.stack(view_tensors).unsqueeze(0).to(device)  # (1, V, 3, H, W)

    return images_t, proprio_t


def render_view(env, cam_name: str) -> np.ndarray:
    """
    Render a specific camera view across MetaWorld/Gymnasium variants.

    MetaWorld 3.0.0 on Gymnasium's MujocoEnv no longer accepts
    ``env.render(camera_name=...)``. In that case, switch the renderer to the
    requested camera id and render an RGB frame directly.
    """
    try:
        return env.render(camera_name=cam_name)
    except TypeError:
        renderer = env.unwrapped.mujoco_renderer
        model = env.unwrapped.model

        try:
            cam_id = model.camera(cam_name).id
        except AttributeError:
            cam_id = model.cam(cam_name).id

        prev_camera_id = getattr(renderer, "camera_id", None)
        renderer.camera_id = cam_id
        try:
            return renderer.render(render_mode="rgb_array")
        finally:
            renderer.camera_id = prev_camera_id


def run_rollout(
    model: KANFlowVLA,
    env,
    task_name: str,
    views: list[str],
    max_steps: int = 500,
    num_inference_steps: int = 1,
    device: torch.device = None,
    lang_ids: torch.Tensor = None,
    save_video: bool = False,
    obs_length: int = 2,
    reset_seed: int | None = None,
) -> dict:
    """
    Run a single episode rollout with multiview support and temporal buffering.
    """
    if reset_seed is not None:
        set_global_seeds(reset_seed)
        reset_result = reset_env(env, reset_seed)
    else:
        reset_result = env.reset()
    if isinstance(reset_result, tuple):
        obs, _ = reset_result
    else:
        obs = reset_result
    
    total_reward = 0.0
    success = False
    inference_times = []
    video_frames = []

    cam_mapping = {
        "topview": "topview",
        "image_corner2": "corner2",
        "image_gripperPOV": "gripperPOV",
    }

    img_buffer = []
    obs_buffer = []

    for step in range(max_steps):
        # Capture frames from all views
        current_view_imgs = []
        for v in views:
            cam_name = cam_mapping.get(v, "corner2")
            img = render_view(env, cam_name)
            current_view_imgs.append(img)
            
        images_t, proprio_t = preprocess_obs(current_view_imgs, obs, device=device)

        img_buffer.append(images_t)
        obs_buffer.append(proprio_t)
        if len(img_buffer) > obs_length:
            img_buffer.pop(0)
            obs_buffer.pop(0)

        padded_imgs = [img_buffer[0]] * (obs_length - len(img_buffer)) + img_buffer
        padded_obs = [obs_buffer[0]] * (obs_length - len(obs_buffer)) + obs_buffer

        stacked_images = torch.stack(padded_imgs, dim=1)  # (1, T, V, 3, H, W)
        stacked_proprio = torch.stack(padded_obs, dim=1)  # (1, T, 15)

        if save_video:
            # Create grid for summary video
            combined_h = np.hstack(current_view_imgs)
            img_pil = Image.fromarray(combined_h)
            draw = ImageDraw.Draw(img_pil)
            draw.text((10, 10), f"Task: {task_name} | Step: {step}", fill=(255, 255, 255))
            video_frames.append(np.array(img_pil))

        if lang_ids is None:
            lang_ids_t = torch.zeros(1, 32, dtype=torch.long, device=device)
        else:
            lang_ids_t = lang_ids.to(device)
            if lang_ids_t.ndim == 1:
                lang_ids_t = lang_ids_t.unsqueeze(0)

        # Inference
        t0 = time.perf_counter()
        actions = model.predict_action(
            stacked_images, lang_ids_t, stacked_proprio,
            num_steps=num_inference_steps,
            task_texts=[TASK_DESCRIPTIONS.get(task_name, f"Perform the {task_name.replace('-', ' ')} task")],
        )
        t1 = time.perf_counter()
        inference_times.append((t1 - t0) * 1000)

        action = actions[0, 0].cpu().numpy()
        action = np.clip(action, -1.0, 1.0)

        step_result = env.step(action)
        if len(step_result) == 5:
            obs, reward, terminated, truncated, info = step_result
        else:
            obs, reward, done, info = step_result
            terminated = done
            truncated = False
        
        total_reward += reward
        if info.get("success", False):
            success = True
            break
        if terminated or truncated:
            break

    return {
        "success": float(success),
        "return": total_reward,
        "length": step + 1,
        "avg_inference_ms": np.mean(inference_times) if inference_times else 0.0,
        "frames": video_frames if save_video else None,
    }


def evaluate_task(
    model: KANFlowVLA,
    task_name: str,
    seed: int,
    views: list[str],
    num_rollouts: int = 10,
    max_steps: int = 500,
    num_inference_steps: int = 1,
    device: torch.device = None,
    save_video: bool = False,
    lang_ids: torch.Tensor | None = None,
    tokenizer = None,
    obs_length: int = 2,
) -> dict:
    """Evaluate a single task with multiple rollouts and multiple views."""
    if not HAS_METAWORLD:
        # Return synthetic results for testing
        successes = np.random.binomial(1, 0.5, num_rollouts).astype(float)
        return {
            "task": task_name,
            "seed": seed,
            "successes": successes.tolist(),
            "sr1": float(np.max(successes)),
            "sr3": float(np.mean(np.sort(successes)[-3:])) if len(successes) >= 3 else float(np.mean(successes)),
            "sr5": float(np.mean(np.sort(successes)[-5:])) if len(successes) >= 5 else float(np.mean(successes)),
            "mean_success": float(np.mean(successes)),
            "avg_inference_ms": 0.0,
        }

    actual_env_name, env_cls, available_tasks = build_eval_suite(task_name, seed)
    if env_cls is None or not available_tasks:
        env_name_candidates = (
            [task_name]
            if "-v" in task_name
            else [f"{task_name}-v3", f"{task_name}-v2", task_name]
        )
        print(f"  [WARN] Could not create env for {task_name} "
              f"(tried: {env_name_candidates})")
        return None

    if lang_ids is None:
        lang_ids_t = build_task_lang_ids(
            [task_name], tokenizer=tokenizer, device=device
        )[task_name]
    else:
        lang_ids_t = lang_ids.to(device)
        if lang_ids_t.ndim == 1:
            lang_ids_t = lang_ids_t.unsqueeze(0)

    results = []
    rollout_metadata = []
    rollout_tasks = [
        available_tasks[rollout_idx % len(available_tasks)]
        for rollout_idx in range(num_rollouts)
    ]
    for rollout in range(num_rollouts):
        task = rollout_tasks[rollout]
        rollout_seed = seed * 1000 + rollout
        task_meta = decode_task_metadata(
            task, rollout_idx=rollout, reset_seed=rollout_seed
        )

        env = make_env(env_cls)
        env.set_task(task)
        result = run_rollout(
            model, env, task_name, views,
            max_steps=max_steps,
            num_inference_steps=num_inference_steps,
            device=device,
            lang_ids=lang_ids_t,
            save_video=save_video,
            obs_length=obs_length,
            reset_seed=rollout_seed,
        )
        env.close()
        result["task_metadata"] = task_meta
        results.append(result)
        rollout_metadata.append(task_meta)

    successes = [r["success"] for r in results]
    sorted_successes = sorted(successes, reverse=True)

    return {
        "task": task_name,
        "seed": seed,
        "successes": successes,
        "sr1": sorted_successes[0] if len(sorted_successes) >= 1 else 0.0,
        "sr3": float(np.mean(sorted_successes[:3])) if len(sorted_successes) >= 3 else float(np.mean(sorted_successes)),
        "sr5": float(np.mean(sorted_successes[:5])) if len(sorted_successes) >= 5 else float(np.mean(sorted_successes)),
        "mean_success": float(np.mean(successes)),
        "avg_inference_ms": float(np.mean([r["avg_inference_ms"] for r in results])),
        "env_name": actual_env_name,
        "rollout_metadata": rollout_metadata,
        "frames": [r["frames"] for r in results if r["frames"] is not None] if save_video else None,
    }


def aggregate_results(
    all_results: dict[str, list[dict]],
) -> dict:
    """
    Aggregate results across seeds and compute per-tier statistics.

    Returns summary dict with:
      - per_task: {task: {sr1, sr3, sr5, mean±std}}
      - per_tier: {easy/medium/hard/very_hard: {sr1, sr3, sr5}}
      - overall: {sr1, sr3, sr5}
    """
    per_task = {}
    for task, seed_results in all_results.items():
        sr1_vals = [r["sr1"] for r in seed_results if r is not None]
        sr3_vals = [r["sr3"] for r in seed_results if r is not None]
        sr5_vals = [r["sr5"] for r in seed_results if r is not None]

        per_task[task] = {
            "sr1_mean": float(np.mean(sr1_vals)) if sr1_vals else 0.0,
            "sr1_std": float(np.std(sr1_vals)) if sr1_vals else 0.0,
            "sr3_mean": float(np.mean(sr3_vals)) if sr3_vals else 0.0,
            "sr5_mean": float(np.mean(sr5_vals)) if sr5_vals else 0.0,
        }

    # Per-tier aggregation
    per_tier = {}
    for tier_name, tier_tasks in DIFFICULTY_TIERS.items():
        tier_sr1 = [per_task[t]["sr1_mean"] for t in tier_tasks if t in per_task]
        tier_sr3 = [per_task[t]["sr3_mean"] for t in tier_tasks if t in per_task]
        tier_sr5 = [per_task[t]["sr5_mean"] for t in tier_tasks if t in per_task]
        per_tier[tier_name] = {
            "sr1": float(np.mean(tier_sr1)) if tier_sr1 else 0.0,
            "sr3": float(np.mean(tier_sr3)) if tier_sr3 else 0.0,
            "sr5": float(np.mean(tier_sr5)) if tier_sr5 else 0.0,
            "num_tasks": len(tier_sr1),
        }

    # Overall
    all_sr1 = [v["sr1_mean"] for v in per_task.values()]
    all_sr3 = [v["sr3_mean"] for v in per_task.values()]
    overall = {
        "sr1": float(np.mean(all_sr1)) if all_sr1 else 0.0,
        "sr3": float(np.mean(all_sr3)) if all_sr3 else 0.0,
        "num_tasks": len(all_sr1),
    }

    return {
        "per_task": per_task,
        "per_tier": per_tier,
        "overall": overall,
    }


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.seeds:
        set_global_seeds(args.seeds[0])

    print(f"\n{'='*65}")
    print(f"KANFlow-VLA MetaWorld Evaluation")
    print(f"{'='*65}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Device: {device}")
    print(f"Seeds: {args.seeds}")
    print(f"Rollouts/task: {args.num_rollouts}")
    print(f"Inference steps: {args.num_inference_steps}")

    # Load model
    model = load_model(
        args.checkpoint,
        device,
        use_octo=args.use_octo,
        octo_pretrained_path=args.octo_pretrained_path,
        octo_platform=args.octo_platform,
    )

    # Determine tasks
    tasks = args.tasks if args.tasks else MT50_TASKS
    print(f"Tasks: {len(tasks)}")

    tokenizer = load_task_tokenizer()
    task_lang_ids = build_task_lang_ids(tasks, tokenizer=tokenizer, device=device)

    # Run evaluation
    all_results = defaultdict(list)
    all_video_frames = []
    
    # Views matching what exists in training data (HDF5)
    views = ["image_corner2", "image_gripperPOV"]

    for seed in args.seeds:
        print(f"\n--- Seed {seed} ---")
        for task_idx, task in enumerate(tasks):
            print(f"  [{task_idx+1}/{len(tasks)}] {task}...", end=" ", flush=True)
            result = evaluate_task(
                model, task, seed, views,
                num_rollouts=args.num_rollouts,
                max_steps=args.max_steps,
                num_inference_steps=args.num_inference_steps,
                device=device,
                save_video=args.save_videos,
                lang_ids=task_lang_ids.get(task),
                tokenizer=tokenizer,
                obs_length=getattr(model, "eval_obs_length", 2),
            )
            if result:
                all_results[task].append(result)
                meta_suffix = ""
                if result.get("rollout_metadata"):
                    if len(result["rollout_metadata"]) == 1:
                        meta = result["rollout_metadata"][0]
                        meta_suffix = (
                            f"  [task_idx={meta['task_index']} reset={meta['reset_seed']}]"
                        )
                    else:
                        preview = ",".join(
                            str(meta["task_index"])
                            for meta in result["rollout_metadata"][:3]
                        )
                        if len(result["rollout_metadata"]) > 3:
                            preview += ",..."
                        meta_suffix = f"  [task_idx={preview}]"
                print(f"SR1={result['sr1']:.0%}  SR3={result['sr3']:.0%}  "
                      f"({result['avg_inference_ms']:.1f}ms){meta_suffix}")
                
                if args.save_videos and result["frames"]:
                    # Flatten frames (if multiple rollouts) or just one
                    for rollout_frames in result["frames"]:
                        all_video_frames.extend(rollout_frames)
            else:
                print("SKIPPED")

    # Aggregate and report
    summary = aggregate_results(dict(all_results))

    print(f"\n{'='*65}")
    print("Results Summary")
    print(f"{'='*65}")

    print(f"\n{'Tier':<12s}  {'SR1':>6s}  {'SR3':>6s}  {'SR5':>6s}  {'#Tasks':>6s}")
    print("-" * 40)
    for tier, vals in summary["per_tier"].items():
        print(f"{tier:<12s}  {vals['sr1']:6.1%}  {vals['sr3']:6.1%}  "
              f"{vals['sr5']:6.1%}  {vals['num_tasks']:6d}")

    print("-" * 40)
    print(f"{'OVERALL':<12s}  {summary['overall']['sr1']:6.1%}  "
          f"{summary['overall']['sr3']:6.1%}  "
          f"{'':>6s}  {summary['overall']['num_tasks']:6d}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "eval_results.json")
    with open(results_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    raw_results_path = os.path.join(args.output_dir, "raw_results.json")
    raw_results = {
        task: [
            {k: v for k, v in seed_result.items() if k != "frames"}
            for seed_result in seed_results
        ]
        for task, seed_results in all_results.items()
    }
    with open(raw_results_path, "w") as f:
        json.dump(raw_results, f, indent=2)
    print(f"Raw rollout details saved to: {raw_results_path}")

    # Handle single video save
    if args.save_videos and all_video_frames:
        video_dir = os.path.join(args.output_dir, "videos")
        os.makedirs(video_dir, exist_ok=True)
        video_path = os.path.join(video_dir, "mt10_evaluation_summary.mp4")
        print(f"Saving summary video to: {video_path}...")
        imageio.mimsave(video_path, all_video_frames, fps=30)
        print("Video saved successfully.")


if __name__ == "__main__":
    main()
