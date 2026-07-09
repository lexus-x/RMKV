"""Failure-Aware Retry evaluation for RCAR-VLA on MT-10.

Three inference modes:
  - none:    1 attempt per task instance (vanilla SR1).
  - failure: up to k attempts; retry only if reliability head signals failure
             (P(fail)>tau_f for >=fail_persist steps, or mode==STOP for >=stop_persist steps).
             If head never triggers, no retry.
  - always:  always retry until success or k attempts exhausted (oracle upper bound).

Reports per-task SR, overall SR, average attempts used (efficiency), and
P(fail) statistics for calibration analysis.
"""
import argparse
import json
import os
import time
from pathlib import Path

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T

import metaworld
from transformers import AutoTokenizer

from models.rcar_vla import RCARVLA


MT10_TASKS = [
    "reach", "push", "pick-place", "door-open", "drawer-open",
    "drawer-close", "button-press-topdown", "peg-insert-side",
    "window-open", "window-close",
]


def build_eval_suite(task_name, seed):
    for cand in [f"{task_name}-v3", f"{task_name}-v2", task_name]:
        try:
            ml1 = metaworld.ML1(cand, seed=seed)
            return ml1, list(ml1.train_classes.values())[0]
        except Exception:
            continue
    raise RuntimeError(task_name)


def make_env(env_cls):
    try:
        return env_cls(render_mode="rgb_array")
    except TypeError:
        return env_cls()


def reset_env(env, seed):
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(seed)
    try:
        return env.reset(seed=seed)
    except TypeError:
        if hasattr(env, "seed"):
            env.seed(seed)
        return env.reset()


def render_view(env, cam_name):
    try:
        return env.render(camera_name=cam_name)
    except TypeError:
        renderer = env.unwrapped.mujoco_renderer
        model = env.unwrapped.model
        try:
            cam_id = model.camera(cam_name).id
        except Exception:
            cam_id = model.cam(cam_name).id
        prev = getattr(renderer, "camera_id", None)
        renderer.camera_id = cam_id
        try:
            return env.render()
        finally:
            renderer.camera_id = prev


_TFM = T.Compose([T.ToPILImage(), T.Resize((224, 224)), T.ToTensor()])


def preprocess_views(rgb_list):
    return torch.stack([_TFM(img) for img in rgb_list]).unsqueeze(0)


def make_lang_ids(tokenizer, task_name, max_tokens=32):
    desc = f"Perform the {task_name.replace('-', ' ')} task"
    enc = tokenizer(desc, padding="max_length", max_length=max_tokens,
                    truncation=True, return_tensors="pt")
    return enc.input_ids.long()


def single_attempt(model, env, lang_ids, device, max_steps, tau_f,
                   fail_persist, stop_persist, seed):
    """Run one attempt; return (success, head_triggered, p_fail_max, steps_used)."""
    obs_t = reset_env(env, seed=seed)
    obs = obs_t[0] if isinstance(obs_t, tuple) else obs_t
    success = False
    head_triggered = False
    p_fail_max = 0.0
    fail_count = 0
    stop_count = 0

    for step in range(max_steps):
        rgbs = [render_view(env, "corner2"), render_view(env, "gripperPOV")]
        images = preprocess_views(rgbs).to(device)
        proprio = torch.from_numpy(np.asarray(obs[:15], dtype=np.float32)).unsqueeze(0).to(device)
        batch = {"images": images, "proprio": proprio, "lang_ids": lang_ids.to(device)}

        with torch.no_grad():
            a, r = model.predict_action(
                batch, tau_f=tau_f, use_gating=False, use_self_correction=False
            )
        action = a[0, 0].cpu().numpy().clip(-1, 1)

        # Track reliability signals
        p_fail = F.softmax(r.failure_logits, dim=-1)[0, 1].item()
        mode_pred = r.mode_logits.argmax(dim=-1).item()
        p_fail_max = max(p_fail_max, p_fail)
        fail_count = fail_count + 1 if p_fail > tau_f else 0
        stop_count = stop_count + 1 if mode_pred == 1 else 0
        if fail_count >= fail_persist or stop_count >= stop_persist:
            head_triggered = True
            return False, True, p_fail_max, step + 1

        sr = env.step(action)
        if len(sr) == 5:
            obs, _, term, trunc, info = sr
        else:
            obs, _, done, info = sr
            term, trunc = done, False

        if info.get("success"):
            success = True
            break
        if term or trunc:
            break

    return success, head_triggered, p_fail_max, step + 1


def run_task(model, ml1, env_cls, task, lang_ids, device, args):
    """Run num_rollouts on a single task with the given retry strategy."""
    rollouts = []
    for r_idx in range(args.num_rollouts):
        # Fixed task instance per rollout (same goal pose across attempts)
        task_inst = ml1.train_tasks[r_idx % len(ml1.train_tasks)]

        attempt_results = []
        max_attempts = args.max_attempts if args.retry_mode != "none" else 1

        for attempt_i in range(max_attempts):
            env = make_env(env_cls)
            env.set_task(task_inst)
            seed = args.seed * 10000 + r_idx * 10 + attempt_i

            success, triggered, p_fail_max, n_steps = single_attempt(
                model, env, lang_ids, device,
                max_steps=args.max_steps, tau_f=args.tau_f,
                fail_persist=args.fail_persist, stop_persist=args.stop_persist,
                seed=seed,
            )
            attempt_results.append({
                "success": success, "triggered": triggered,
                "p_fail_max": p_fail_max, "n_steps": n_steps,
            })
            try:
                env.close()
            except Exception:
                pass

            if success:
                break
            # Decide whether to retry
            if args.retry_mode == "none":
                break
            elif args.retry_mode == "failure":
                # Only retry if head triggered (model predicted failure)
                if not triggered:
                    break
            elif args.retry_mode == "always":
                continue  # always retry until success or budget exhausted

        # Top-1 success: did any attempt succeed
        any_success = any(a["success"] for a in attempt_results)
        rollouts.append({
            "any_success": any_success,
            "n_attempts": len(attempt_results),
            "attempts": attempt_results,
        })
    return rollouts


def evaluate(args):
    device = torch.device(args.device)
    print(f"[eval-retry] mode={args.retry_mode} max_attempts={args.max_attempts} "
          f"tau_f={args.tau_f} fail_persist={args.fail_persist}")

    model = RCARVLA(action_dim=4, horizon=args.horizon).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()

    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    results = {}
    overall_succ = 0
    overall_total = 0
    overall_attempts = 0
    t0 = time.time()

    for task in MT10_TASKS:
        ml1, env_cls = build_eval_suite(task, seed=args.seed)
        lang_ids = make_lang_ids(tok, task)
        ros = run_task(model, ml1, env_cls, task, lang_ids, device, args)

        succ = sum(1 for r in ros if r["any_success"])
        avg_att = float(np.mean([r["n_attempts"] for r in ros]))
        avg_pfail_succ = float(np.mean(
            [a["p_fail_max"] for r in ros for a in r["attempts"] if a["success"]] or [0]))
        avg_pfail_fail = float(np.mean(
            [a["p_fail_max"] for r in ros for a in r["attempts"] if not a["success"]] or [0]))

        sr = succ / args.num_rollouts
        results[task] = {
            "sr": sr, "n": args.num_rollouts, "successes": succ,
            "avg_attempts": avg_att,
            "avg_pfail_max_on_success": avg_pfail_succ,
            "avg_pfail_max_on_failure": avg_pfail_fail,
            "rollouts": ros,
        }
        overall_succ += succ
        overall_total += args.num_rollouts
        overall_attempts += sum(r["n_attempts"] for r in ros)
        print(f"  [{task:24s}] SR={sr*100:5.1f}%  ({succ}/{args.num_rollouts}) "
              f"avg_attempts={avg_att:.2f}")

    overall_sr = overall_succ / overall_total
    avg_attempts_global = overall_attempts / overall_total
    elapsed = (time.time() - t0) / 60
    print(f"\n=== OVERALL MT-10 SR ({args.retry_mode}) = {overall_sr*100:.1f}% "
          f"| avg_attempts={avg_attempts_global:.2f} | wall={elapsed:.1f}min ===")

    out = {
        "retry_mode": args.retry_mode,
        "overall_sr": overall_sr,
        "avg_attempts": avg_attempts_global,
        "per_task": results,
        "args": vars(args),
        "elapsed_min": elapsed,
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(args.output_dir) / f"retry_{args.retry_mode}.json"
    with open(fp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {fp}")
    return overall_sr


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--retry_mode", choices=["none", "failure", "always"], default="none")
    p.add_argument("--max_attempts", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_rollouts", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--tau_f", type=float, default=0.5)
    p.add_argument("--fail_persist", type=int, default=5)
    p.add_argument("--stop_persist", type=int, default=3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="outputs/eval/rcar_retry")
    args = p.parse_args()
    evaluate(args)
