"""Real MT-10 evaluation for RCAR-VLA. Supports 4 ablation modes."""
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


MT10_TASKS = ["reach", "push", "pick-place", "door-open", "drawer-open"]


def build_eval_suite(task_name: str, seed: int):
    for cand in [f"{task_name}-v3", f"{task_name}-v2", task_name]:
        try:
            ml1 = metaworld.ML1(cand, seed=seed)
            env_cls = list(ml1.train_classes.values())[0]
            return ml1, env_cls
        except Exception:
            continue
    raise RuntimeError(f"could not build suite for {task_name}")


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


_TFM = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def preprocess_views(rgb_list):
    """rgb_list: list of HxWx3 uint8 arrays."""
    tensors = [_TFM(img) for img in rgb_list]
    return torch.stack(tensors).unsqueeze(0)  # (1, V, 3, 224, 224)


def make_lang_ids(tokenizer, task_name, max_tokens=32):
    desc = f"Perform the {task_name.replace('-', ' ')} task"
    enc = tokenizer(desc, padding="max_length", max_length=max_tokens,
                    truncation=True, return_tensors="pt")
    return enc.input_ids.long()


def rollout(model, env, task_name, lang_ids, device, max_steps=200,
            use_gating=False, use_self_correction=False, tau_f=0.5):
    obs_t = reset_env(env, seed=0)
    obs = obs_t[0] if isinstance(obs_t, tuple) else obs_t
    success = False
    stop_count = 0
    fail_count = 0
    
    chunk = None
    chunk_idx = 0

    for step in range(max_steps):
        if chunk is None or chunk_idx >= model.horizon:
            rgbs = [render_view(env, "corner2"), render_view(env, "gripperPOV")]
            images = preprocess_views(rgbs).to(device)
            proprio = torch.from_numpy(np.asarray(obs[:7], dtype=np.float32)).unsqueeze(0).to(device)

            batch = {"images": images, "proprio": proprio, "lang_ids": lang_ids.to(device)}
            with torch.no_grad():
                print("[DEBUG] infer")
                a, r = model.predict_action(
                    batch, tau_f=tau_f,
                    use_gating=use_gating,
                    use_self_correction=use_self_correction,
                )
            chunk = a[0].detach().cpu().numpy()
            chunk_idx = 0

            # Closed-loop gating: terminate early if STOP/CORRUPT persists
            if use_gating:
                mode_pred = r.mode_logits.argmax(dim=-1).item()
                p_fail = F.softmax(r.failure_logits, dim=-1)[0, 1].item()
                stop_count = stop_count + 1 if mode_pred == 1 else 0
                fail_count = fail_count + 1 if p_fail > tau_f else 0
                if stop_count >= 3 or fail_count >= 5:
                    break

        action = chunk[chunk_idx].clip(-1, 1)
        chunk_idx += 1

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
    return success


def evaluate(args):
    device = torch.device(args.device)
    print(f"[eval] device={device} mode={args.mode}")

    use_gating = args.mode in ("gating", "full")
    use_sc = args.mode == "full"

    # ── Load model ──
    model = RCARVLA(action_dim=4, horizon=args.horizon).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"[load] missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    results = {}
    overall_succ = 0
    overall_total = 0
    for task in MT10_TASKS:
        try:
            ml1, env_cls = build_eval_suite(task, seed=args.seed)
        except Exception as e:
            print(f"  [{task}] suite-build-failed: {e}")
            results[task] = {"sr": 0.0, "n": 0, "error": str(e)}
            continue

        lang_ids = make_lang_ids(tokenizer, task)

        succ = 0
        for r_idx in range(args.num_rollouts):
            env = make_env(env_cls)
            env.set_task(ml1.train_tasks[r_idx % len(ml1.train_tasks)])
            ok = rollout(model, env, task, lang_ids, device,
                         max_steps=args.max_steps,
                         use_gating=use_gating,
                         use_self_correction=use_sc,
                         tau_f=args.tau_f)
            if ok:
                succ += 1
            try:
                env.close()
            except Exception:
                pass

        sr = succ / args.num_rollouts
        results[task] = {"sr": sr, "n": args.num_rollouts, "successes": succ}
        overall_succ += succ
        overall_total += args.num_rollouts
        print(f"  [{task:24s}] SR={sr*100:5.1f}%  ({succ}/{args.num_rollouts})")

    overall_sr = overall_succ / overall_total
    print(f"\n=== OVERALL MT-10 SR ({args.mode}) = {overall_sr*100:.1f}% ===")

    out = {"mode": args.mode, "overall_sr": overall_sr,
           "per_task": results, "args": vars(args)}
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    fp = Path(args.output_dir) / f"mt10_{args.mode}.json"
    with open(fp, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[saved] {fp}")
    return overall_sr


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--mode", choices=["base", "aux", "gating", "full"], default="full")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--num_rollouts", type=int, default=10)
    p.add_argument("--max_steps", type=int, default=200)
    p.add_argument("--horizon", type=int, default=4)
    p.add_argument("--tau_f", type=float, default=0.5)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="outputs/eval/rcar_mt10")
    args = p.parse_args()
    evaluate(args)
