"""Evaluation with video recording for RCAR-VLA."""
import argparse
import os
from pathlib import Path

if "DISPLAY" not in os.environ:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
import imageio
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
    tensors = [_TFM(img) for img in rgb_list]
    return torch.stack(tensors).unsqueeze(0)

def make_lang_ids(tokenizer, task_name, max_tokens=32):
    desc = f"Perform the {task_name.replace('-', ' ')} task"
    enc = tokenizer(desc, padding="max_length", max_length=max_tokens,
                    truncation=True, return_tensors="pt")
    return enc.input_ids.long()

def rollout_and_record(model, env, task_name, lang_ids, device, output_path, max_steps=200):
    obs_t = reset_env(env, seed=0)
    obs = obs_t[0] if isinstance(obs_t, tuple) else obs_t
    
    writer = imageio.get_writer(output_path, fps=20)
    
    chunk = None
    chunk_idx = 0

    for step in range(max_steps):
        # Render for video (high res)
        frame = render_view(env, "corner2")
        writer.append_data(frame)
        
        if chunk is None or chunk_idx >= model.horizon:
            rgbs = [render_view(env, "corner2"), render_view(env, "gripperPOV")]
            images = preprocess_views(rgbs).to(device)
            proprio = torch.from_numpy(np.asarray(obs[:7], dtype=np.float32)).unsqueeze(0).to(device)

            batch = {"images": images, "proprio": proprio, "lang_ids": lang_ids.to(device)}
            with torch.no_grad():
                a, r = model.predict_action(batch, use_gating=False, use_self_correction=False)
            chunk = a[0].detach().cpu().numpy()
            chunk_idx = 0

        action = chunk[chunk_idx].clip(-1, 1)
        chunk_idx += 1

        sr = env.step(action)
        if len(sr) == 5:
            obs, _, term, trunc, info = sr
        else:
            obs, _, done, info = sr
            term, trunc = done, False
            
        if info.get("success") or term or trunc:
            # Final frame
            frame = render_view(env, "corner2")
            writer.append_data(frame)
            break
            
    writer.close()
    return info.get("success", False)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output_dir", default="outputs/videos")
    args = p.parse_args()

    device = torch.device(args.device)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    model = RCARVLA(action_dim=4, horizon=4).to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    sd = ckpt.get("model", ckpt)
    model.load_state_dict(sd, strict=False)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for task in MT10_TASKS:
        print(f"Recording {task}...")
        try:
            ml1, env_cls = build_eval_suite(task, seed=0)
            env = make_env(env_cls)
            env.set_task(ml1.train_tasks[0])
            lang_ids = make_lang_ids(tokenizer, task)
            
            video_path = Path(args.output_dir) / f"{task}.mp4"
            success = rollout_and_record(model, env, task, lang_ids, device, str(video_path))
            print(f"  Result: {'Success' if success else 'Failure'}")
            env.close()
        except Exception as e:
            print(f"  Failed {task}: {e}")

if __name__ == "__main__":
    main()
