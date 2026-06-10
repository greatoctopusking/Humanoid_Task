import argparse
import os
import pickle
import random

import cv2
import gymnasium as gym
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=True if torch.cuda.is_available() else False, action="store_true")
    parser.add_argument("--env", default="Humanoid-v5")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-envs", type=int, default=32)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--n-epochs", type=int, default=76)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--train-iters", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.995)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=1e-3)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--target-kl", type=float, default=0.01)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--render-epoch", type=int, default=10)
    parser.add_argument("--eval-freq", type=int, default=50000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_env(env_id, normalize=True, render=False, fps=30):
    if render:
        env = gym.make(env_id, render_mode="rgb_array")
        env.metadata["render_fps"] = fps
    else:
        env = gym.make(env_id)
    if normalize:
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.NormalizeReward(env)
    return env


def make_eval_env(env_id, seed=None, render=False, fps=30):
    if render:
        env = gym.make(env_id, render_mode="rgb_array")
        env.metadata["render_fps"] = fps
    else:
        env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def save_normalize_params(norm_env, path):
    wrappers = []
    e = norm_env
    while hasattr(e, "env"):
        wrappers.append(e)
        e = e.env
    obs_rms = None
    ret_rms = None
    for w in wrappers:
        if hasattr(w, "obs_rms"):
            obs_rms = {"mean": w.obs_rms.mean.copy(), "var": w.obs_rms.var.copy(), "count": w.obs_rms.count}
    for w in wrappers:
        if hasattr(w, "return_rms"):
            ret_rms = {"mean": w.return_rms.mean, "var": w.return_rms.var, "count": w.return_rms.count}
    params = {"obs_rms": obs_rms, "ret_rms": ret_rms}
    with open(path, "wb") as f:
        pickle.dump(params, f)


def load_normalize_params(obs_norm_env, ret_norm_env, path):
    with open(path, "rb") as f:
        params = pickle.load(f)
    if params["obs_rms"] is not None and obs_norm_env is not None and hasattr(obs_norm_env, "obs_rms"):
        obs_norm_env.obs_rms.mean = params["obs_rms"]["mean"]
        obs_norm_env.obs_rms.var = params["obs_rms"]["var"]
        obs_norm_env.obs_rms.count = params["obs_rms"]["count"]
    if params["ret_rms"] is not None and ret_norm_env is not None and hasattr(ret_norm_env, "return_rms"):
        ret_norm_env.return_rms.mean = params["ret_rms"]["mean"]
        ret_norm_env.return_rms.var = params["ret_rms"]["var"]
        ret_norm_env.return_rms.count = params["ret_rms"]["count"]


def log_video(env, agent, device, video_path, fps=30):
    agent.eval()
    frames = []
    obs, _ = env.reset()
    done = False
    while not done:
        frames.append(env.render())
        with torch.no_grad():
            action, _, _, _ = agent.get_action_and_value(
                torch.tensor(np.array([obs], dtype=np.float32), device=device)
            )
        obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
