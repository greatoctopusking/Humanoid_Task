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
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--total-steps", type=int, default=5_000_000)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--alpha-lr", type=float, default=3e-4)
    parser.add_argument("--start-steps", type=int, default=10000)
    parser.add_argument("--updates-per-step", type=int, default=1)
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
        env = gym.make(env_id, render_mode="human")
    else:
        env = gym.make(env_id)
    env.reset(seed=seed)
    return env


def save_normalize_params(env, path):
    obs_rms = None
    ret_rms = None
    w = env
    while hasattr(w, "env"):
        if hasattr(w, "obs_rms") and obs_rms is None:
            obs_rms = w.obs_rms
        if hasattr(w, "return_rms") and ret_rms is None:
            ret_rms = w.return_rms
        w = w.env

    params = {
        "obs_rms": {"mean": obs_rms.mean.copy(), "var": obs_rms.var.copy(), "count": obs_rms.count}
        if obs_rms is not None else None,
        "ret_rms": {"mean": ret_rms.mean, "var": ret_rms.var, "count": ret_rms.count}
        if ret_rms is not None else None,
    }
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
            action, _ = agent.get_action(
                torch.tensor(np.array([obs], dtype=np.float32), device=device),
                deterministic=True,
            )
        obs, _, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
        done = terminated or truncated
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (frames[0].shape[1], frames[0].shape[0]))
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    out.release()
