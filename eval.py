import argparse
import pickle

import gymnasium as gym
import numpy as np
import torch

from lib.agent import SACAgent
from lib.utils import make_eval_env, extract_obs_rms


def parse_eval_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/best.pt")
    parser.add_argument("--normalize", type=str, default="checkpoints/normalize.pkl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--env", default="Humanoid-v5")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_eval_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.render:
        env = gym.make(args.env, render_mode="human")
    else:
        env = gym.make(args.env)
    env.reset(seed=args.seed)

    if args.normalize and args.normalize != "":
        env = gym.wrappers.NormalizeObservation(env)
        obs_rms = extract_obs_rms(env)
        if obs_rms is not None:
            with open(args.normalize, "rb") as f:
                params = pickle.load(f)
            if params["obs_rms"] is not None:
                obs_rms.mean = params["obs_rms"]["mean"]
                obs_rms.var = params["obs_rms"]["var"]
                obs_rms.count = params["obs_rms"]["count"]

    obs_dim = env.observation_space.shape
    action_dim = env.action_space.shape
    action_low = float(env.action_space.low[0])
    action_high = float(env.action_space.high[0])

    agent = SACAgent(obs_dim[0], action_dim[0], action_low, action_high).to(device)

    try:
        agent.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))
    except Exception:
        agent.load_state_dict(torch.load(args.model, map_location=device))

    agent.eval()

    episode_rewards = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        total_reward = 0.0
        step_count = 0
        while not done:
            if args.render:
                env.render()
            with torch.no_grad():
                action, _ = agent.get_action(
                    torch.tensor(np.array([obs], dtype=np.float32), device=device),
                    deterministic=True,
                )
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            total_reward += reward
            step_count += 1
            done = terminated or truncated
        episode_rewards.append(total_reward)
        print(f"Episode {ep + 1}: reward = {total_reward:.2f}, steps = {step_count}")

    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"\n=== Result ===")
    print(f"Seed: {args.seed}")
    print(f"Episodes: {args.episodes}")
    print(f"Mean reward: {mean_reward:.2f}")
    print(f"Std reward: {std_reward:.2f}")
    print(f"Min reward: {np.min(episode_rewards):.2f}")
    print(f"Max reward: {np.max(episode_rewards):.2f}")

    env.close()
