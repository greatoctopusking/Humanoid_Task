import argparse

import gymnasium as gym
import numpy as np
import torch

from lib.agent import SACAgent
from lib.utils import make_eval_env, load_normalize_params


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

    env = make_eval_env(args.env, seed=args.seed, render=args.render)
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

    if args.normalize and args.normalize != "":
        obs_norm_env = None
        ret_norm_env = None
        norm_env = gym.make(args.env)
        norm_env = gym.wrappers.NormalizeObservation(norm_env)
        norm_env = gym.wrappers.NormalizeReward(norm_env)
        w = norm_env
        while hasattr(w, "env"):
            if hasattr(w, "obs_rms") and obs_norm_env is None:
                obs_norm_env = w
            if hasattr(w, "return_rms") and ret_norm_env is None:
                ret_norm_env = w
            w = w.env
        load_normalize_params(obs_norm_env, ret_norm_env, args.normalize)
        norm_env.close()

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
