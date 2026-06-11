import datetime
import os
import sys
import time

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from lib.agent import SACAgent
from lib.buffer import ReplayBuffer
from lib.utils import parse_args, set_seed, make_env, make_eval_env, save_normalize_params, log_video


class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.stdout.flush()
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def evaluate(agent, env, device, num_episodes=10):
    agent.eval()
    episode_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                action, _ = agent.get_action(
                    torch.tensor(np.array([obs], dtype=np.float32), device=device),
                    deterministic=True,
                )
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


def sac_update(agent, critic_optim, actor_optim, alpha_optim,
               batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones,
               gamma, tau, update_actor,
               scaler_critic, scaler_actor, scaler_alpha, device):
    alpha = agent.alpha.detach()

    with torch.no_grad():
        next_actions, next_log_probs = agent.get_action(batch_next_obs)
        next_log_probs = next_log_probs.unsqueeze(-1)
        q1_next, q2_next = agent.get_q_values_target(batch_next_obs, next_actions)
        q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
        q_target = batch_rewards + gamma * (1.0 - batch_dones) * q_next
        q_target = torch.clamp(q_target, -100.0, 100.0)

    with torch.amp.autocast(str(device)):
        q1, q2 = agent.get_q_values(batch_obs, batch_actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

    critic_params = list(agent.critic_1.parameters()) + list(agent.critic_2.parameters())
    critic_optim.zero_grad()
    scaler_critic.scale(critic_loss).backward()
    scaler_critic.unscale_(critic_optim)
    nn.utils.clip_grad_norm_(critic_params, 1.0)
    scaler_critic.step(critic_optim)
    scaler_critic.update()

    actor_loss_val = 0.0
    actor_loss = torch.tensor(0.0, device=device)
    alpha_loss_val = 0.0

    if update_actor:
        with torch.amp.autocast(str(device)):
            new_actions, new_log_probs = agent.get_action(batch_obs)
            new_log_probs_sq = new_log_probs.unsqueeze(-1)
            q1_new, q2_new = agent.get_q_values(batch_obs, new_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (alpha * new_log_probs_sq - q_new).mean()

        actor_params = list(agent.actor_fc.parameters()) + \
                       [agent.actor_mu.weight, agent.actor_mu.bias,
                        agent.actor_log_std.weight, agent.actor_log_std.bias]
        actor_optim.zero_grad()
        scaler_actor.scale(actor_loss).backward()
        scaler_actor.unscale_(actor_optim)
        nn.utils.clip_grad_norm_(actor_params, 1.0)
        scaler_actor.step(actor_optim)
        scaler_actor.update()
        actor_loss_val = actor_loss.item()

        with torch.amp.autocast(str(device)):
            alpha_loss = -(agent.log_alpha * (new_log_probs.detach() + agent.target_entropy)).mean()

        alpha_optim.zero_grad()
        scaler_alpha.scale(alpha_loss).backward()
        scaler_alpha.unscale_(alpha_optim)
        scaler_alpha.step(alpha_optim)
        scaler_alpha.update()
        alpha_loss_val = alpha_loss.item()

    agent.soft_update(tau)

    return critic_loss.item(), actor_loss_val, alpha_loss_val, alpha.item()


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(__file__)
    folder_name = f"sac_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    videos_dir = os.path.join(current_dir, "videos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)
    checkpoint_dir = os.path.join(current_dir, "checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = os.path.join(current_dir, "logs", folder_name)
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "train.log")
    tee = Tee(log_file)
    sys.stdout = tee
    writer = SummaryWriter(log_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    train_env = make_env(args.env, normalize=True)
    raw_eval_env = make_eval_env(args.env, seed=args.seed)
    test_video_env = gym.make(args.env, render_mode="rgb_array")
    test_video_env.metadata["render_fps"] = 30
    test_video_env.reset(seed=args.seed)

    obs_dim = train_env.observation_space.shape
    act_dim = train_env.action_space.shape
    action_low = float(train_env.action_space.low[0])
    action_high = float(train_env.action_space.high[0])

    agent = SACAgent(obs_dim[0], act_dim[0], action_low, action_high).to(device)

    actor_params = list(agent.actor_fc.parameters()) + \
                   [agent.actor_mu.weight, agent.actor_mu.bias,
                    agent.actor_log_std.weight, agent.actor_log_std.bias]
    critic_params = list(agent.critic_1.parameters()) + list(agent.critic_2.parameters())

    actor_optim = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)
    critic_optim = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)
    alpha_optim = optim.Adam([agent.log_alpha], lr=args.alpha_lr, eps=1e-5)

    scaler_critic = torch.amp.GradScaler(str(device))
    scaler_actor = torch.amp.GradScaler(str(device))
    scaler_alpha = torch.amp.GradScaler(str(device))

    buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_size, device)

    print(agent)

    global_step = 0
    best_mean_reward = -np.inf
    episode_reward = 0.0
    episode_steps = 0
    critic_update_count = 0
    actor_update_freq = 2

    obs, _ = train_env.reset()

    try:
        pbar = tqdm(total=args.total_steps, desc="Training")
        while global_step < args.total_steps:
            if global_step < args.start_steps:
                action = train_env.action_space.sample()
            else:
                with torch.no_grad():
                    action, _ = agent.get_action(
                        torch.tensor(np.array([obs], dtype=np.float32), device=device)
                    )
                    action = action.squeeze(0).cpu().numpy()

            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            real_done = float(terminated)
            done = terminated or truncated

            buffer.store(obs, action, reward, next_obs, real_done)

            obs = next_obs
            episode_reward += reward
            episode_steps += 1
            global_step += 1
            pbar.update(1)

            if done:
                writer.add_scalar("train/episode_reward", episode_reward, global_step)
                writer.add_scalar("train/episode_steps", episode_steps, global_step)
                episode_reward = 0.0
                episode_steps = 0
                obs, _ = train_env.reset()

            if global_step >= args.start_steps:
                for _ in range(args.updates_per_step):
                    batch = buffer.sample(args.batch_size)

                    do_actor = (critic_update_count % actor_update_freq == 0)
                    critic_loss, actor_loss, alpha_loss, alpha = sac_update(
                        agent, critic_optim, actor_optim, alpha_optim,
                        batch[0], batch[1], batch[2], batch[3], batch[4],
                        args.gamma, args.tau, do_actor,
                        scaler_critic, scaler_actor, scaler_alpha, device,
                    )
                    critic_update_count += 1

                    if critic_update_count % 1000 == 0:
                        writer.add_scalar("loss/critic", critic_loss, critic_update_count)
                        if do_actor:
                            writer.add_scalar("loss/actor", actor_loss, critic_update_count)
                            writer.add_scalar("loss/alpha", alpha_loss, critic_update_count)
                            writer.add_scalar("metrics/alpha", alpha, critic_update_count)

            if global_step % args.eval_freq == 0:
                raw_mean, raw_std = evaluate(agent, raw_eval_env, device, args.eval_episodes)
                writer.add_scalar("reward/raw_mean", raw_mean, global_step)
                writer.add_scalar("reward/raw_std", raw_std, global_step)
                print(f"\nStep {global_step}: Eval (raw env) mean={raw_mean:.2f}, std={raw_std:.2f}")

                if raw_mean > best_mean_reward:
                    best_mean_reward = raw_mean
                    torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
                    save_normalize_params(train_env, os.path.join(checkpoint_dir, "normalize.pkl"))
                    print(f"  New best model saved with raw reward: {raw_mean:.2f}")

                torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "last.pt"))
                save_normalize_params(train_env, os.path.join(checkpoint_dir, "normalize_last.pkl"))

            if global_step % 250000 == 0:
                log_video(test_video_env, agent, device, os.path.join(videos_dir, f"step_{global_step}.mp4"))

    finally:
        train_env.close()
        raw_eval_env.close()
        test_video_env.close()
        writer.close()
        tee.close()
        sys.stdout = tee.stdout
