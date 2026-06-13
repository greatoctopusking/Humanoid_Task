import datetime
import os
import sys

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tensorboardX import SummaryWriter
from tqdm import tqdm

from lib.agent import SACAgent
from lib.buffer import ReplayBuffer
from lib.utils import parse_args, set_seed, make_env, make_eval_env, save_normalize_params, log_video, extract_obs_rms


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


def sac_update(agent, critic_optim, actor_optim,
               batch_obs, batch_actions, batch_rewards, batch_next_obs, batch_dones,
               gamma, update_actor,
               scaler_critic, scaler_actor, device):
    alpha = agent.alpha

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

    if update_actor:
        with torch.amp.autocast(str(device)):
            new_actions, new_log_probs = agent.get_action(batch_obs)
            q1_new, q2_new = agent.get_q_values(batch_obs, new_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (alpha * new_log_probs.unsqueeze(-1) - q_new).mean()

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

    return critic_loss.item(), actor_loss_val


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

    envs = gym.vector.AsyncVectorEnv(
        [lambda i=i: make_env(args.env, normalize=True) for i in range(args.n_envs)]
    )
    norm_eval_env = make_env(args.env, normalize=True)
    test_video_env = gym.make(args.env, render_mode="rgb_array")
    test_video_env.metadata["render_fps"] = 30
    test_video_env.reset(seed=args.seed)

    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape
    action_low = float(envs.single_action_space.low[0])
    action_high = float(envs.single_action_space.high[0])

    agent = SACAgent(obs_dim[0], act_dim[0], action_low, action_high).to(device)

    actor_params = list(agent.actor_fc.parameters()) + \
                   [agent.actor_mu.weight, agent.actor_mu.bias,
                    agent.actor_log_std.weight, agent.actor_log_std.bias]
    critic_params = list(agent.critic_1.parameters()) + list(agent.critic_2.parameters())

    actor_optim = optim.Adam(actor_params, lr=args.learning_rate, eps=1e-5)
    critic_optim = optim.Adam(critic_params, lr=args.learning_rate, eps=1e-5)

    scaler_critic = torch.amp.GradScaler(str(device))
    scaler_actor = torch.amp.GradScaler(str(device))

    buffer = ReplayBuffer(obs_dim, act_dim, args.buffer_size, device)

    print(agent)

    global_step = 0
    best_mean_reward = -np.inf
    episode_rewards = np.zeros(args.n_envs)
    critic_update_count = 0
    actor_update_freq = 2

    obs, _ = envs.reset(seed=args.seed)

    try:
        pbar = tqdm(total=args.total_steps, desc="Training")
        while global_step < args.total_steps:
            if global_step < args.start_steps:
                actions = envs.action_space.sample()
            else:
                with torch.no_grad():
                    actions, _ = agent.get_action(
                        torch.tensor(np.array(obs, dtype=np.float32), device=device)
                    )
                    actions = actions.cpu().numpy()

            next_obs, rewards, terminations, truncations, _ = envs.step(actions)

            for i in range(args.n_envs):
                real_done = float(terminations[i])
                buffer.store(obs[i], actions[i], rewards[i], next_obs[i], real_done)
                episode_rewards[i] += rewards[i]

                if terminations[i] or truncations[i]:
                    writer.add_scalar("train/episode_reward", episode_rewards[i], global_step)
                    episode_rewards[i] = 0.0

            obs = next_obs
            global_step += args.n_envs
            pbar.update(args.n_envs)

            if global_step >= args.start_steps:
                for _ in range(args.updates_per_step):
                    batch = buffer.sample(args.batch_size)

                    do_actor = (critic_update_count % actor_update_freq == 0)
                    critic_loss, actor_loss = sac_update(
                        agent, critic_optim, actor_optim,
                        batch[0], batch[1], batch[2], batch[3], batch[4],
                        args.gamma, do_actor,
                        scaler_critic, scaler_actor, device,
                    )
                    critic_update_count += 1

                    if critic_update_count % 5000 == 0:
                        print(f"  [update {critic_update_count}] critic={critic_loss:.4f} actor={actor_loss:.4f}")

            if global_step >= args.start_steps:
                agent.soft_update(args.tau)

            if global_step % args.eval_freq < args.n_envs or global_step >= args.total_steps:
                if global_step >= args.start_steps:
                    train_rms = extract_obs_rms(envs.envs[0])
                    eval_rms = extract_obs_rms(norm_eval_env)
                    if train_rms is not None and eval_rms is not None:
                        eval_rms.mean = train_rms.mean.copy()
                        eval_rms.var = train_rms.var.copy()
                        eval_rms.count = train_rms.count

                raw_mean, raw_std = evaluate(agent, norm_eval_env, device, args.eval_episodes)
                writer.add_scalar("reward/raw_mean", raw_mean, global_step)
                writer.add_scalar("reward/raw_std", raw_std, global_step)
                print(f"\nStep {global_step}: Eval (raw env) mean={raw_mean:.2f}, std={raw_std:.2f}")

                if raw_mean > best_mean_reward:
                    best_mean_reward = raw_mean
                    torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
                    save_normalize_params(envs, os.path.join(checkpoint_dir, "normalize.pkl"))
                    print(f"  New best model saved with raw reward: {raw_mean:.2f}")

                torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "last.pt"))
                save_normalize_params(envs, os.path.join(checkpoint_dir, "normalize_last.pkl"))

            if global_step % 250000 < args.n_envs and global_step >= 250000:
                log_video(test_video_env, agent, device, os.path.join(videos_dir, f"step_{global_step}.mp4"))

    finally:
        envs.close()
        norm_eval_env.close()
        test_video_env.close()
        writer.close()
        tee.close()
        sys.stdout = tee.stdout
