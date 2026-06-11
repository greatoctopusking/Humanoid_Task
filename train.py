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

from lib.agent import PPOAgent
from lib.buffer import PPOBuffer
from lib.utils import parse_args, set_seed, make_env, make_eval_env, save_normalize_params, log_video


class Tee:
    def __init__(self, file_path):
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)
        self.file.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def ppo_update(agent, optimizer, scaler, batch_obs, batch_actions, batch_returns,
               batch_old_log_probs, batch_adv, clip_epsilon, vf_coef, ent_coef,
               max_grad_norm, device):
    agent.train()
    optimizer.zero_grad()

    with torch.amp.autocast(str(device)):
        _, new_log_probs, entropies, new_values = agent.get_action_and_value(batch_obs, batch_actions)
        ratio = torch.exp(new_log_probs - batch_old_log_probs)
        kl = ((batch_old_log_probs - new_log_probs) / batch_actions.size(-1)).mean()

        surr1 = ratio * batch_adv
        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_adv
        policy_loss = -torch.min(surr1, surr2).mean()
        value_loss = nn.MSELoss()(new_values.squeeze(1), batch_returns)
        entropy = entropies.mean()
        loss = policy_loss + vf_coef * value_loss - ent_coef * entropy

    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
    scaler.step(optimizer)
    scaler.update()

    return loss.item(), policy_loss.item(), value_loss.item(), entropy.item(), kl.item()


def evaluate(agent, env, device, num_episodes=10):
    agent.eval()
    episode_rewards = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            with torch.no_grad():
                action, _, _, _ = agent.get_action_and_value(
                    torch.tensor(np.array([obs], dtype=np.float32), device=device)
                )
            obs, reward, terminated, truncated, _ = env.step(action.squeeze(0).cpu().numpy())
            total_reward += reward
            done = terminated or truncated
        episode_rewards.append(total_reward)
    return float(np.mean(episode_rewards)), float(np.std(episode_rewards))


if __name__ == "__main__":
    args = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(__file__)
    folder_name = f"{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    videos_dir = os.path.join(current_dir, "videos", folder_name)
    os.makedirs(videos_dir, exist_ok=True)
    checkpoint_dir = os.path.join(current_dir, "checkpoints", folder_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    log_dir = os.path.join(current_dir, "logs", folder_name)
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
    raw_eval_env = make_eval_env(args.env, seed=args.seed)
    test_video_env = gym.make(args.env, render_mode="rgb_array")
    test_video_env.metadata["render_fps"] = 30
    test_video_env.reset(seed=args.seed)

    obs_dim = envs.single_observation_space.shape
    act_dim = envs.single_action_space.shape
    action_low = float(envs.single_action_space.low[0])
    action_high = float(envs.single_action_space.high[0])

    agent = PPOAgent(obs_dim[0], act_dim[0], action_low, action_high).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    def lr_lambda(epoch):
        warmup_epochs = 10
        if epoch < warmup_epochs:
            return float(epoch) / float(max(1, warmup_epochs))
        else:
            T_cur = epoch - warmup_epochs
            T_total = max(1, args.n_epochs - warmup_epochs)
            return 0.5 * (1 + np.cos(np.pi * T_cur / T_total))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler(str(device))

    print(agent.actor_mu)
    print(agent.actor_logstd)
    print(agent.critic)

    buffer = PPOBuffer(obs_dim, act_dim, args.n_steps, args.n_envs, device, args.gamma, args.gae_lambda)

    global_step_idx = 0
    best_mean_reward = -np.inf
    start_time = time.time()
    next_obs = torch.tensor(np.array(envs.reset(seed=args.seed)[0], dtype=np.float32), device=device)
    next_terminateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    next_truncateds = torch.zeros(args.n_envs, dtype=torch.float32, device=device)
    reward_list = []

    try:
        for epoch in range(1, args.n_epochs + 1):
            for _ in tqdm(range(0, args.n_steps), desc=f"Epoch {epoch}: Collecting"):
                global_step_idx += args.n_envs
                obs = next_obs
                terminateds = next_terminateds
                truncateds = next_truncateds

                with torch.no_grad():
                    actions, logprobs, _, values = agent.get_action_and_value(obs)
                    values = values.reshape(-1)

                next_obs, rewards, next_terminateds, next_truncateds, _ = envs.step(actions.cpu().numpy())
                next_obs = torch.tensor(np.array(next_obs, dtype=np.float32), device=device)
                reward_list.extend(rewards)
                rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
                next_terminateds = torch.as_tensor(next_terminateds, dtype=torch.float32, device=device)
                next_truncateds = torch.as_tensor(next_truncateds, dtype=torch.float32, device=device)

                buffer.store(obs, actions, rewards, values, terminateds, truncateds, logprobs)

            with torch.no_grad():
                next_values = agent.get_value(next_obs).reshape(1, -1)
                next_terminateds = next_terminateds.reshape(1, -1)
                next_truncateds = next_truncateds.reshape(1, -1)
                traj_adv, traj_ret = buffer.calculate_advantages(next_values, next_terminateds, next_truncateds)

            traj_obs, traj_act, traj_logprob = buffer.get()

            traj_obs = traj_obs.view(-1, *obs_dim)
            traj_act = traj_act.view(-1, *act_dim)
            traj_logprob = traj_logprob.view(-1)
            traj_adv = traj_adv.view(-1)
            traj_ret = traj_ret.view(-1)

            traj_adv = (traj_adv - traj_adv.mean()) / (traj_adv.std() + 1e-8)

            dataset_size = args.n_steps * args.n_envs
            traj_indices = np.arange(dataset_size)

            losses_policy = []
            losses_value = []
            entropies = []
            losses_total = []
            kl_list = []
            kl_early_stop = False

            for _ in tqdm(range(args.train_iters), desc=f"Epoch {epoch}: Training"):
                np.random.shuffle(traj_indices)
                for start_idx in range(0, dataset_size, args.batch_size):
                    end_idx = start_idx + args.batch_size
                    batch_indices = traj_indices[start_idx:end_idx]

                    batch_obs = traj_obs[batch_indices]
                    batch_actions = traj_act[batch_indices]
                    batch_returns = traj_ret[batch_indices]
                    batch_old_log_probs = traj_logprob[batch_indices]
                    batch_adv = traj_adv[batch_indices]

                    loss, policy_loss, value_loss, entropy, kl = ppo_update(
                        agent, optimizer, scaler,
                        batch_obs, batch_actions, batch_returns,
                        batch_old_log_probs, batch_adv,
                        args.clip_ratio, args.vf_coef, args.ent_coef,
                        args.max_grad_norm, device
                    )

                    losses_policy.append(policy_loss)
                    losses_value.append(value_loss)
                    entropies.append(entropy)
                    losses_total.append(loss)
                    kl_list.append(kl)

                    if kl > args.target_kl:
                        kl_early_stop = True
                        break

                if kl_early_stop:
                    break

            total_loss = np.mean(losses_total)
            policy_loss = np.mean(losses_policy)
            value_loss = np.mean(losses_value)
            entropy = np.mean(entropies)
            kl = np.mean(kl_list)

            writer.add_scalar("loss/total", total_loss, epoch)
            writer.add_scalar("loss/policy", policy_loss, epoch)
            writer.add_scalar("loss/value", value_loss, epoch)
            writer.add_scalar("loss/entropy", entropy, epoch)
            writer.add_scalar("metrics/kl", kl, epoch)
            writer.add_scalar("metrics/learning_rate", scheduler.get_last_lr()[0], epoch)

            mean_reward = float(np.mean(reward_list))
            writer.add_scalar("reward/normalized_mean", mean_reward, epoch)
            reward_list = []

            print(f"Epoch {epoch} done in {time.time() - start_time:.2f}s, "
                  f"norm mean reward: {mean_reward:.4f}, "
                  f"total loss: {total_loss:.4f}, policy loss: {policy_loss:.4f}, "
                  f"value loss: {value_loss:.4f}, entropy: {entropy:.4f}, kl: {kl:.4f}, "
                  f"learning rate: {scheduler.get_last_lr()[0]:.2e}")
            start_time = time.time()

            if global_step_idx % args.eval_freq < args.n_envs * args.n_steps or epoch == args.n_epochs:
                raw_mean, raw_std = evaluate(agent, raw_eval_env, device, args.eval_episodes)
                writer.add_scalar("reward/raw_mean", raw_mean, epoch)
                writer.add_scalar("reward/raw_std", raw_std, epoch)
                print(f"  Eval (raw env): mean={raw_mean:.2f}, std={raw_std:.2f}")

                if raw_mean > best_mean_reward:
                    best_mean_reward = raw_mean
                    torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "best.pt"))
                    save_normalize_params(envs, os.path.join(checkpoint_dir, "normalize.pkl"))
                    print(f"  New best model saved with raw reward: {raw_mean:.2f}")

            torch.save(agent.state_dict(), os.path.join(checkpoint_dir, "last.pt"))
            save_normalize_params(envs, os.path.join(checkpoint_dir, "normalize_last.pkl"))

            if epoch % args.render_epoch == 0:
                log_video(test_video_env, agent, device, os.path.join(videos_dir, f"epoch_{epoch}.mp4"))

            scheduler.step()

    finally:
        envs.close()
        raw_eval_env.close()
        test_video_env.close()
        writer.close()
        tee.close()
        sys.stdout = tee.stdout
