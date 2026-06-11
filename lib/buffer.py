import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, capacity, device):
        self.capacity = capacity
        self.device = device

        self.obs_buf = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.act_buf = np.zeros((capacity, *act_dim), dtype=np.float32)
        self.rew_buf = np.zeros((capacity, 1), dtype=np.float32)
        self.next_obs_buf = np.zeros((capacity, *obs_dim), dtype=np.float32)
        self.done_buf = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0
        self.size = 0

    def store(self, obs, action, reward, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = action
        self.rew_buf[self.ptr] = reward
        self.next_obs_buf[self.ptr] = next_obs
        self.done_buf[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.tensor(self.obs_buf[indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.act_buf[indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.rew_buf[indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_obs_buf[indices], dtype=torch.float32, device=self.device),
            torch.tensor(self.done_buf[indices], dtype=torch.float32, device=self.device),
        )

    def __len__(self):
        return self.size
