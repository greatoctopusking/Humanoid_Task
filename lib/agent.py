import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class SACAgent(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int, action_low: float = -0.4, action_high: float = 0.4):
        super().__init__()

        self.action_scale = (action_high - action_low) / 2.0
        self.action_bias = (action_high + action_low) / 2.0

        hidden = [256, 256, 256]

        actor_layers = []
        prev = num_inputs
        for h in hidden:
            actor_layers.append(nn.Linear(prev, h))
            actor_layers.append(nn.ReLU())
            prev = h
        self.actor_fc = nn.Sequential(*actor_layers)
        self.actor_mu = nn.Linear(prev, num_actions)
        self.actor_log_std = nn.Linear(prev, num_actions)

        def build_critic():
            layers = []
            prev = num_inputs + num_actions
            for h in hidden:
                layers.append(nn.Linear(prev, h))
                layers.append(nn.ReLU())
                prev = h
            layers.append(nn.Linear(prev, 1))
            return nn.Sequential(*layers)

        self.critic_1 = build_critic()
        self.critic_2 = build_critic()
        self.critic_1_target = build_critic()
        self.critic_2_target = build_critic()
        self.critic_1_target.load_state_dict(self.critic_1.state_dict())
        self.critic_2_target.load_state_dict(self.critic_2.state_dict())

        self.target_entropy = -num_actions
        self.log_alpha = nn.Parameter(torch.tensor(0.0))

        self._init_weights()

    def _init_weights(self):
        for module in self.actor_fc.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
                nn.init.constant_(module.bias, 0)
        nn.init.orthogonal_(self.actor_mu.weight, gain=0.01)
        nn.init.constant_(self.actor_mu.bias, 0)
        nn.init.orthogonal_(self.actor_log_std.weight, gain=0.01)
        nn.init.constant_(self.actor_log_std.bias, 0)

        for critic in [self.critic_1, self.critic_2, self.critic_1_target, self.critic_2_target]:
            modules = list(critic.modules())
            for module in modules[:-1]:
                if isinstance(module, nn.Linear):
                    nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
                    nn.init.constant_(module.bias, 0)
            last = modules[-1]
            nn.init.orthogonal_(last.weight, gain=1.0)
            nn.init.constant_(last.bias, 0)

    def forward(self, obs):
        x = self.actor_fc(obs)
        mu = self.actor_mu(x)
        log_std = self.actor_log_std(x)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def get_action(self, obs, deterministic=False):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)

        if deterministic:
            raw = mu
        else:
            raw = dist.rsample()

        y = torch.tanh(raw)
        action = y * self.action_scale + self.action_bias

        log_prob = dist.log_prob(raw)
        log_prob -= torch.log(1 - y.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1)

        return action, log_prob

    def get_q_values(self, obs, action):
        action_norm = (action - self.action_bias) / self.action_scale
        cat = torch.cat([obs, action_norm], dim=-1)
        q1 = self.critic_1(cat)
        q2 = self.critic_2(cat)
        return q1, q2

    def get_q_values_target(self, obs, action):
        action_norm = (action - self.action_bias) / self.action_scale
        cat = torch.cat([obs, action_norm], dim=-1)
        q1 = self.critic_1_target(cat)
        q2 = self.critic_2_target(cat)
        return q1, q2

    def soft_update(self, tau=0.005):
        for target, source in [(self.critic_1_target, self.critic_1), (self.critic_2_target, self.critic_2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(tau * sp.data + (1.0 - tau) * tp.data)

    @property
    def alpha(self):
        return self.log_alpha.exp()
