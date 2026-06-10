import torch
import torch.nn as nn
from torch.distributions import Normal


class PPOAgent(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int, action_low: float = -0.4, action_high: float = 0.4):
        super().__init__()

        self.action_low = action_low
        self.action_high = action_high

        hidden = [256, 256, 256]

        actor_layers = []
        prev = num_inputs
        for h in hidden:
            actor_layers.append(nn.Linear(prev, h))
            actor_layers.append(nn.Tanh())
            prev = h
        actor_layers.append(nn.Linear(prev, num_actions))
        actor_layers.append(nn.Tanh())
        self.actor_mu = nn.Sequential(*actor_layers)

        self.actor_logstd = nn.Parameter(torch.ones(1, num_actions) * -0.5)

        critic_layers = []
        prev = num_inputs
        for h in hidden:
            critic_layers.append(nn.Linear(prev, h))
            critic_layers.append(nn.Tanh())
            prev = h
        critic_layers.append(nn.Linear(prev, 1))
        self.critic = nn.Sequential(*critic_layers)

        self._init_weights()

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if "actor_mu" in name:
                    if name.endswith(str(len(self.actor_mu) - 1)):
                        nn.init.orthogonal_(module.weight, gain=0.01)
                    else:
                        nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
                    nn.init.constant_(module.bias, 0)
                elif "critic" in name:
                    if name.endswith(str(len(self.critic) - 1)):
                        nn.init.orthogonal_(module.weight, gain=1.0)
                    else:
                        nn.init.orthogonal_(module.weight, gain=2 ** 0.5)
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        mu = self.actor_mu(x)
        std = torch.exp(self.actor_logstd).expand_as(mu)
        return mu, std

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        mu, std = self.forward(x)
        dist = Normal(mu, std)
        if action is None:
            action = dist.rsample()
        else:
            action = (action - self.action_low) / (self.action_high - self.action_low) * 2 - 1
        log_prob = dist.log_prob(action).sum(-1)
        entropy = dist.entropy().mean(-1)
        raw_action = action * (self.action_high - self.action_low) / 2 + (self.action_high + self.action_low) / 2
        return raw_action, log_prob, entropy, self.get_value(x)
