# SAC（Soft Actor-Critic）替换 PPO 实施计划

## 一、背景与动机

PPO 在 5M 步限制下只有 76 次策略更新机会，远未收敛（参考项目需要 1000 轮 / 65M 步）。
SAC 作为 off-policy 算法，每步都可从 Replay Buffer 采样进行梯度更新，5M 步内可达百万次更新，样本效率远超 PPO。

---

## 二、文件改动清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `lib/agent.py` | **重写** | SAC Agent：Actor + 双 Critic + Target 网络 + 自动熵 α |
| `lib/buffer.py` | **重写** | Replay Buffer：环形缓冲区，随机采样 |
| `lib/utils.py` | **修改** | 参数解析适配 SAC，环境构建函数保留 |
| `train.py` | **重写** | 新的训练循环：边收集边更新 |
| `eval.py` | **微调** | 受影响的仅 Agent 类和 get_action 调用方式 |
| `requirements.txt` | 不变 | 依赖包不变 |
| `项目方案.md` | 不变 | 保留原方案供参考 |

---

## 三、各模块详细设计

### 3.1 `lib/agent.py` — SACAgent

```
SACAgent(nn.Module)
├── actor: MLP [256, 256, 256] → mean + log_std
│   └── 初始化：正交初始化 gain=0.01 (输出层) / gain=√2
├── critic_1: MLP [256, 256, 256] → 1
├── critic_2: MLP [256, 256, 256] → 1
├── critic_1_target: 同结构，软更新
├── critic_2_target: 同结构，软更新
└── log_alpha: nn.Parameter(torch.tensor(0.0))
```

**关键方法：**

```python
# 动作采样（训练用，带 rsample 重参数化）
def get_action(self, obs, deterministic=False):
    mu, std = actor(obs)
    dist = Normal(mu, std)
    if deterministic:
        raw = mu
    else:
        raw = dist.rsample()          # 重参数化，保留梯度
    y = torch.tanh(raw)                # Tanh 压缩到 (-1, 1)
    action = y * (action_high - action_low) / 2 + (action_high + action_low) / 2
    # log_prob 含 Tanh 修正项：
    log_prob = dist.log_prob(raw) - torch.log(1 - y.pow(2) + 1e-6)
    log_prob = log_prob.sum(-1)
    return action, log_prob

# 获取 Q 值
def get_q_values(self, obs, action):
    # 将 action 映射回 (-1, 1) 后拼接
    cat = torch.cat([obs, normed_action], dim=-1)
    q1 = critic_1(cat)
    q2 = critic_2(cat)
    return q1, q2

# 软更新 Target 网络
def soft_update(self, tau=0.005):
    for target, source in zip([critic_1_target, critic_2_target],
                              [critic_1, critic_2]):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_((1 - tau) * tp.data + tau * sp.data)
```

**Target 网络初始化：** 训练开始时直接将 Q 网络的参数复制给 Target 网络（不做随机初始化的软更新）。

---

### 3.2 `lib/buffer.py` — ReplayBuffer

```
ReplayBuffer
├── capacity: 1,000,000（环形缓冲区，FIFO）
├── obs_buf, act_buf, rew_buf, next_obs_buf, done_buf
├── store(obs, action, reward, next_obs, done)
├── sample(batch_size) → (obs, act, rew, next_obs, done)
└── size → 当前存储量
```

**与 PPO Buffer 的本质区别：**
- PPO Buffer：存整条轨迹（n_steps × n_envs），按步序排列，用完就清空
- SAC Buffer：环形随机存储，容量 1M，新数据覆盖最旧数据，随机采样打破时间相关性

---

### 3.3 `lib/utils.py` — 修改 `parse_args()`

**删除的 PPO 专用参数：**
- `--n-steps`、`--n-epochs`、`--train-iters`、`--clip-ratio`、`--target-kl`、`--ent-coef`

**新增的 SAC 专用参数：**

| 参数 | 默认值 | 说明 |
|------|------|------|
| `--total-steps` | 5_000_000 | 总训练步数 |
| `--buffer-size` | 1_000_000 | Replay Buffer 容量 |
| `--batch-size` | 256 | 每次采样的 batch 大小 |
| `--gamma` | 0.99 | 折扣因子 |
| `--tau` | 0.005 | Target 网络软更新速率 |
| `--alpha-lr` | 3e-4 | 熵温度 α 学习率 |
| `--start-steps` | 10_000 | 随机探索步数（先填充 Buffer） |
| `--updates-per-step` | 1 | 每环境步的梯度更新次数 |

**保留参数：** `--env`, `--seed`, `--n-envs`, `--learning-rate`, `--render-epoch`, `--eval-freq`, `--eval-episodes`, `--cuda`

**环境构建：** `make_env` 和 `make_eval_env` 保持不变。

---

### 3.4 `train.py` — 训练循环

```
训练主循环：
1. 初始化 SACAgent、ReplayBuffer、3个 optimizer (actor/critic/alpha)
2. env.reset() → obs
3. for step in 1..total_steps:
   ├── 选择动作：
   │   ├── step < start_steps: 随机均匀动作 (探索)
   │   └── step >= start_steps: agent.get_action(obs) (SAC 策略)
   ├── env.step(action) → next_obs, reward, done
   ├── real_done = done 且非 TimeLimit.truncation
   │   （MuJoCo 有 1000 步截断，截断不算 episode 结束）
   ├── buffer.store(obs, action, reward, next_obs, real_done)
   ├── obs = next_obs
   ├── 如果 done: obs = env.reset()
   │
   ├── if step >= start_steps:
   │   └── for _ in range(updates_per_step):
   │       SAC 更新（填充 buffer 期间不更新）
   │
   ├── 每 eval_freq 步：评估（10 episodes，确定性策略）+ 保存 best model
   └── 每 render_epoch 步：录视频
```

**SAC 单步更新流程：**

```python
def sac_update(agent, actor_optim, critic_optim, alpha_optim,
               batch_obs, batch_actions, batch_rewards,
               batch_next_obs, batch_dones, gamma, device):

    # 1. Critic 更新
    with torch.no_grad():
        next_actions, next_log_probs = agent.get_action(batch_next_obs)
        q1_next, q2_next = agent.get_q_values_target(batch_next_obs, next_actions)
        q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs.unsqueeze(-1)
        q_target = batch_rewards + gamma * (1 - batch_dones) * q_next

    q1, q2 = agent.get_q_values(batch_obs, batch_actions)
    critic_loss = MSE(q1, q_target) + MSE(q2, q_target)
    critic_optim.zero_grad(); critic_loss.backward(); critic_optim.step()

    # 2. Actor 更新（降低更新频率：每 2 次 critic 更新做 1 次 actor 更新）
    if total_critic_updates % 2 == 0:
        new_actions, new_log_probs = agent.get_action(batch_obs)
        q1_new, q2_new = agent.get_q_values(batch_obs, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (alpha * new_log_probs - q_new).mean()
        actor_optim.zero_grad(); actor_loss.backward(); actor_optim.step()

        # 3. Alpha 更新
        alpha_loss = -(log_alpha * (new_log_probs + target_entropy).detach()).mean()
        alpha_optim.zero_grad(); alpha_loss.backward(); alpha_optim.step()
        alpha = log_alpha.exp()

    # 4. 软更新 Target 网络
    agent.soft_update(tau)

    return critic_loss, actor_loss, alpha_loss, alpha
```

**注意：** 使用 `torch.amp.autocast` + `GradScaler` 保持混合精度训练。

---

### 3.5 `eval.py` — 评估

修改点：
- `from lib.agent import SACAgent` 替换 `PPOAgent`
- `agent.get_action(obs, deterministic=True)` 替换 `agent.get_action_and_value(obs)`
- 参数解析移除 PPO 专用参数（`--model`, `--normalize` 等保留）

---

## 四、SAC 关键超参数

| 参数 | 值 | 说明 |
|------|------|------|
| total_steps | 5,000,000 | 总训练步数（硬限制） |
| n_envs | 1 | SAC 用单环境（off-policy 无需并行） |
| buffer_size | 1,000,000 | Replay Buffer 容量 |
| batch_size | 256 | Mini-batch 大小 |
| learning_rate | 3e-4 | Actor / Critic / Alpha 学习率 |
| gamma | 0.99 | 折扣因子 |
| tau | 0.005 | 软更新速率 |
| target_entropy | -17 | = -action_dim（自动 α 的目标） |
| start_steps | 10,000 | 随机探索阶段 |
| actor_update_freq | 2 | 每 2 次 critic 更新做 1 次 actor 更新 |
| updates_per_step | 1 | 每环境步的更新次数 |

---

## 五、实施步骤（按顺序）

| 步骤 | 文件 | 任务 | 预计时间 |
|------|------|------|----------|
| 1 | `lib/agent.py` | 重写为 SACAgent（Actor + 双 Critic + Target + log_alpha） | 30 分钟 |
| 2 | `lib/buffer.py` | 重写为 ReplayBuffer（环形缓冲区 + 随机采样） | 15 分钟 |
| 3 | `lib/utils.py` | 修改 parse_args()，替换为 SAC 参数 | 10 分钟 |
| 4 | `train.py` | 重写训练循环（边收集边更新 + 混合精度） | 40 分钟 |
| 5 | `eval.py` | 微调：SACAgent + deterministic get_action | 10 分钟 |
| 6 | — | 运行测试，调试验证 | 30 分钟 |
| 7 | — | 5M 步训练 + 结果记录 | 若干小时 |

---

## 六、与作业要求的对照检查

| 作业要求 | 状态 | SAC 对应措施 |
|------|------|------|
| Humanoid-v5 | ✅ | env_id 固定 |
| gymnasium==1.2.3, mujoco==3.8.1 | ✅ | requirements.txt 不动 |
| 随机种子固定 | ✅ | set_seed() 函数保留 |
| 禁止修改物理参数 | ✅ | 不修改环境 |
| 原生环境累计奖励评测 | ✅ | 仍用 VecNormalize + 评测用原生环境 |
| 步数 ≤ 5,000,000 | ✅ | total_steps=5,000,000 严格控制 |
| 归一化参数保存/加载 | ✅ | save/load_normalize_params 保留 |
| eval 支持 --seed | ✅ | 保留 |
| requirements.txt 含版本号 | ✅ | 不变 |

---

## 七、风险与注意事项

1. **单环境 vs 多环境**：SAC 通常用单环境，因为 off-policy 不需要同步收集。但如果单环境运行太慢，可切到 `n_envs=4`（小并行 + 更好的探索多样性）。

2. **TimeLimit.truncation 处理**：MuJoCo 的 Humanoid-v5 在 1000 步自动截断。截断时不应将 `done` 设为 True（否则 Q target 计算中会错误地终止未来奖励传播）。需要从 `info` 中提取 `TimeLimit.truncated` 标志。

3. **Actor 更新延迟**：`actor_update_freq=2` 是 SAC 标准实践，让 Critic 先收敛一步再更新 Actor，提升稳定性。

4. **Tanh log_prob 修正**：`log_prob_corrected = dist.log_prob(raw) - log(1 - tanh(raw)^2 + ε)`，必须逐元素求和后才是最终 log_prob。这个修正项是 SAC 的正确定性关键，错误实现会导致策略不收敛。

5. **混合精度**：Critic 的 Q 值可能很大（几百量级），需要确保 autocast 下 `q_target` 计算不被 float16 溢出。必要时在 critic 更新的关键计算上使用 float32。
