# RL_brain.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import random
import torch.nn.functional as F
import os


# ------------------------------------- #
# 经验回放池
# ------------------------------------- #

class ReplayBuffer:
    def __init__(self, capacity):  
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def size(self):
        return len(self.buffer)


# ------------------------------------- #
# 策略网络
# ------------------------------------- #

class PolicyNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions, action_bound):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(n_states, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, n_actions)
        self.register_buffer('action_bound_tensor', torch.tensor(action_bound, dtype=torch.float32))

    def forward(self, x):
        x = self.fc1(x) 
        x = F.relu(x)
        x = self.fc2(x) 
        x = F.relu(x)
        x = self.fc3(x)  
        x = torch.tanh(x)  
        x = x * self.action_bound_tensor
        return x


# ------------------------------------- #
# 价值网络
# ------------------------------------- #

class QValueNet(nn.Module):
    def __init__(self, n_states, n_hiddens, n_actions):
        super(QValueNet, self).__init__()
        self.fc1 = nn.Linear(n_states + n_actions, n_hiddens)
        self.fc2 = nn.Linear(n_hiddens, n_hiddens)
        self.fc3 = nn.Linear(n_hiddens, 1)

    def forward(self, x, a):
        cat = torch.cat([x, a], dim=1)  
        x = self.fc1(cat)  
        x = F.relu(x)
        x = self.fc2(x) 
        x = F.relu(x)
        x = self.fc3(x) 
        return x


# ------------------------------------- #
# 算法主体
# ------------------------------------- #

class DDPG:
    def __init__(self, n_states, n_hiddens, n_actions, action_bound,
                 sigma, actor_lr, critic_lr, tau, gamma, device):

        self.actor = PolicyNet(n_states, n_hiddens, n_actions, action_bound).to(device)
        self.critic = QValueNet(n_states, n_hiddens, n_actions).to(device)
        self.target_actor = PolicyNet(n_states, n_hiddens, n_actions, action_bound).to(device)
        self.target_critic = QValueNet(n_states, n_hiddens, n_actions).to(device)

        self.target_critic.load_state_dict(self.critic.state_dict())
        self.target_actor.load_state_dict(self.actor.state_dict())

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.gamma = gamma  
        self.sigma = sigma 
        self.tau = tau  
        self.n_actions = n_actions
        self.device = device

    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).view(1, -1).to(self.device)
        with torch.no_grad():
            action = self.actor(state).cpu().numpy().flatten()
        action = action + self.sigma * np.random.randn(self.n_actions)
        action = np.clip(action, -self.actor.action_bound_tensor.cpu().numpy(),
                         self.actor.action_bound_tensor.cpu().numpy())
        return action

    def soft_update(self, net, target_net):
        for param_target, param in zip(target_net.parameters(), net.parameters()):
            param_target.data.copy_(param_target.data * (1 - self.tau) + param.data * self.tau)

    def update(self, transition_dict):
            states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)  # [b,n_states]
            actions = torch.tensor(transition_dict['actions'], dtype=torch.float32).to(self.device)  # [b,n_actions]
            rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1, 1).to(self.device)  # [b,1]
            next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float32).to(self.device)  # [b,n_states]
            dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1, 1).to(self.device)  # [b,1]

            next_actions = self.target_actor(next_states)
            next_q_values = self.target_critic(next_states, next_actions)
            q_targets = rewards + self.gamma * next_q_values * (1 - dones)

            q_values = self.critic(states, actions)

            td_errors_tensor = (q_targets - q_values).detach().view(-1)

            critic_loss = F.mse_loss(q_values, q_targets)
            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            self.critic_optimizer.step()

            actor_actions = self.actor(states)
            actor_q_values = self.critic(states, actor_actions)
            actor_loss = -torch.mean(actor_q_values)
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            self.soft_update(self.actor, self.target_actor)
            self.soft_update(self.critic, self.target_critic)

            td_err_cpu = td_errors_tensor.cpu()
            td_mean = float(td_err_cpu.mean().item()) if td_err_cpu.numel() > 0 else 0.0
            td_std = float(td_err_cpu.std(unbiased=False).item()) if td_err_cpu.numel() > 0 else 0.0
            max_samples = 32
            if td_err_cpu.numel() > max_samples:
                idx = torch.randperm(td_err_cpu.numel())[:max_samples]
                td_samples = td_err_cpu[idx].tolist()
            else:
                td_samples = td_err_cpu.tolist()

            return {
                'critic_loss': float(critic_loss.item()),
                'actor_loss': float(actor_loss.item()),
                'td_error_mean': td_mean,
                'td_error_std': td_std,
                'td_error_samples': td_samples,
            }

    def save(self, checkpoint_dir='checkpoints'):
        """保存actor和critic网络参数。"""
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        torch.save(self.actor.state_dict(), os.path.join(checkpoint_dir, 'actor.pth'))
        torch.save(self.critic.state_dict(), os.path.join(checkpoint_dir, 'critic.pth'))
        torch.save(self.target_actor.state_dict(), os.path.join(checkpoint_dir, 'target_actor.pth'))
        torch.save(self.target_critic.state_dict(), os.path.join(checkpoint_dir, 'target_critic.pth'))
        print(f"模型参数保存在： {checkpoint_dir}.")

    def load(self, checkpoint_dir='checkpoints'):
        """加载actor和critic网络参数。"""
        try:
            self.actor.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'actor.pth'), map_location=self.device))
            self.critic.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, 'critic.pth'), map_location=self.device))
            self.target_actor.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, 'target_actor.pth'), map_location=self.device))
            self.target_critic.load_state_dict(
                torch.load(os.path.join(checkpoint_dir, 'target_critic.pth'), map_location=self.device))
            print(f"模型从 {checkpoint_dir} 加载完成。")
        except FileNotFoundError:
            print(f"在 {checkpoint_dir} 中未找到检查点。开始新的训练。")