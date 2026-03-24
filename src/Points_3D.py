# train_pointcloud.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
import gym
from Env3D import PointCloudEnv  
from parsers import args  
from RL_brain import ReplayBuffer, DDPG

class DDPGModified(DDPG):
    def take_action(self, state):
        action = super().take_action(state)
        azimuth = np.clip(action[0], -np.pi, np.pi)
        elevation = np.clip(action[1], -np.pi/2, np.pi/2)
        return np.array([azimuth, elevation])

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
            if 'weights_only' in torch.load.__code__.co_varnames:
                weights_only = True
            else:
                weights_only = False

            actor_state_dict = torch.load(os.path.join(checkpoint_dir, 'actor.pth'),
                                         map_location=self.device,
                                         weights_only=weights_only)
            self.actor.load_state_dict(actor_state_dict)

            critic_state_dict = torch.load(os.path.join(checkpoint_dir, 'critic.pth'),
                                          map_location=self.device,
                                          weights_only=weights_only)
            self.critic.load_state_dict(critic_state_dict)

            target_actor_state_dict = torch.load(os.path.join(checkpoint_dir, 'target_actor.pth'),
                                                map_location=self.device,
                                                weights_only=weights_only)
            self.target_actor.load_state_dict(target_actor_state_dict)

            target_critic_state_dict = torch.load(os.path.join(checkpoint_dir, 'target_critic.pth'),
                                                 map_location=self.device,
                                                 weights_only=weights_only)
            self.target_critic.load_state_dict(target_critic_state_dict)

            print(f"模型从 {checkpoint_dir} 加载完成。")
        except FileNotFoundError:
            print(f"在 {checkpoint_dir} 中未找到检查点。开始新的训练。")
        except TypeError as e:
            print(f"加载模型时发生类型错误: {e}. 请确保 PyTorch 版本支持 weights_only 参数。")

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print('开始运行! -- 使用的是:', device)

# -------------------------------------- #
# 环境设置
# -------------------------------------- #

env = PointCloudEnv(
    pcd_file='Pcd_progress/scaled_output.pcd',
    voxel_size=0.1, 
    initial_agent_radius=2.0, 
    action_distance=0.5, 
    k_neighbors=7,
    curvature_threshold=10.0,
    min_clusters=2,
    max_clusters=100,
    grid_size=2.5,
    exploration_bonus=0.05,
    penalty_factor=0.1
)

n_states = env.observation_space.shape[0]  
n_actions = env.action_space.shape[0]  
action_bound = np.array([np.pi, np.pi/2], dtype=np.float32)  

# -------------------------------------- #
# 模型构建
# -------------------------------------- #

replay_buffer = ReplayBuffer(capacity=args.buffer_size)

agent = DDPGModified(
    n_states=n_states,
    n_hiddens=args.n_hiddens,
    n_actions=n_actions,
    action_bound=action_bound, 
    sigma=args.sigma,
    actor_lr=args.actor_lr,
    critic_lr=args.critic_lr,
    tau=args.tau,
    gamma=args.gamma,
    device=device
)

# -------------------------------------- #
# 如果可用则加载检查点
# -------------------------------------- #

checkpoint_dir = 'checkpoints'  
if os.path.exists(checkpoint_dir):
    agent.load(checkpoint_dir)
else:
    print("未找到模型参数，从头开始训练。")

# -------------------------------------- #
# 模型训练
# -------------------------------------- #

return_list = []        
mean_return_list = [] 

for i in range(30): 
    episode_return = 0
    state = env.reset()
    done = False

    while not done:
        action = agent.take_action(state)
        next_state, reward, done, info = env.step(action)

        replay_buffer.add(state, action, reward, next_state, done)
        state = next_state
        episode_return += reward

        if replay_buffer.size() > args.min_size:
            s, a, r, ns, d = replay_buffer.sample(args.batch_size)
            transition_dict = {
                'states': s,
                'actions': a,
                'rewards': r,
                'next_states': ns,
                'dones': d,
            }
            agent.update(transition_dict)

        # env.render()

    return_list.append(episode_return)
    mean_return = np.mean(return_list[-100:]) 
    mean_return_list.append(mean_return)

    print(f'========Episode: {i+1}, Return: {episode_return:.2f}, Mean Return: {mean_return:.2f}========')
    print('\n')

    if (i + 1) % 2 == 0:  
        agent.save(checkpoint_dir)
        print('====已保存模型参数！！====')

agent.save(checkpoint_dir)

env.close()
