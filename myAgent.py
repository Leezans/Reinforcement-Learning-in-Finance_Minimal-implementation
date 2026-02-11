import torch
import torch.nn as nn
from torch.distributions import Categorical
import os
from myNN import LSTMResNet,ResidualBlock
import config

param = config.Param()


if(torch.cuda.is_available()): 
    device = torch.device('cuda')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    device = torch.device('cpu')
    print("Device set to : cpu")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class PolicyNetwork(nn.Module):
    def __init__(self, input_dim,action_dim):
        super(PolicyNetwork, self).__init__()
        self.lstmResNet = LSTMResNet(input_dim = input_dim, output_dim = action_dim)
    
    def forward(self, x):
        x = self.lstmResNet(x)
        logits = torch.nn.functional.softmax(x,dim=-1)
        return logits


class ValueNetwork(nn.Module):
    def __init__(self, input_dim):
        super(ValueNetwork, self).__init__()
        self.lstmResNet = LSTMResNet(input_dim = input_dim,
                                     output_dim=param.hiddenDim)
        self.fcOut = nn.Linear(param.hiddenDim, 1)
    
    def forward(self, x):
        x = self.lstmResNet(x)
        value = self.fcOut(x)
        return value

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim,test=False):
        super(ActorCritic, self).__init__()

        self.actor = PolicyNetwork(state_dim, action_dim)
        self.critic = ValueNetwork(state_dim)
        if test:
            self.actor = nn.Sequential(
                                nn.Linear(state_dim, 64),
                                nn.Tanh(),
                                nn.Linear(64, 64),
                                nn.Tanh(),
                                nn.Linear(64, action_dim),
                                nn.Softmax(dim=-1)
                            )
            self.critic = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, 1)
                        )

    
    def act(self, state):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        state_val = self.critic(state)
        return action.detach(), action_logprob.detach(), state_val.detach()
    
    def evaluate(self, state, action):
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action) 
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor=0.0001, lr_critic=0.0001, gamma=0.99, K_epochs=80, eps_clip=0.2,test=False):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(state_dim, action_dim, test=test).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim,test=test).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()


    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            if state.dim() == 2:
                state = state[None, :, :]
            
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)
        return action.item()

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)
        advantages = rewards.detach() - old_state_values.detach()

        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            state_values = torch.squeeze(state_values)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.buffer.clear()
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   
    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        
        
if __name__ == "__main__":
    
    import gymnasium as gym
    import numpy as np
    import matplotlib.pyplot as plt
    import datetime
    import os

    envName = "CartPole-v1"
    env = gym.make(envName)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    ppo_agent = PPO(state_dim, action_dim, test=True)
    evaluate = True
    if evaluate:

        ppo_agent.load(f"testModels/ppo_{envName}_final.pth")

        env = gym.make(envName, render_mode="human")

        test_episodes = 100
        test_rewards = []
        
        for episode in range(test_episodes):
            state, _ = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = ppo_agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                state = next_state
            
            test_rewards.append(total_reward)
        
        avg_test_reward = np.mean(test_rewards)
        print(f"Average Test Reward over {test_episodes} episodes: {avg_test_reward:.2f}")

    else:

        max_episodes = 1000    
        save_interval = 100       
        print_interval = 20    

        episode_rewards = []                  
        timestep = 0                           
        start_time = datetime.datetime.now().replace(microsecond=0)

        for episode in range(1, max_episodes+1):
            state, _ = env.reset()
            done = False
            total_reward = 0

            while not done:
                action = ppo_agent.select_action(state)
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                total_reward += reward
                state = next_state
                timestep += 1

            if episode % print_interval == 0:
                ppo_agent.update()

            episode_rewards.append(total_reward)

            if episode % print_interval == 0:
                avg_reward = np.mean(episode_rewards[-print_interval:])
                time_elapsed = datetime.datetime.now().replace(microsecond=0) - start_time
                print(f"Episode: {episode:5d} | Timestep: {timestep:6d} | "
                    f"Avg Reward: {avg_reward:6.2f} | Time: {time_elapsed}")

            if episode % save_interval == 0:
                if not os.path.exists('testModels'):
                    os.makedirs('testModels')
                checkpoint_path = f"testModels/ppo_{envName}_{episode}.pth"
                ppo_agent.save(checkpoint_path)

        ppo_agent.save(f"testModels/ppo_{envName}_final.pth")

        plt.figure(figsize=(12, 6))
        plt.plot(episode_rewards, alpha=0.6)
        plt.plot(np.convolve(episode_rewards, np.ones(100)/100, mode='valid'), 
                color='red', linewidth=2)  
        plt.xlabel("Episode")
        plt.ylabel("Total Reward")
        plt.title(f"Training Progress ({envName})")
        plt.savefig(f"training_curve_{envName}.png")
        plt.show()

        
        
    pass


