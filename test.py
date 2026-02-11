import os
import glob
import time
from datetime import datetime
import pandas as pd
import torch
import numpy as np
import gymnasium as gym
from myAgent import PPO
from myStockEnv import SingleStockDayEnv


from config import Param
param = Param()


def test():
    print("============================================================================================")



if __name__ == '__main__':
    train =False
    fixed = True
    stockCode = param.code
    seq_length = param.seq_length
    env_name = param.envName
    log_dir = os.path.join("log", env_name)
    if not os.path.exists(log_dir):
          os.makedirs(log_dir)
    
    env = SingleStockDayEnv(param)
    df = pd.read_csv(env.csv_path)

    df['reward'] = 0.0
    df['step'] = 0
    df['action'] = 0
    df['asset'] = 0.0
    df['balance'] = 0.0
    df['stock_owned'] = 0
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    K_epochs = param.K_epochs             # update policy for K epochs in one PPO update
    eps_clip = param.eps_clip          # clip parameter for PPO
    gamma = param.gamma            # discount factor
    lr_actor = param.lr_actor       # learning rate for actor network
    lr_critic = param.lr_critic       # learning rate for critic network
    ppo_agent = PPO(state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip)
    episode = param.episodeNum - 1 
    loadPath = os.path.join("model", "PPO_{}_{}.pth".format(env_name, episode))
    ppo_agent.load(loadPath)
    
    params_file_path = os.path.join(log_dir, f"params.txt")
    with open(params_file_path, "w") as f:
        f.write(f"stockCode: {stockCode}\n")
        f.write(f"env_name: {env_name}\n")
        f.write(f"seq_length: {seq_length}\n")
        f.write(f"state_dim: {state_dim}\n")
        f.write(f"action_dim: {action_dim}\n")
        f.write(f"update policy for K epochs in one PPO update.K_epochs: {K_epochs}\n")
        f.write(f"eps_clip: {eps_clip}\n")
        f.write(f"gamma: {gamma}\n")
        f.write(f"learning rate for actor network: {lr_actor}\n")
        f.write(f"learning rate for critic network: {lr_critic}\n")
        
    for episode in range(1):
        state,info = env.reset()
        modelSavePath = os.path.join("model", "PPO_Eval_{}_{}.pth".format(env_name, episode))
        for step in range(6000):
            action = ppo_agent.select_action(state)
            state, reward, done, truncated, info = env.step(action)
            print("episode:",episode,"step:",step)
            print("action",action,"\nreward:",reward,"\ninfo:",info)
            loc = step + seq_length -1
            df.loc[loc, 'reward'] = reward
            df.loc[loc, 'step'] = step
            df.loc[loc, 'action'] = action
            df.loc[loc, 'asset'] = info['asset']
            df.loc[loc, 'balance'] = info['balance']
            df.loc[loc, 'stock_owned'] = info['stock_owned']
            ppo_agent.buffer.rewards.append(reward)
            ppo_agent.buffer.is_terminals.append(done)
            if len(ppo_agent.buffer.rewards) >= 10:
                output_csv_path = os.path.join(log_dir, f"{env_name}_Eval_episode{episode}.csv")
                df.to_csv(output_csv_path, index=False)
                ppo_agent.update()
            if done:
                output_csv_path = os.path.join(log_dir, f"{env_name}_Eval_episode{episode}.csv")
                df.to_csv(output_csv_path, index=False)
                ppo_agent.save(modelSavePath)
                break
        ppo_agent.save(modelSavePath)
    
    
    
    
    pass