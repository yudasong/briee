import argparse
import torch
import numpy as np

import random
import os

from envs.Lock_batch import LockBatch
from envs.Lock_batch_target import LockBatchTarget
from envs.Lock_batch_part import LockBatchPartition

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num_sources', default=5, type=int)

    parser.add_argument('--opt_sampling', default=False, type=bool)
    parser.add_argument('--target', default=False, type=bool)
    parser.add_argument('--online', default=False, type=bool)
    parser.add_argument('--partition', default=False, type=bool)
    parser.add_argument('--load_transfer_rep', default=False, type=bool)
    parser.add_argument('--load_source', default=False, type=bool)
    parser.add_argument('--reward_free', default=False, type=bool)
    parser.add_argument('--load_pre_train_buffer', default=False, type=bool)

    parser.add_argument('--exp_name', default="test", type=str)
    parser.add_argument('--num_threads', default=10, type=int)
    parser.add_argument('--update_frequency', default=1, type=int)

    parser.add_argument('--temp_path', default="temp", type=str)
    parser.add_argument('--load_path', default="temp", type=str)

    parser.add_argument('--num_target_planning', default=1000, type=int)
    parser.add_argument('--num_envs', default=50, type=int)
    parser.add_argument('--num_plan_envs', default=10, type=int)
    parser.add_argument('--recent_size', default=10000, type=int)
    parser.add_argument('--lsvi_recent_size', default=10000, type=int)
    parser.add_argument('--load', default=False, type=bool)
    parser.add_argument('--dense', default=False, type=bool)

    parser.add_argument('--seed', default=12, type=int)
    parser.add_argument('--num_warm_start', default=0, type=int)
    parser.add_argument('--num_episodes', default=1e6, type=int)
    parser.add_argument('--batch_size', default=512, type=int)

    parser.add_argument('--pretrain_size', default=500, type=int)


    #environment
    parser.add_argument('--horizon', default=25, type=int)
    parser.add_argument('--switch_prob', default=0.5, type=float)
    parser.add_argument('--anti_reward', default=0.1, type=float)
    parser.add_argument('--anti_reward_prob', default=0.5, type=float)
    parser.add_argument('--num_actions', default=10, type=int)
    parser.add_argument('--observation_noise', default=0.1, type=float)
    parser.add_argument('--variable_latent', default=False, type=bool)
    parser.add_argument('--env_temperature', default=0.2, type=float)

    #rep
    parser.add_argument('--rep_num_update', default=30, type=int)
    parser.add_argument('--rep_num_feature_update', default=64, type=int)
    parser.add_argument('--rep_num_adv_update', default=64, type=int)
    parser.add_argument('--discriminator_lr', default=1e-2, type=float)
    parser.add_argument('--discriminator_beta', default=0.9, type=float)
    parser.add_argument('--feature_lr', default=1e-2, type=float)
    parser.add_argument('--feature_beta', default=0.9, type=float)
    parser.add_argument('--linear_lr', default=1e-2, type=float)
    parser.add_argument('--linear_beta', default=0.9, type=float)
    parser.add_argument('--rep_lamb', default=0.01, type=float)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--temperature', default=1, type=float)
    parser.add_argument('--target_temperature', default=1, type=float)
    parser.add_argument('--phi0_temperature', default=0.1, type=float)
    parser.add_argument('--plan_temperature', default=0.1, type=float)

    parser.add_argument('--reuse_weights', default=True, type=bool)
    parser.add_argument('--optimizer', default='sgd', type=str)

    parser.add_argument('--softmax', default='vanilla', type=str)

    #lsvi
    parser.add_argument('--alpha', default=1, type=float)
    parser.add_argument('--lsvi_lamb', default=1, type=float)

    #eval
    parser.add_argument('--num_eval', default=20, type=int)

    args = parser.parse_args()
    return args

def make_batch_env(args):
    env = LockBatch()
    env.init(horizon=args.horizon, 
             action_dim=args.num_actions, 
             p_switch=args.switch_prob, 
             p_anti_r=args.anti_reward_prob, 
             anti_r=args.anti_reward,
             noise=args.observation_noise,
             num_envs=args.num_envs,
             temperature=args.env_temperature,
             variable_latent=args.variable_latent,
             dense=args.dense)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    eval_env = LockBatch()
    eval_env.init(horizon=args.horizon, 
             action_dim=args.num_actions, 
             p_switch=args.switch_prob, 
             p_anti_r=args.anti_reward_prob, 
             anti_r=args.anti_reward,
             noise=args.observation_noise,
             num_envs=args.num_eval,
             temperature=args.env_temperature,
             variable_latent=args.variable_latent,
             dense=args.dense)

    eval_env.seed(args.seed)
    eval_env.opt_a = env.opt_a
    eval_env.opt_b = env.opt_b

    return env, eval_env

def make_batch_target_env(args, source_envs):
    env = LockBatchTarget()
    env.init(horizon=args.horizon, 
             action_dim=args.num_actions, 
             p_switch=args.switch_prob, 
             p_anti_r=args.anti_reward_prob, 
             anti_r=args.anti_reward,
             noise=args.observation_noise,
             num_envs=args.num_envs,
             temperature=args.env_temperature,
             variable_latent=args.variable_latent,
             dense=args.dense,
             partition=args.partition,
             source_envs=source_envs)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    eval_env = LockBatchTarget()
    eval_env.init(horizon=args.horizon, 
             action_dim=args.num_actions, 
             p_switch=args.switch_prob, 
             p_anti_r=args.anti_reward_prob, 
             anti_r=args.anti_reward,
             noise=args.observation_noise,
             num_envs=args.num_eval,
             temperature=args.env_temperature,
             variable_latent=args.variable_latent,
             dense=args.dense,
             partition=args.partition,
             source_envs=source_envs)

    eval_env.seed(args.seed)
    eval_env.opt_a = env.opt_a
    eval_env.opt_b = env.opt_b

    if args.partition:
        eval_env.index0 = env.index0
        eval_env.source_index = env.source_index

    return env, eval_env

def make_batch_partition_env(args, index):
    env = LockBatchPartition()
    env.init(num_partition=args.num_sources,
             index=index,
             horizon=args.horizon, 
             action_dim=args.num_actions, 
             p_switch=args.switch_prob, 
             p_anti_r=args.anti_reward_prob, 
             anti_r=args.anti_reward,
             noise=args.observation_noise,
             num_envs=args.num_envs,
             temperature=args.env_temperature,
             variable_latent=args.variable_latent,
             dense=args.dense)

    env.seed(args.seed)
    env.action_space.seed(args.seed)

    eval_env = LockBatchPartition()
    eval_env.init(num_partition=args.num_sources,
             index=index,
             horizon=args.horizon, 
             action_dim=args.num_actions, 
             p_switch=args.switch_prob, 
             p_anti_r=args.anti_reward_prob, 
             anti_r=args.anti_reward,
             noise=args.observation_noise,
             num_envs=args.num_eval,
             temperature=args.env_temperature,
             variable_latent=args.variable_latent,
             dense=args.dense)

    eval_env.seed(args.seed)
    eval_env.opt_a = env.opt_a
    eval_env.opt_b = env.opt_b
    eval_env.rotation = env.rotation

    return env, eval_env

def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    #if torch.cuda.is_available():
    #    torch.cuda.manual_seed_all(seed)

    np.random.seed(seed)
    random.seed(seed)

class Buffer(object):
    def __init__(self, num_actions):
        
        self.num_actions = num_actions
        self.obses = []
        self.next_obses = []
        self.actions = []
        self.rewards = []
        self.idx = 0

    def add(self, obs, action, reward, next_obs):
        self.obses.append(obs)
        aoh = np.zeros(self.num_actions)
        aoh[action] = 1
        self.actions.append(aoh)
        self.rewards.append(reward)
        self.next_obses.append(next_obs)

        self.idx += 1

    def get_batch(self):
        return self.idx, np.array(self.obses), np.array(self.actions), np.array(self.rewards), np.array(self.next_obses) 

    def get(self, h):
        return self.obses[h], self.actions[h], self.rewards[h], self.next_obses[h]


class ReplayBuffer(object):
    """Buffer to store environment transitions."""
    def __init__(self, obs_shape, num_actions, capacity, batch_size, device, recent_size=0):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.num_actions = num_actions

        self.obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.empty((capacity, num_actions), dtype=np.int)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.states = np.empty((capacity), dtype=np.float32)
        self.next_states = np.empty((capacity), dtype=np.float32)


        self.recent_size = recent_size

        self.idx = 0
        self.last_save = 0
        self.full = False

    def add(self, obs, state, action, reward, next_obs, next_state):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.states[self.idx], state)
        aoh = np.zeros(self.num_actions, dtype=np.int)
        aoh[action] = 1
        np.copyto(self.actions[self.idx], aoh)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.next_states[self.idx], next_state)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def add_batch(self, obs, state, action, reward, next_obs, next_state, size):
        np.copyto(self.obses[self.idx:self.idx+size], obs)
        np.copyto(self.states[self.idx:self.idx+size], state)
        aoh = np.zeros((size,self.num_actions), dtype=np.int)
        aoh[np.arange(size), action] = 1
        np.copyto(self.actions[self.idx:self.idx+size], aoh)
        np.copyto(self.rewards[self.idx:self.idx+size], reward)
        np.copyto(self.next_obses[self.idx:self.idx+size], next_obs)
        np.copyto(self.next_states[self.idx:self.idx+size], next_state)

        self.idx = (self.idx + size) % self.capacity
        self.full = self.full or self.idx == 0

    def add_from_buffer(self, buf, h):
        obs, action, reward, next_obs = buf.get(h)
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def get_full(self, recent_size=0, device=None, state=False):

        if device is None:
            device = self.device

        if self.idx <= recent_size or recent_size == 0: 
            start_index = 0
        else:
            start_index = self.idx - recent_size

        if self.full:
            obses = torch.as_tensor(self.obses[start_index:], device=device)
            states = torch.as_tensor(self.states[start_index:], device=device)
            actions = torch.as_tensor(self.actions[start_index:], device=device)
            rewards = torch.as_tensor(self.rewards[start_index:], device=device)
            next_obses = torch.as_tensor(self.next_obses[start_index:], device=device)
            next_states = torch.as_tensor(self.next_states[start_index:], device=device)
            
                
        else:
            obses = torch.as_tensor(self.obses[start_index:self.idx], device=device)
            states = torch.as_tensor(self.states[start_index:self.idx], device=device)
            actions = torch.as_tensor(self.actions[start_index:self.idx], device=device)
            rewards = torch.as_tensor(self.rewards[start_index:self.idx], device=device)
            next_obses = torch.as_tensor(self.next_obses[start_index:self.idx], device=device)
            next_states = torch.as_tensor(self.next_states[start_index:self.idx], device=device)
        
        if state:
            return obses, state, actions, rewards, next_obses, next_states
        else:
            return obses, actions, rewards, next_obses

    def sample(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size

        if self.recent_size == 0 or self.idx < self.recent_size: 
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size 
            )
        else:
            idxs = np.random.randint(
                self.idx - self.recent_size, self.capacity if self.full else self.idx, size=self.batch_size 
            )


        obses = torch.as_tensor(self.obses[idxs], device=self.device)
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device)
        
        return obses, actions, rewards, next_obses

    def sample_visitation(self, batch_size=50):
        if batch_size is None:
            batch_size = self.batch_size

        if self.recent_size == 0 or self.idx < self.recent_size: 
            idxs = np.random.randint(
                0, self.capacity if self.full else self.idx, size=self.batch_size 
            )
        else:
            idxs = np.random.randint(
                self.idx - self.recent_size, self.capacity if self.full else self.idx, size=self.batch_size 
            )

        obses = self.obses[idxs].copy()
        states = self.states[idxs].copy()
        actions = self.actions[idxs].copy().argmax(-1)
        
        return obses, states, actions
        
    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.states[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.next_states[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.states[start:end] = payload[1]
            self.next_obses[start:end] = payload[2]
            self.next_states[start:end] = payload[3]
            self.actions[start:end] = payload[4]
            self.rewards[start:end] = payload[5]
            self.idx = end
