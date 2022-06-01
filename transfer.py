import numpy as np
import torch
import os
import math
import gym
import sys
import random
import time
import json
import copy

from collections import deque
import multiprocessing

from utils import make_batch_env, parse_args, set_seed_everywhere, ReplayBuffer, make_batch_target_env, make_batch_partition_env
from envs.Lock import Lock

from algs.lsvi_ucb import LSVI_UCB

from source_train import source_train, make_rep_learner, rep_train, evaluate

from algs.multi_rep_learn import MultiRepLearn

def make_multirep_learner(env, device, args, path):    

    rep_learners = []
    for h in range(args.horizon):
        if h == 0:
            temperature = args.phi0_temperature
        else:
            temperature = args.target_temperature

        rep_learners.append(
                MultiRepLearn(env.observation_space.shape[0],
                         env.state_dim,
                         env.action_dim,
                         args.hidden_dim,
                         args.rep_num_update,
                         args.rep_num_feature_update,
                         args.rep_num_adv_update,
                         device,
                         discriminator_lr=args.discriminator_lr,
                         discriminator_beta=args.discriminator_beta,
                         feature_lr=args.feature_lr,
                         feature_beta=args.feature_beta,
                         weight_lr=args.linear_lr,
                         weight_beta=args.linear_beta, 
                         batch_size = args.batch_size,
                         lamb = args.rep_lamb,
                         tau =  temperature,
                         optimizer = args.optimizer,
                         softmax = args.softmax,
                         reuse_weights = args.reuse_weights,
                         temp_path = path,
                         num_sources = args.num_sources)
                )
    return rep_learners   

def load_source(args, noise_types, device):

    source_policies = []
    source_envs = []

    for i in range(args.num_sources):

        if args.partition:
            env, _ = make_batch_partition_env(args, i)
            save_path = "partition_{}".format(i)
        else:
            env, _ = make_batch_env(args)
            save_path = "comblock_{}".format(i)
        
        temp_path = os.path.join(args.load_path, save_path, str(args.seed))
        env.opt_a = np.load(os.path.join(temp_path, "opt_a.npy"))
        env.opt_b = np.load(os.path.join(temp_path, "opt_b.npy"))

        source_envs.append(env)

        rep_learners = make_rep_learner(env, device, args, temp_path)
        for h in range(args.horizon):
            rep_learners[h].load_phi(h)
        
        agent = LSVI_UCB(env.observation_space.shape[0],
                    env.state_dim,
                    env.action_dim,
                    args.horizon,
                    args.alpha,
                    device,
                    rep_learners,
                    recent_size = args.lsvi_recent_size,
                    lamb = args.lsvi_lamb)

        agent.load_weight(temp_path)

        source_policies.append(agent)

    return source_policies, source_envs

def generate_transfer_buffer(args, source_policies, source_envs, device):

    env = source_envs[0]

    num_states = env.state_dim
    num_actions = env.action_space.n
    pre_train_buffer = []    

    for _ in range(args.horizon):
        buffer_h = []
        for _ in range(args.num_sources):
            buffer_h.append(
                    ReplayBuffer(env.observation_space.shape, 
                                env.action_space.n, 
                                args.pretrain_size * args.num_envs * args.num_sources ** 2 * 2 + 100, 
                                args.batch_size, 
                                device)
                )

        pre_train_buffer.append(buffer_h)

    if args.load_pre_train_buffer:
        return pre_train_buffer

    for n in range(args.pretrain_size):
        #print(n)
        for h in range(args.horizon):
            for i in range(args.num_sources):                
                
                if not args.opt_sampling:
                    agent = source_policies[i]
                    t = 0
                    obs, state = source_envs[i].reset()
                    while t < h-1:
                        action = agent.act_batch(obs, t)
                        next_obs,next_state, _, _, _ = source_envs[i].step(action)
                        obs = next_obs
                        state = next_state
                        t += 1
                    if n <= args.pretrain_size - args.pretrain_size/10:
                        action_i = agent.act_batch(obs, t)
                    else:
                        action_i = np.random.randint(0, num_actions, args.num_envs)

                    obs_i = obs
                    state_i = state
                
                for cur_j in range(args.num_sources):

                    num_repeat = 1

                    if not args.opt_sampling:
                        if args.online:
                            j = i
                        else:
                            j = cur_j
                        num_repeat = 1
                        if i == cur_j:
                            num_repeat = max(args.num_sources-1,1) 

                    for _ in range(num_repeat):

                        if not args.opt_sampling:
                            if h != 0:
                                obs,state, _, _, _ = source_envs[j].reset_and_step(h-1, obs_i, state_i, action_i)
                            else:
                                obs,state = source_envs[j].reset()
                                
                        else:
                            if args.online:
                                cur_j = i
                            state = np.random.randint(0,num_states, size=[args.num_envs])
                            obs = source_envs[cur_j].make_obs(state,h)

                        action = np.random.randint(0, num_actions, args.num_envs)
                        next_obs, next_state, reward, done, _ = source_envs[i].reset_and_step(h, obs, state, action)                        
                        pre_train_buffer[h][i].add_batch(obs,state,action,reward,next_obs,next_state,args.num_envs)
                        
    for h in range(args.horizon):
        for s in range(args.num_sources):
            path_hs = os.path.join(args.temp_path, "buffer_{}_{}".format(h,s))
            if not os.path.exists(path_hs):
                os.makedirs(path_hs)
            pre_train_buffer[h][s].save(path_hs)

    return pre_train_buffer

def plan(args, source_envs, transfer_rep_learners, device):

    args.num_envs = args.num_plan_envs
    
    target_env, target_eval_env = make_batch_target_env(args, source_envs)

    num_actions = target_env.action_space.n

    target_agent = LSVI_UCB(target_env.observation_space.shape[0],
                        target_env.state_dim,
                        target_env.action_dim,
                        args.horizon,
                        args.alpha,
                        device,
                        transfer_rep_learners,
                        recent_size = args.lsvi_recent_size,
                        lamb = args.lsvi_lamb)

    target_buffer = []    

    for _ in range(args.horizon):
        target_buffer.append(
                ReplayBuffer(target_env.observation_space.shape, 
                             target_env.action_space.n, 
                             args.num_target_planning * args.num_envs + 100, 
                             args.batch_size, 
                             device,
                             recent_size=args.recent_size)
        )

    counts = np.zeros((args.horizon,3),dtype=np.int)

    returns = deque(maxlen=5)

    for n in range(args.num_target_planning):

        obs, state = target_env.reset()
        for h in range(args.horizon):
            with torch.no_grad():
                latent = transfer_rep_learners[h].phi.encode_state(obs)

            action = target_agent.act_batch(obs, h)
            rand_action = np.random.randint(0, num_actions, args.num_envs)
            action[0] = rand_action[0]

            next_obs, next_state, reward, done, _ = target_env.step(action)

            target_buffer[h].add_batch(obs,state,action,reward,next_obs,next_state,args.num_envs)
            obs = next_obs
            state = next_state

            count = target_env.get_counts()
            counts[h] = counts[h] + count

        target_agent.update(target_buffer)

        eval_return = evaluate(target_eval_env, target_agent, args)
        returns.append(eval_return)

        wandb.log({"eval": np.mean(list(returns)) if args.variable_latent else eval_return,
                "episode:": n * args.num_envs})

        if np.mean(list(returns)) == 1:
            return


def main(args):
    
    set_seed_everywhere(args.seed)
    if not os.path.exists(args.temp_path):
        os.makedirs(args.temp_path)

    device = torch.device("cpu")
    
    if args.load_source:
        source_policies, source_envs = load_source(args, device)

    else:
        source_policies = []
        source_envs = []

        for i in range(args.num_sources):
            policy, env = source_train(args,i)
            source_policies.append(policy)
            source_envs.append(env)

    if args.target:

        env = source_envs[0]
        transfer_rep_learners = make_multirep_learner(env, device, args, args.temp_path)

        if args.load_transfer_rep:
            for h in range(args.horizon):
                transfer_rep_learners[h].load_phi(h)
                transfer_rep_learners[h].phi.tau = args.plan_temperature
        
        else:
            pre_train_buffer = generate_transfer_buffer(args, source_policies, source_envs, device)

            if args.load_pre_train_buffer:
                for h in range(args.horizon):
                    for s in range(args.num_sources):
                        path_hs = os.path.join(args.temp_path, "buffer_{}_{}".format(h,s))
                        pre_train_buffer[h][s].load(path_hs)        

            queue = multiprocessing.Queue()
            workers = []
            for h in range(args.horizon):
                worker_args = (transfer_rep_learners[h], pre_train_buffer[h], h, queue)
                workers.append(multiprocessing.Process(target=rep_train, args=worker_args))
            
            for worker in workers:
                worker.start()

            for _ in workers:
                pid, _, _ = queue.get()
                transfer_rep_learners[pid].load_phi(pid)
                transfer_rep_learners[pid].phi.tau = args.plan_temperature
        
        
        
            exp_name = "seed_{}".format(args.seed)
            project_name = "transfer" if not args.online else "online_transfer"
            if args.partition:
                project_name += "_partition"

            with wandb.init(
                    project=project_name,
                    job_type="ratio_search",
                    config=vars(args),
                    name=exp_name): 
                plan(args, source_envs, transfer_rep_learners, device)


if __name__ == '__main__':

    args = parse_args()

    import wandb
    main(args)









