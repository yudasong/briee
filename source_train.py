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

from utils import parse_args, set_seed_everywhere, ReplayBuffer, make_batch_env, make_batch_partition_env

from algs.lsvi_ucb import LSVI_UCB
from algs.lsvi_ucb_rfree import LSVI_UCB_RFREE

from algs.rep_learn import RepLearn

import wandb


os.environ["OMP_NUM_THREADS"] = "1"


def make_rep_learner(env, device, args, path):    

    rep_learners = []
    for h in range(args.horizon):
        if h == 0:
            temperature = args.phi0_temperature
        else:
            temperature = args.temperature

        rep_learners.append(
                RepLearn(env.observation_space.shape[0],
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
                         temp_path = path)
                )
    return rep_learners        


def evaluate(env, agent, args):
    returns = np.zeros((args.num_eval,1))
    
    obs, _ = env.reset()
    for h in range(args.horizon):
        action = agent.act_batch(obs, h)
        next_obs, _, reward, done, _ = env.step(action)
        obs = next_obs
        returns += reward

    return np.mean(returns)

def rep_train(rep_learners,buffers,h, queue):

    feature_loss, adv_loss = rep_learners.update(buffers, plot=False, h=h)

    rep_learners.save_phi(h)

    if queue is not None:
        queue.put([h, feature_loss, adv_loss])
    else:
        return feature_loss, adv_loss

def source_train(args,env_type,log=True):
    
    
    if not log:
        os.environ['WANDB_MODE'] = 'offline'  

    if args.partition:
        run_name = "partition_{}".format(env_type)
    else:
        run_name = "comblock_{}".format(env_type)
    
    project_name =  "result_train_source_{}_s{}".format(args.horizon, args.seed)

    with wandb.init(
            project=project_name,
            job_type="ratio_search",
            config=vars(args),
            name=run_name):

        agent, env = run(args, env_type)
        return agent, env
    

def run(args,env_type):
    
    #set_seed_everywhere(args.seed)

    if not os.path.exists(args.load_path):
        os.makedirs(args.load_path)

    if args.partition:
        env, eval_env = make_batch_partition_env(args, env_type)
        save_path = "partition_{}".format(env_type)
        if env_type != 0:
            temp_path_0 = os.path.join(args.load_path, "partition_0", str(args.seed)) 
            env.opt_a = np.load(os.path.join(temp_path_0, "opt_a.npy"))
            env.opt_b = np.load(os.path.join(temp_path_0, "opt_b.npy"))
            eval_env.opt_a = env.opt_a
            eval_env.opt_b = env.opt_b

    else:
        env, eval_env = make_batch_env(args)
        save_path = "comblock_{}".format(env_type)

    num_actions = env.action_space.n

    device = torch.device("cpu")

    temp_path = os.path.join(args.load_path, save_path, str(args.seed))

    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    np.save(os.path.join(temp_path, "opt_a"), env.opt_a)
    np.save(os.path.join(temp_path, "opt_b"), env.opt_b)
    np.save(os.path.join(temp_path, "rotation"), env.rotation)
    
    for h in range(args.horizon):
        temp_path_h = os.path.join(temp_path, "buffer_{}".format(h))
        if not os.path.exists(temp_path_h):
            os.makedirs(temp_path_h)


    num_runs = int(args.num_episodes / args.horizon / args.num_envs)

    buffers = []    

    for _ in range(args.horizon):
        buffers.append(
                ReplayBuffer(env.observation_space.shape, 
                             env.action_space.n, 
                             int(args.num_episodes / args.horizon) * 2 + args.num_warm_start * args.num_envs, 
                             args.batch_size, 
                             device,
                             recent_size=args.recent_size)
            )

    rep_learners = make_rep_learner(env, device, args, temp_path)

    if args.dense:
        args.alpha = args.horizon / 50
    else:
        args.alpha = args.horizon / 5

    agent = LSVI_UCB(env.observation_space.shape[0],
                     env.state_dim,
                     env.action_dim,
                     args.horizon,
                     args.alpha,
                     device,
                     rep_learners,
                     recent_size = args.lsvi_recent_size,
                     lamb = args.lsvi_lamb)

    if args.horizon >= 50:
        args.num_warm_start = 200

    for _ in range(args.num_warm_start):
        obs, state = env.reset()
        for h in range(args.horizon):
            action = np.random.randint(0, num_actions, args.num_envs)
            next_obs, next_state, reward, done, _ = env.step(action)
            buffers[h].add_batch(obs,state,action,reward,next_obs,next_state,args.num_envs)
            obs = next_obs
            state = next_state

    counts = np.zeros((args.horizon,3),dtype=np.int)

    results = []

    if args.variable_latent:
        returns = deque(maxlen=50)
    else:
        returns = deque(maxlen=5)

    inference_start_time = time.time()

    for n in range(num_runs):

        for h in range(args.horizon):
            t = 0
            obs, state = env.reset()
            while t < h:
                action = agent.act_batch(obs, t)
                next_obs,next_state, _, _, _ = env.step(action)
                obs = next_obs
                state = next_state
                t += 1
            #print(t)
            action = np.random.randint(0, num_actions, args.num_envs)
            next_obs, next_state, reward, done, _ = env.step(action)
            buffers[h].add_batch(obs,state,action,reward,next_obs,next_state,args.num_envs)

            count = env.get_counts()
            counts[h] = counts[h] + count

            if h != args.horizon - 1:
                obs = next_obs
                state = next_state
                action = np.random.randint(0, num_actions, args.num_envs)
                next_obs, next_state, reward, done, _ = env.step(action)
                buffers[h+1].add_batch(obs,state,action,reward,next_obs,next_state,args.num_envs)

                count = env.get_counts()
                counts[h+1] = counts[h+1] + count

            else:
                obs, state = env.reset()
                action = np.random.randint(0, num_actions, args.num_envs)
                next_obs, next_state, reward, done, _ = env.step(action)
                buffers[0].add_batch(obs,state,action,reward,next_obs,next_state,args.num_envs)

                count = env.get_counts()
                counts[0] = count + counts[0]
        
        for b in range(len(buffers)):
            buffers[b].save(os.path.join(temp_path, "buffer_{}".format(b)))

        if n % args.update_frequency == 0:

            inference_time = time.time() - inference_start_time

            assert args.horizon % args.num_threads == 0
            start_time = time.time()
            num_multi_runs = int(args.horizon / args.num_threads) 
            
            feature_loss_list = []
            adv_loss_list = []

            for m in range(num_multi_runs):
                queue = multiprocessing.Queue()
                workers = []
                for i in range(args.num_threads):
                    h = m*args.num_threads + i
                    worker_args = (rep_learners[h], buffers[h], h, queue)
                    workers.append(multiprocessing.Process(target=rep_train, args=worker_args))
                for worker in workers:
                    worker.start()

                for _ in workers:
                    pid, feature_loss, adv_loss = queue.get()
                    feature_loss_list.append(feature_loss)
                    adv_loss_list.append(adv_loss)
                    rep_learners[pid].load_phi(pid)
                
            rep_learn_time = time.time() - start_time

            start_time = time.time()
            agent.update(buffers)
            lsvi_time = time.time() - start_time


            start_time = time.time()

            eval_return = evaluate(eval_env, agent, args)

            returns.append(eval_return)

            eval_time = time.time() - start_time

            reached = 0
            for h in range(args.horizon):
                if counts[h,:2].sum() < 5:
                    reached = h
                    break

            wandb.log({"rep_learn_time": rep_learn_time,
                        "lsvi_time": lsvi_time,
                        "eval": np.mean(list(returns)) if args.variable_latent else eval_return,
                        "episode:": n * args.num_envs,
                        "reached": reached,
                        "state 0": counts[-1,0],
                        "state 1": counts[-1,1],
                        "episode:": n * args.num_envs * args.horizon,
                        "sampling time": inference_time,
                        "eval time": eval_time})


            agent.save_weight(temp_path)

            np.save("{}/counts".format(temp_path), counts)

            inference_start_time = time.time()

            if args.reward_free:
                if counts[-1,0] >= 100 and counts[-1,1] >= 100:
                    return agent, env
            else:
                if np.mean(list(returns)) == 1 and not args.variable_latent and not args.dense:
                    return agent, env

    return agent, env
