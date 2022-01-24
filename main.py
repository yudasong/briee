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

from utils import parse_args, set_seed_everywhere, ReplayBuffer, make_batch_env

from algs.lsvi_ucb import LSVI_UCB
from algs.rep_learn import RepLearn


os.environ["OMP_NUM_THREADS"] = "1"


def make_rep_learner(env, device, args):

    rep_learners = []
    for h in range(args.horizon):
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
                         tau = args.phi0_temperature if h == 0 else args.temperature,
                         optimizer = args.optimizer,
                         softmax = args.softmax,
                         reuse_weights = args.reuse_weights,
                         temp_path = args.temp_path)
                )
    return rep_learners        


def evaluate(env, agent, args):
    returns = np.zeros((args.num_eval,1))
    
    obs = env.reset()
    for h in range(args.horizon):
        action = agent.act_batch(obs, h)
        next_obs, reward, done, _ = env.step(action)
        obs = next_obs
        returns += reward

    return np.mean(returns)

def rep_train(rep_learners,buffers,h, queue, counts):

    feature_loss, adv_loss = rep_learners.update(buffers, plot=(h==1))

    rep_learners.save_phi(h)

    if queue is not None:
        queue.put([h, feature_loss, adv_loss])
    else:
        return feature_loss, adv_loss

def main(args):
    
    set_seed_everywhere(args.seed)

    env, eval_env = make_batch_env(args)

    num_actions = env.action_space.n

    device = torch.device("cpu")

    if not os.path.exists(args.temp_path):
        os.makedirs(args.temp_path)

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

    rep_learners = make_rep_learner(env, device, args)

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
        obs = env.reset()
        for h in range(args.horizon):
            action = np.random.randint(0, num_actions, args.num_envs)
            next_obs, reward, done, _ = env.step(action)
            buffers[h].add_batch(obs,action,reward,next_obs,args.num_envs)
            obs = next_obs

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
            obs = env.reset()
            while t < h:
                action = agent.act_batch(obs, t)
                next_obs, _, _, _ = env.step(action)
                obs = next_obs
                t += 1
            #print(t)
            action = np.random.randint(0, num_actions, args.num_envs)
            next_obs, reward, done, _ = env.step(action)
            buffers[h].add_batch(obs,action,reward,next_obs,args.num_envs)

            count = env.get_counts()
            counts[h] = counts[h] + count

            if h != args.horizon - 1:
                obs = next_obs
                action = np.random.randint(0, num_actions, args.num_envs)
                next_obs, reward, done, _ = env.step(action)
                buffers[h+1].add_batch(obs,action,reward,next_obs,args.num_envs)

                count = env.get_counts()
                counts[h+1] = counts[h+1] + count

            else:
                obs = env.reset()
                action = np.random.randint(0, num_actions, args.num_envs)
                next_obs, reward, done, _ = env.step(action)
                buffers[0].add_batch(obs,action,reward,next_obs,args.num_envs)

                count = env.get_counts()
                counts[0] = count + counts[0]

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
                    worker_args = (rep_learners[h], buffers[h], h, queue, counts)
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


            agent.save_weight(args.temp_path)

            np.save("{}/counts".format(args.temp_path), counts)

            inference_start_time = time.time()

            if np.mean(list(returns)) == 1 and not args.variable_latent and not args.dense:
                break


if __name__ == '__main__':

    args = parse_args()

    import wandb

    #os.environ['WANDB_MODE'] = 'offline'
    if args.variable_latent:
        project_name = "bmdp_et{}".format(str(args.env_temperature))
    elif args.dense:
        project_name = "bmdp_dense_h{}".format(str(args.horizon))
    else:
        project_name = "bmdp_h{}".format(args.horizon)

    with wandb.init(
            project= project_name,
            job_type="ratio_search",
            config=vars(args),
            name=args.exp_name):
        main(args)









