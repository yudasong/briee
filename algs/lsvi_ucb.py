import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

class LSVI_UCB(object): 

    def __init__(
        self,
        obs_dim,
        state_dim,
        action_dim,
        horizon,
        alpha,
        device,
        rep_learners,
        lamb = 1,
        recent_size=0,
    ):

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.horizon = horizon

        self.feature_dim = state_dim * action_dim

        self.device = device

        self.rep_learners = rep_learners

        self.lamb = lamb
        self.alpha = alpha

        self.recent_size = recent_size

        self.W = torch.rand((self.horizon, self.feature_dim)).to(self.device)
        self.Sigma_invs = torch.zeros((self.horizon, self.feature_dim, self.feature_dim)).to(self.device)

        self.Q_max = torch.tensor(self.horizon)

    def Q_values(self, obs, h):
        Qs = torch.zeros((len(obs),self.action_dim)).to(self.device)
        for a in range(self.action_dim):
            actions = torch.zeros((len(obs),self.action_dim)).to(self.device)
            actions[:,a] = 1
            with torch.no_grad():
                feature = self.rep_learners[h].phi(obs,actions)
            Q_est = torch.matmul(feature, self.W[h].to(self.device)) 
            ucb = torch.sqrt(torch.sum(torch.matmul(feature, self.Sigma_invs[h].to(self.device))*feature, 1))
            
            Qs[:,a] = torch.minimum(Q_est + self.alpha * ucb, self.Q_max)

        return Qs

    def act_batch(self, obs, h):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            Qs = self.Q_values(obs, h)
            action = torch.argmax(Qs, dim=1)

        return action.cpu().data.numpy().flatten()


    def act(self, obs, h):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            Qs = self.Q_values(obs, h)
            action = torch.argmax(Qs, dim=1)

        return action.cpu().data.numpy().flatten()

    def update(self, buffers):
        assert len(buffers) == self.horizon

        for h in range(self.horizon)[::-1]:
            if self.recent_size > 0:
                obses, actions, rewards, next_obses = buffers[h].get_full(device=self.device, recent_size=self.recent_size)
            else:
                obses, actions, rewards, next_obses = buffers[h].get_full(device=self.device)
            
            with torch.no_grad():
                feature = self.rep_learners[h].phi(obses,actions)
            Sigma = torch.matmul(feature.T, feature) + self.lamb * torch.eye(self.feature_dim).to(self.device)
            self.Sigma_invs[h] = torch.inverse(Sigma)

            if h == self.horizon - 1:
                target_Q = rewards
            else:
                Q_prime = torch.max(self.Q_values(next_obses, h+1),dim=1)[0].unsqueeze(-1)
                target_Q = rewards + Q_prime

            self.W[h] = torch.matmul(self.Sigma_invs[h].to(self.device), torch.sum(feature * target_Q, 0))            

    def save_weight(self, path):
        for h in range(self.horizon):
            torch.save(self.W[h],"{}/W_{}.pth".format(path,str(h)))
            torch.save(self.Sigma_invs[h], "{}/Sigma_{}.pth".format(path,str(h)))

    def load_weight(self, path):
        for h in range(self.horizon):
            self.W[h] = torch.load("{}/W_{}.pth".format(path,str(h)))
            self.Sigma_invs[h] = torch.load("{}/Sigma_{}.pth".format(path,str(h)))

