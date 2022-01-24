import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

from algs.base_learner import kron, weight_init, Feature, Discriminator, BaseLearner

class RepLearn(BaseLearner):
    def __init__(self,
                obs_dim,
                state_dim,
                action_dim,
                hidden_dim,
                num_update,
                num_feature_update,
                num_adv_update,
                device, 
                **kwargs):

        super().__init__(obs_dim,
                        state_dim,
                        action_dim,
                        hidden_dim,
                        num_update,
                        num_feature_update,
                        num_adv_update,
                        device, 
                        **kwargs)


    def adv_learning(self, replay_buffer, T):

        self.phi_tilde.copy_encoder(self.phi)

        loss_list = []

        total_loss = 0

        for i in range(self.num_adv_update):
            obs, actions, rewards, next_obs = replay_buffer.sample(batch_size=self.batch_size)

            with torch.no_grad():
                dis_out = self.discriminators.get_one(next_obs,T)

                feature = self.phi(obs, actions)
                Sigma = torch.matmul(feature.T, feature) + self.lamb * torch.eye(self.feature_dim).to(self.device)

                feature_tilde = self.phi_tilde(obs, actions)
                Sigma_tilde = torch.matmul(feature_tilde.T, feature_tilde) + self.lamb * torch.eye(self.feature_dim).to(self.device)

                w = torch.matmul(torch.inverse(Sigma), torch.sum(torch.mul(feature,dis_out),0))
                w_tilde = torch.matmul(torch.inverse(Sigma_tilde), torch.sum(torch.mul(feature_tilde,dis_out),0))

                dis_out = self.discriminators.get_one(next_obs,T).squeeze()
                phi_out = torch.matmul(feature, w)


            feature_tilde = self.phi_tilde(obs, actions)            
            phi_tilde_out = torch.matmul(feature_tilde, w_tilde)
            
            loss = F.mse_loss(phi_tilde_out, dis_out) - F.mse_loss(phi_out, dis_out)
            
            self.phi_tilde_optimizer.zero_grad()
            loss.backward()
            self.phi_tilde_optimizer.step()


            with torch.no_grad():
                feature_tilde = self.phi_tilde(obs, actions)            
                phi_tilde_out = torch.matmul(feature_tilde, w_tilde)

            dis_out = self.discriminators.get_one(next_obs,T).squeeze()
            loss = F.mse_loss(phi_tilde_out, dis_out) - F.mse_loss(phi_out, dis_out)

            self.dis_optimizer.zero_grad()
            loss.backward()
            self.dis_optimizer.step()
            loss_list.append(-loss.item())

            total_loss += loss.item()

            

        return loss_list












