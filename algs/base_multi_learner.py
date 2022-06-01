import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math

import matplotlib.pyplot as plt

def kron(a, b):
    siz1 = torch.Size(torch.tensor(a.shape[-1:]) * torch.tensor(b.shape[-1:]))
    res = a.unsqueeze(-1) * b.unsqueeze(-2)
    siz0 = res.shape[:-2]
    return res.reshape(siz0 + siz1)

def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Feature(nn.Module):
    def __init__(self, obs_dim, action_dim, device, state_dim=3, tau=1, num_sources=3, softmax="vanilla"):
        super().__init__()

        self.device = device
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim

        self.tau = tau
        self.softmax = softmax

        self.num_sources = num_sources

        self.encoder = nn.Linear(obs_dim, state_dim, bias=False)
        self.weights = []
        for s in range(num_sources):
            self.weights.append(nn.Linear(state_dim * action_dim, 1, bias=False))

        self.apply(weight_init)

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)
        state_encoding = self.encoder(obs)
        if self.softmax == "gumble":
            state_encoding = F.gumbel_softmax(state_encoding, tau=self.tau, hard=False)
        elif self.softmax == 'vanilla': 
            state_encoding = F.softmax(state_encoding / self.tau, dim=-1)
        phi = kron(action, state_encoding)

        return phi

    def encode_state(self, obs):
        obs = torch.FloatTensor(obs).to(self.device)
        state_encoding = self.encoder(obs)
        if self.softmax == "gumble":
            state_encoding = F.gumbel_softmax(state_encoding, tau=self.tau, hard=False)
        elif self.softmax == 'vanilla': 
            state_encoding = F.softmax(state_encoding / self.tau, dim=-1)
        
        return state_encoding

    def reset_weights(self, T):
        self.weights = []
        for s in num_sources:
            self.weights.append(nn.Linear(self.state_dim * self.action_dim, T+1, bias=False).to(self.device))

    def predict(self, obs, action, source_index):
        phi = self.forward(obs,action)
        out = self.weights[source_index](phi)
        return out

    def copy_encoder(self, target):
        self.encoder.load_state_dict(target.encoder.state_dict())

    def save_encoder(self, path):
        torch.save(self.encoder.state_dict(), path)

    def load_encoder(self, path):
        state_dict = torch.load(path)
        self.encoder.load_state_dict(state_dict)


class Discriminator(nn.Module):
    def __init__(self, obs_dim, hidden_dim, total_rounds):
        super().__init__()

        self.obs_dim = obs_dim
        self.hidden_dim = hidden_dim

        self.networks = nn.ModuleList([self.create_network() for _ in range(total_rounds)])
        self.num_nets = total_rounds

        self.apply(weight_init)
        

    def forward(self, obs):
        return torch.cat([self.networks[i] for i in range(self.num_nets)],-1)

    def get_one(self, obs, t):
        return self.networks[t](obs)

    def get_till(self, obs, t):
        return torch.cat([self.networks[i](obs) for i in range(t+1)],-1)

    def create_network(self):
        net = nn.Sequential(
            nn.Linear(self.obs_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim), nn.Tanh(),
            nn.Linear(self.hidden_dim, 1), nn.Tanh()
        )

        return net



class BaseLearner(object):
    def __init__(
        self,
        obs_dim,
        state_dim,
        action_dim,
        hidden_dim,
        num_update,
        num_feature_update,
        num_adv_update,
        device,
        discriminator_lr=1e-3,
        discriminator_beta=0.9,
        feature_lr=1e-3,
        feature_beta=0.9,
        weight_lr=1e-3,
        weight_beta=0.9, 
        batch_size = 128,
        lamb = 1,
        tau = 1,
        optimizer = "sgd",
        softmax = "vanilla",
        reuse_weights = True,
        temp_path = "temp",
        num_sources = 3,
    ):

        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.feature_dim = state_dim * action_dim
        
        self.device = device

        self.lamb = lamb

        self.num_feature_update = num_feature_update
        self.num_adv_update = num_adv_update
        self.num_update = num_update

        self.batch_size = batch_size

        self.phi = Feature(obs_dim, action_dim, device, tau=tau, softmax=softmax).to(device)
        self.phi_tilde = Feature(obs_dim, action_dim, device, tau=tau, softmax=softmax).to(device)

        self.optimizer = optimizer

        self.reuse_weights = reuse_weights

        self.discriminators = []
        for s in range(num_sources):
            self.discriminators.append(
                Discriminator(obs_dim, hidden_dim, num_update).to(device)
            )

        self.feature_lr = feature_lr
        self.feature_beta = feature_beta

        self.discriminator_lr = discriminator_lr
        self.discriminator_beta = discriminator_beta

        self.weight_lr = weight_lr

        self.temp_path = temp_path

        self.num_sources = num_sources

        self.dis_optimizer = []

        if self.optimizer == "Adam":
            self.phi_optimizer = torch.optim.Adam(
                self.phi.parameters(), lr=feature_lr, betas=(feature_beta, 0.999)
            )
            self.phi_tilde_optimizer = torch.optim.Adam(
                self.phi_tilde.parameters(), lr=feature_lr, betas=(feature_beta, 0.999)
            )
            for s in range(num_sources):
                self.dis_optimizer.append(torch.optim.Adam(
                    self.discriminators[s].parameters(), lr=discriminator_lr, betas=(discriminator_beta, 0.999)
                ))

        else:
            self.phi_optimizer = torch.optim.SGD(
                self.phi.parameters(), lr=feature_lr, momentum=0.99
            )
            self.phi_tilde_optimizer = torch.optim.SGD(
                self.phi_tilde.parameters(), lr=feature_lr, momentum=0.99
            )
            for s in range(num_sources):
                self.dis_optimizer.append(torch.optim.SGD(
                    self.discriminators[s].parameters(), lr=discriminator_lr, momentum=0.99
                ))



    def feature_learning(self, replay_buffer, T):

        total_loss = 0

        loss_list = []

        for i in range(self.num_feature_update):
            loss = 0
            
            for s in range(self.num_sources):
                obs, actions, rewards, next_obs = replay_buffer[s].sample(batch_size=self.batch_size)

                with torch.no_grad():
                    dis_out = self.discriminators[s].get_till(next_obs, T)
                    feature = self.phi(obs,actions)
                    Sigma = torch.matmul(feature.T, feature) + self.lamb * torch.eye(self.feature_dim).to(self.device)

                W = torch.matmul(torch.inverse(Sigma), torch.sum(torch.mul(feature.unsqueeze(-1),dis_out.unsqueeze(-2)),0))
                feature = self.phi(obs,actions)
                out = torch.matmul(feature, W)
                loss = loss + F.mse_loss(out, dis_out)  

            loss = loss / self.num_sources
            self.phi_optimizer.zero_grad()
            loss.backward()
            self.phi_optimizer.step()

            total_loss += loss.item()

            loss_list.append(loss.item())

        return loss_list



    def adv_learning(self, replay_buffer, T):

        pass


    def update(self, replay_buffer, plot=False, h=0):

        self.phi.apply(weight_init)
        
        self.phi_tilde.apply(weight_init)
        
        for s in range(self.num_sources):
            self.discriminators[s].apply(weight_init)        

        feature_losses = []
        adv_losses = []

        for t in range(self.num_update-1):
            feature_loss = self.feature_learning(replay_buffer, t)
            adv_loss = self.adv_learning(replay_buffer, t+1)

            feature_losses.extend(feature_loss)
            adv_losses.extend(adv_loss)

        feature_loss = self.feature_learning(replay_buffer, self.num_update-1)
        feature_losses.extend(feature_loss)

        if plot:
            plt.plot(feature_losses)
            plt.savefig("{}/feature_loss_{}.pdf".format(self.temp_path, h))
            plt.close()
            plt.plot(adv_losses)
            plt.savefig("{}/adv_loss_{}.pdf".format(self.temp_path, h))
            plt.close()

        return np.mean(feature_losses), np.mean(adv_losses)


    def save_phi(self,h):
        self.phi.save_encoder("{}/phi_{}.pth".format(self.temp_path,str(h)))
        self.phi_tilde.save_encoder("{}/phi_tilde_{}.pth".format(self.temp_path,str(h)))

    def load_phi(self,h):
        self.phi.load_encoder("{}/phi_{}.pth".format(self.temp_path,str(h)))
        self.phi_tilde.load_encoder("{}/phi_tilde_{}.pth".format(self.temp_path,str(h)))
