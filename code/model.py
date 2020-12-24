from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Union

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.dataloader import default_collate
from torch.utils.data.dataset import Dataset
from torchvision import transforms

from .utils import print_params_number


img_shape = (3, 218, 178)


class Generator(nn.Module):
    def __init__(self, configs):
        super(Generator, self).__init__()
        
        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *block(configs.latent_dim, 128, normalize=False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(img_shape))),
            nn.Tanh()
        )

    def forward(self, z):
        img = self.model(z)
        img = img.view(img.shape[0], *img_shape)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(int(np.prod(img_shape)), 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1)
        validity = self.model(img_flat)
        return validity


class WGAN(nn.Module):
    def __init__(self, configs):
        super(WGAN, self).__init__()
        self.model_G = Generator(configs).to(configs.device)
        self.model_D = Discriminator().to(configs.device)

        print_params_number(self.model_G)
        print_params_number(self.model_D)
        
        self.optimizer_G = torch.optim.Adam(self.model_G.parameters(),
                                            lr=configs.learning_rate,
                                            betas=(configs.beta1, 0.999))
        self.optimizer_D = torch.optim.Adam(self.model_D.parameters(),
                                            lr=configs.learning_rate,
                                            betas=(configs.beta1, 0.999))

        self.losses = {'loss_D': None, 'loss_G': None}
        self.configs = configs


    def backward_D(self, train_it=True):
        pred_real = self.model_D(self.real_imgs)
        pred_fake = self.model_D(self.real_imgs)
        
        self.loss_D =  -torch.mean(pred_real) + torch.mean(pred_fake)
        
        if train_it:
            self.loss_D.backward()
        
        self.losses['loss_D'] = self.loss_D
        
    def backward_G(self, train_it=True):
        self.loss_G =  -torch.mean(self.model_D(self.fake_imgs))
        if train_it:
            self.loss_G.backward()
        self.losses['loss_G'] = self.loss_G

        
    def set_requires_grad(self, model, requires_grad):
        for p in model.parameters():
            p.requires_grad = requires_grad
                 
    def optimize_parameters(self, images):
        self.real_imgs = images
        
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1,
                                                             (self.real_imgs.shape[0], self.configs.latent_dim))))
        self.fake_imgs = self.model_G(z)

        self.set_requires_grad(self.model_D, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()
        
        self.set_requires_grad(self.model_D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def evaluate(self, images):
        self.real_imgs = images
        z = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1,
                                                             (self.real_imgs.shape[0], self.configs.latent_dim))))
        self.fake_imgs = self.model_G(z)
        
        self.backward_D(train_it=False)
        self.backward_G(train_it=False)
