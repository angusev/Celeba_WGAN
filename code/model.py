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


# Size of z latent vector (i.e. size of generator input)
nz = 40


class PrintLayer(nn.Module):
    def __init__(self):
        super(PrintLayer, self).__init__()

    def forward(self, x):
        #         print(x.shape)
        return x


# class Generator(nn.Module):
#     def __init__(self):
#         super(Generator, self).__init__()

#         def block(in_feat, out_feat, normalize=True):
#             layers = [nn.Linear(in_feat, out_feat)]
#             if normalize:
#                 layers.append(nn.BatchNorm1d(out_feat, 0.8))
#             layers.append(nn.LeakyReLU(0.2, inplace=True))
#             return layers

#         self.model = nn.Sequential(
#             *block(80, 128, normalize=False),
#             *block(128, 256),
#             *block(256, 512),
#             *block(512, 1024),
#             nn.Linear(1024, int(np.prod(img_shape))),
#             nn.Tanh()
#         )

#     def forward(self, conditions):
#         noise = torch.randn(conditions.shape).to('cuda')
#         to_input = torch.cat((conditions, noise), 1)
#         img = self.model(to_input)
#         img = img.view(img.shape[0], *img_shape)
#         return img


# class Discriminator(nn.Module):
#     def __init__(self):
#         super(Discriminator, self).__init__()

#         self.model = nn.Sequential(
#             nn.Linear(int((img_shape[0] + 1) * np.prod(img_shape[1:])), 512),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(512, 256),
#             nn.LeakyReLU(0.2, inplace=True),
#             nn.Linear(256, 1),
#         )

#     def forward(self, images, conditions):
#         cond = F.interpolate(conditions.unsqueeze(1).unsqueeze(dim=3),
#                              size=img_shape[1:]).squeeze(dim=3)
#         to_input = torch.cat((cond, images), 1)
#         img_flat = to_input.view(to_input.shape[0], -1)
#         validity = self.model(img_flat)
#         return validity


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        ngf = 64

        def block(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            normalize=True,
        ):
            layers = [
                nn.ConvTranspose2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=normalize,
                    bias=False,
                )
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.main = nn.Sequential(
            *block(nz * 2, ngf * 8, stride=1, padding=0, normalize=False),
            PrintLayer(),
            *block(ngf * 8, ngf * 8),
            PrintLayer(),
            *block(ngf * 8, ngf * 8),
            PrintLayer(),
            *block(ngf * 8, ngf * 4),
            PrintLayer(),
            *block(ngf * 4, ngf * 2),
            PrintLayer(),
            *block(ngf * 2, ngf),
            PrintLayer(),
            torch.nn.ConvTranspose2d(ngf, 3, 4, 2, 1, bias=False),
            PrintLayer(),
            nn.Tanh()
        )

    def forward(self, conditions):
        conditions_exp = conditions.unsqueeze(2).unsqueeze(3)
        noise = torch.randn(conditions_exp.shape).to("cuda")
        to_input = torch.cat((conditions_exp, noise), 1)
        images = self.main(to_input)
        images = images[:, :, : img_shape[1], : img_shape[2]]  # cropping
        return images


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        ndf = 64

        def block(
            in_channels,
            out_channels,
            kernel_size=4,
            stride=2,
            padding=1,
            normalize=True,
        ):
            layers = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=normalize,
                    bias=False,
                )
            ]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.main = nn.Sequential(
            *block(3 + 1, ndf, normalize=False),
            *block(ndf, ndf * 2),
            *block(ndf * 2, ndf * 4),
            *block(ndf * 4, ndf * 8),
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, images, conditions):
        cond = F.interpolate(
            conditions.unsqueeze(1).unsqueeze(dim=3), size=img_shape[1:]
        ).squeeze(dim=3)
        to_input = torch.cat((cond, images), 1)
        return self.main(to_input)


class WGAN(nn.Module):
    def __init__(self, configs):
        super(WGAN, self).__init__()
        self.model_G = Generator().to(configs.device)
        self.model_D = Discriminator().to(configs.device)
        print_params_number(self.model_G)
        print_params_number(self.model_D)

        self.optimizer_G = torch.optim.Adam(
            self.model_G.parameters(),
            lr=configs.learning_rate_G,
            betas=(configs.beta1, 0.999),
        )
        self.optimizer_D = torch.optim.Adam(
            self.model_D.parameters(),
            lr=configs.learning_rate_D,
            betas=(configs.beta1, 0.999),
        )

        self.losses = {"loss_D": None, "loss_G": None}
        self.configs = configs

    def backward_D(self, train_it=True):
        pred_real = self.model_D(self.real_imgs, self.conditions)
        pred_fake = self.model_D(self.fake_imgs, self.conditions)

        self.loss_D = -torch.mean(pred_real) + torch.mean(pred_fake)

        if train_it:
            self.loss_D.backward()

        self.losses["loss_D"] = self.loss_D

    def backward_G(self, train_it=True):
        self.loss_G = -torch.mean(self.model_D(self.fake_imgs, self.conditions))
        if train_it:
            self.loss_G.backward()
        self.losses["loss_G"] = self.loss_G

    @staticmethod
    def set_requires_grad(model, requires_grad):
        for p in model.parameters():
            p.requires_grad = requires_grad

    def optimize_parameters(self, images, conditions):
        self.real_imgs = images
        self.conditions = conditions

        #         conditions = torch.randn(conditions.shape).to('cuda')  ####################!!!!!!!!!!!!!!!!!!
        #         conditions = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1,
        #                                                              (self.real_imgs.shape[0], self.configs.latent_dim))))
        self.fake_imgs = self.model_G(conditions)

        self.set_requires_grad(self.model_D, True)
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.fake_imgs = self.model_G(conditions)

        self.set_requires_grad(self.model_D, False)
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def evaluate(self, images, conditions):
        #         conditions = torch.randn(conditions.shape).to('cuda')  ####################!!!!!!!!!!!!!!!!!!
        #         conditions = Variable(torch.cuda.FloatTensor(np.random.normal(0, 1,
        #                                                              (self.real_imgs.shape[0], self.configs.latent_dim))))

        self.real_imgs = images

        #         self.fake_imgs = self.model_G(conditions.unsqueeze(2).unsqueeze(3))
        self.fake_imgs = self.model_G(conditions)

        self.backward_D(train_it=False)
        self.backward_G(train_it=False)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
