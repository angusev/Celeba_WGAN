import numpy as np
import random
import os
from PIL import Image
from .utils import SSIM

import torch
import torch.nn as nn
from torch.nn.modules import loss
from torch import optim
from torchvision import transforms


def train(model, device, train_loader):
    model.train()
    model.to(device)
    train_loss_D, train_loss_G = 0., 0.
    
    ssim = 0.0
    for batch_idx, batch_data in enumerate(train_loader):
        images = batch_data.images.to(device)
        
        model.optimize_parameters(images)
        
        ratio = len(images) / len(train_loader.dataset)
        train_loss_D += model.losses['loss_D'] * ratio
        train_loss_G += model.losses['loss_G'] * ratio
    return train_loss_D, train_loss_G


@torch.no_grad()
def test(model, device, test_loader):
    model.eval()
    model.to(device)
    test_loss_D, test_loss_G = 0., 0.
    ssim = 0.0
    
    for batch_idx, batch_data in enumerate(test_loader):
        images = batch_data.images.to(device)
        model.evaluate(images)
        
        ratio = len(images) / len(test_loader.dataset)
        test_loss_D += model.losses['loss_D'] * ratio
        test_loss_G += model.losses['loss_G'] * ratio
    return test_loss_D, test_loss_G
