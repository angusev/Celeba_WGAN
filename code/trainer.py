import numpy as np
import random
import os
from tqdm import tqdm
from PIL import Image

import torch
import torch.nn as nn
from torch.nn.modules import loss
from torch import optim
from torchvision import transforms


def train(model, device, train_loader):
    model.train()
    model.to(device)
    train_loss_D, train_loss_G = 0., 0.
    
    for batch_idx, batch_data in enumerate(tqdm(train_loader)):
        images, conditions = batch_data.images.to(device), batch_data.conditions.to(device)
        
        model.optimize_parameters(images, conditions)
        
        ratio = len(images) / len(train_loader.dataset)
        train_loss_D += model.losses['loss_D'] * ratio
        train_loss_G += model.losses['loss_G'] * ratio
    return train_loss_D, train_loss_G


@torch.no_grad()
def test(model, device, test_loader):
    model.eval()
    model.to(device)
    test_loss_D, test_loss_G = 0., 0.
    
    for batch_idx, batch_data in enumerate(test_loader):
        images, conditions = batch_data.images.to(device), batch_data.conditions.to(device)
        model.evaluate(images, conditions)
        
        ratio = len(images) / len(test_loader.dataset)
        test_loss_D += model.losses['loss_D'] * ratio
        test_loss_G += model.losses['loss_G'] * ratio
    return test_loss_D, test_loss_G
