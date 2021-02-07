import random
import os
import numpy as np
import sys
from PIL import Image

import torch

# torch.multiprocessing.set_sharing_strategy('file_system')
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split
from torchvision import models

from code.dataclasses import get_dataloaders
from code.model import WGAN
from code.utils import show_random_examples, save_random_examples
from code.trainer import train, test
from code.configs import Parser
from fid.fid_score import calculate_fid_given_paths

import wandb


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


def main(args):
    #     set_seed(179)

    parser = Parser()
    paths, training_configs, launch_configs = parser.parse_args(args)
    training_configs.device = torch.device("cuda")


    train_dataset, valid_dataset, train_loader, valid_loader = get_dataloaders(
        paths, training_configs, launch_configs
    )

    model = WGAN(training_configs).to(training_configs.device)
    
    if launch_configs.wandb:
        wandb.init(project="celeba_wgan", name=launch_configs.wandb)
        wandb.watch(model)

    try:
        for epoch in range(training_configs.epochs):
            train_losses = train(model, training_configs.device, train_loader)
            test_losses = test(model, training_configs.device, valid_loader)

            print(f"Epoch {epoch + 1} / {training_configs.epochs} \t  [loss_D, loss_G]")
            print(f"\tTrain losses: {[np.round(l, 4) for l in train_losses]}")
            print(f"\tValid loss:   {[np.round(l, 4) for l in test_losses]}")

            show_random_examples(
                model.model_G,
                valid_dataset,
                training_configs.device,
                paths.examples / f"epoch-{epoch + 1}.jpg",
            )

            if epoch % 10 == 0:
                save_random_examples(
                    model.model_G,
                    valid_dataset,
                    training_configs.device,
                    paths.fid_orig,
                    paths.fid_gen
                )
                fid_score = calculate_fid_given_paths(
                    (paths.fid_orig, paths.fid_gen),
                    batch_size=100,
                    device=torch.device("cuda"),
                    dims=64
                )
                print(f'FID score:', np.round(fid_score, 2))

            if launch_configs.wandb:
                train_loss_D, train_loss_G = train_losses
                test_loss_D, test_loss_G = test_losses
                wandb.log(
                    {
                        "train_loss_D": train_loss_D,
                        "train_loss_G": train_loss_G,
                        "test_loss_D": test_loss_D,
                        "test_loss_G": test_loss_G,
                        "fid_score": fid_score
                    }
                )
    except KeyboardInterrupt:
        print("\nBye")


if __name__ == "__main__":
    main(sys.argv)
