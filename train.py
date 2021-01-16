import random
import os
import numpy as np
import datetime
import logging
from PIL import Image
from pathlib import Path

import hydra
import wandb
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import random_split
from torchvision import models
from omegaconf import DictConfig, OmegaConf

from code.dataclasses import get_dataloaders
from code.model import WGAN, weights_init
from code.utils import get_image_files, show_random_examples, random_string
from code.trainer import train, test


log = logging.getLogger(__name__)

# DEVICE = torch.device("cuda")
CONFIG_PATH = "config.yaml"


def prepare_config(config):
    project_root = hydra.utils.get_original_cwd()
    curr_time = str(datetime.datetime.now()).replace(" ", "_")
    print("config.paths.dataset", config.paths.dataset)
    print("config.paths.dataset", project_root)
    config.paths.dataset = os.path.join(project_root, config.paths.dataset)
    config.paths.examples = os.path.join(project_root, config.paths.examples, curr_time)
    os.makedirs(config.paths.examples, exist_ok=True)


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)


@hydra.main(config_path=CONFIG_PATH)
def main(config):
    log.info(OmegaConf.to_yaml(config))
    log.info("Current working directory  : {}".format(os.getcwd()))

    prepare_config(config)

    log.debug("Started training")

    if config.launch.wandb:
        wandb.init(project="celeba_wgan", name=config.launch.wandb)

    train_dataset, valid_dataset, train_loader, valid_loader = get_dataloaders(config)

    model = WGAN(config.training).to(config.training.device)
    weights_init(model)

    try:
        for epoch in range(training_configs.epochs):
            train_losses = train(model, DEVICE, train_loader)
            test_losses = test(model, DEVICE, valid_loader)

            print(f"Epoch {epoch + 1} / {config.training.epochs} \t  [loss_D, loss_G]")
            print(f"\tTrain losses: {[np.round(l.item(), 4) for l in train_losses]}")
            print(f"\tValid loss:   {[np.round(l.item(), 4) for l in test_losses]}")

            show_random_examples(
                model.model_G,
                valid_dataset,
                DEVICE,
                config.paths.examples / f"epoch{epoch + 1}.jpg",
            )
            if lconfig.launch.wandb:
                train_loss_D, train_loss_G = train_losses
                test_loss_D, test_loss_G = test_losses
                wandb.log(
                    {
                        "train_loss_D": train_loss_D,
                        "train_loss_G": train_loss_G,
                        "test_loss_D": test_loss_D,
                        "test_loss_G": test_loss_G,
                    }
                )
    except KeyboardInterrupt:
        print("\nBye")


if __name__ == "__main__":
    main()
