import argparse
import os
import datetime
from pathlib import Path, PosixPath
from .utils import random_string

    
class Parser:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.subparsers = self.parser.add_subparsers(title='subcommands',
                                        description='valid subcommands',
                                        help='description')

        self.paths_parser = self.subparsers.add_parser('paths', help='args for paths to files')
        self.paths_parser.add_argument('--dataset', type=str, default='img_align_celeba')
        self.paths_parser.add_argument('--examples', type=str, default='examples')

        self.training_configs_parser = self.subparsers.add_parser('training', help='training configurations')
        self.training_configs_parser.add_argument('--batchsize', '-bs', type=int, default=16)
        self.training_configs_parser.add_argument('--learning_rate_G', '-lrG', type=float, default=3e-4)
        self.training_configs_parser.add_argument('--learning_rate_D', '-lrD', type=float, default=15e-4)
        self.training_configs_parser.add_argument('--epochs', '-e', type=int, default=10)
        self.training_configs_parser.add_argument("--latent_dim", type=int, default=100)
        self.training_configs_parser.add_argument('--beta1', type=float, default=0.5)

        self.launch_configs_parser = self.subparsers.add_parser('launch', help='special settings for a launch')
        self.launch_configs_parser.add_argument('--wandb', type=str, default=None)
        self.launch_configs_parser.add_argument('--dry_try', action='store_true')

    def parse_args(self, args):
        paths, _ = self.paths_parser.parse_known_args(args)
        
        project_root = Path.cwd()
        curr_time = str(datetime.datetime.now()).replace(' ', '_')
        paths.dataset = project_root / paths.dataset
        
        paths.examples = project_root / paths.examples / curr_time
        paths.examples.parent.mkdir(exist_ok=True)
        paths.examples.mkdir()
        
        training_configs, _ = self.training_configs_parser.parse_known_args(args)
        launch_configs, _ = self.launch_configs_parser.parse_known_args(args)
        return paths, training_configs, launch_configs
