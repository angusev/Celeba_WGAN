import numpy as np
import os
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import models, transforms

from .utils import get_image_files

Transform = Callable[[Image.Image], Image.Image]


@dataclass()
class ItemsBatch:
    images: torch.Tensor


@dataclass()
class DatasetItem:
    image: Union[torch.Tensor, Image.Image]

    @classmethod
    def collate(cls, items: Sequence["DatasetItem"]) -> ItemsBatch:
        items = list(items)
        return ItemsBatch(
            images=default_collate([item.image for item in items]),
        )


class ImageDataset(Dataset):
    _files: List[str]
    _transform: Transform

    def __init__(self, files: List[str], transform: Transform):
        self._files = files
        self._transform = transform

    def __getitem__(self, index: int) -> DatasetItem:
        path = self._files[index]
        image = Image.open(path)
        image = self._transform(image)
        
        return DatasetItem(image=image)

    def __len__(self) -> int:
        return len(self._files)

    
def make_transform():
    transform = transforms.Compose(
        [
            # transforms.Resize((250,250)),
            # transforms.RandomRotation(20, resample=Image.BILINEAR),
            # transforms.CenterCrop(224),
            # transforms.ColorJitter(hue=.05, saturation=.05),
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            # transforms.Lambda(lambda x: x + torch.rand(x.shape) * 1e-2),
            # transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262])
        ]
    )
    
    return transform


def get_dataloaders(dataset_path, training_configs, launch_configs):
    files = get_image_files(dataset_path)
    if launch_configs.dry_try:
        files = files[::100]
    
    train_files_count = len(files) * 7 // 10
    train_files, valid_files = random_split(files,
                                            [train_files_count, len(files) - train_files_count], 
                                            generator=torch.Generator().manual_seed(42))
    num_workers = max(os.cpu_count() - 1, 1)

    train_dataset = ImageDataset(train_files, make_transform())
    valid_dataset = ImageDataset(valid_files, make_transform())
    
    print(f'train_files number: {len(train_files)}')
    print(f'valid_files number: {len(valid_files)}')
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_configs.batchsize,
        shuffle=True,
        collate_fn=DatasetItem.collate,
        num_workers=num_workers,
        drop_last=True
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=training_configs.batchsize,
        shuffle=True,
        collate_fn=DatasetItem.collate,
        num_workers=num_workers,
        drop_last=True
    )
    
    return train_loader, valid_loader
