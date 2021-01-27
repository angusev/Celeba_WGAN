import numpy as np
import pandas as pd
import os
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Callable, List, Optional, Sequence, Union, Dict

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
    conditions: torch.Tensor


@dataclass()
class DatasetItem:
    image: Union[torch.Tensor, Image.Image]
    condition: torch.Tensor

    @classmethod
    def collate(cls, items: Sequence["DatasetItem"]) -> ItemsBatch:
        items = list(items)
        return ItemsBatch(
            images=default_collate([item.image for item in items]),
            conditions=default_collate([item.condition for item in items]),
        )


class ImageDataset(Dataset):
    _files: List[str]
    _id2idx: Dict[str, int]
    _conditions: torch.Tensor
    _transform: Transform

    def __init__(
        self,
        files: List[str],
        conditions: np.array,
        id2idx: Dict[str, int],
        transform: Transform,
    ):
        self._files = files
        self._id2idx = id2idx
        self._conditions = conditions
        self._transform = transform

    def __getitem__(self, index: int) -> DatasetItem:
        path = self._files[index]
        filename = path.split("/")[-1]
        image = Image.open(path)
        image = self._transform(image)

        condition = self._conditions[self._id2idx[filename]]

        return DatasetItem(image=image, condition=condition)

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


def get_dataloaders(paths, training_configs, launch_configs):
    files = get_image_files(paths.dataset)

    with open("Anno/list_attr_celeba.txt") as f:
        attrs = list(map(lambda x: x.split(), f.read().splitlines()))
        attrs = pd.DataFrame(attrs[1:], columns=attrs[0])

        id2idx = dict(zip(attrs["id"].values.tolist(), range(attrs.shape[0])))
        attrs = torch.cuda.FloatTensor(
            attrs.drop(columns=["id"]).values.astype(float)
        ).to(training_configs.device)

    if launch_configs.dry_try:
        files = files[::1000]

    dataset = ImageDataset(files, attrs, id2idx, make_transform())
    train_files_count = len(files) * 7 // 10
    valid_files_count = len(files) - train_files_count
    train_dataset, valid_dataset = random_split(
        dataset,
        [train_files_count, valid_files_count],
        generator=torch.Generator().manual_seed(42),
    )
    num_workers = max(os.cpu_count() - 1, 1)

    print(f"train_files number: {train_files_count}")
    print(f"valid_files number: {valid_files_count}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=training_configs.batchsize,
        shuffle=True,
        collate_fn=DatasetItem.collate,
        #         num_workers=num_workers,
        drop_last=True,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=training_configs.batchsize,
        shuffle=True,
        collate_fn=DatasetItem.collate,
        #         num_workers=num_workers,
        drop_last=True,
    )

    return train_dataset, valid_dataset, train_loader, valid_loader
