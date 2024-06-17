# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from PIL import Image
import json
import zipfile
import io
import numpy as np
import lmdb

import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    root = os.path.join(args.data_path, 'train' if is_train else 'val')
    dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


class CustomDataset(Dataset):
    def __init__(self, lmdb_path=None, transform=None):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False, readahead=False, meminit=False)
        self.transform = transform

        with self.env.begin(write=False) as txn:
            self.keys = [key for key, _ in txn.cursor()]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        with self.env.begin(write=False) as txn:
            key = self.keys[index]
            img_bytes = txn.get(key)

        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')

        if self.transform:
            img = self.transform(img)

        return img


def build_AU_dataset(json_path, is_train, args):
    transform = build_AU_transform(is_train, args)
    dataset = AUDataset(args.root_path, json_path, transform=transform)

    return dataset


def build_AU_transform(is_train, args):
    mean = IMAGENET_DEFAULT_MEAN
    std = IMAGENET_DEFAULT_STD

    # train transform
    if is_train:
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform
    # eval transform
    else:
        return transforms.Compose(
            [transforms.Resize([args.input_size, args.input_size]),
             transforms.ToTensor(),
             transforms.Normalize(mean, std)]
        )

class AUDataset(Dataset):
    """
    accept the json file to construct the dataset.
    Each line of the json file contains the image path and the AU labels.
    """
    def __init__(self, root_path, json_file, transform=None):
        self.data = self._load_data(json_file)
        print(f"building dataset from: {json_file}")
        self.root_path = root_path
        print(f"dataset path: {self.root_path}")
        self.transform = transform
        # self.AUs = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
        self.AUs = ['F01', 'F02', 'F03', 'F04', 'F05', 'F06', 'F07', 'F08', 'F09', 'F10',
                    'F11', 'F12', 'F13', 'F14', 'F15', 'F16', 'F17', 'F18', 'F19', 'F20',
                    'F21', 'F22', 'F23',
                    'M01', 'M02', 'M03', 'M04', 'M05', 'M06', 'M07', 'M08', 'M09', 'M10',
                    'M11', 'M12', 'M13', 'M14', 'M15', 'M16', 'M17', 'M18']
        self.label2idx = {label: idx for idx, label in enumerate(self.AUs)}

    def _load_data(self, json_file):
        dict_list = []
        with open(json_file, 'r') as file:
            for line in file:
                loaded_dict = json.loads(line)
                dict_list.append(loaded_dict)
        return dict_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.root_path + self.data[idx]['img_path']
        image = Image.open(image_path).convert('RGB')

        # Convert label indices to binary representation
        # AUs = self.data[idx]['AUs']  # e.g. [4, 10, 14]
        AUs = self.data[idx]['img_path'][1:4]

        labels = torch.zeros(len(self.AUs))  # 12 classes
        # for au in AUs:
        #     labels[self.label2idx[au]] = 1
        labels[self.label2idx[AUs]] = 1

        if self.transform:
            image = self.transform(image)

        return image, labels