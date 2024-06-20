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


def BP4D_AU_dataset(json_path, is_train, args):
    transform = build_AU_transform(is_train, args)
    dataset = BP4D_dataset(args.root_path, json_path, transform=transform)
    print(dataset)
    return dataset


def BP4D_plus_AU_dataset(json_path, is_train, args):
    transform = build_AU_transform(is_train, args)
    dataset = BP4D_plus_dataset(args.root_path, json_path, transform=transform)
    print(dataset)
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

class BP4D_dataset(Dataset):
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
        self.AUs = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
        self.IDs = ['F001', 'F002', 'F003', 'F004', 'F005', 'F006', 'F007', 'F008', 'F009', 'F010',
                    'F011', 'F012', 'F013', 'F014', 'F015', 'F016', 'F017', 'F018', 'F019', 'F020',
                    'F021', 'F022', 'F023',
                    'M001', 'M002', 'M003', 'M004', 'M005', 'M006', 'M007', 'M008', 'M009', 'M010',
                    'M011', 'M012', 'M013', 'M014', 'M015', 'M016', 'M017', 'M018']
        self.AU_label2idx = {label: idx for idx, label in enumerate(self.AUs)}
        self.ID_label2idx = {label: idx for idx, label in enumerate(self.IDs)}

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
        AUs = self.data[idx]['AUs']  # e.g. [4, 10, 14]
        ID = self.data[idx]['img_path'][0:4]

        AU_labels = torch.zeros(len(self.AUs))  # 12 classes
        ID_labels = torch.zeros(len(self.IDs))  # 41 classes

        for au in AUs:
            AU_labels[self.AU_label2idx[au]] = 1

        ID_labels[self.ID_label2idx[ID]] = 1

        if self.transform:
            image = self.transform(image)

        return image, (AU_labels, ID_labels)


class BP4D_plus_dataset(Dataset):
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
        self.AUs = [1, 2, 4, 6, 7, 10, 12, 14, 15, 17, 23, 24]
        self.IDs = ['F015', 'M005', 'F057', 'M009', 'M053', 'F078', 'F081', 'M047', 'F004', 'M026', 'M056', 'F058',
                    'F007', 'M031', 'M052', 'F041', 'F014', 'F074', 'M039', 'F044', 'F056', 'F064', 'F048', 'M040',
                    'F024', 'F076', 'M030', 'F016', 'F035', 'F062', 'M014', 'F033', 'M058', 'M055', 'M006', 'M048',
                    'F032', 'M022', 'M036', 'M017', 'F053', 'F002', 'F018', 'F067', 'F050', 'F034', 'F027', 'M001',
                    'F029', 'F012', 'F010', 'M044', 'F079', 'M013', 'M043', 'F059', 'M050', 'M023', 'M008', 'F070',
                    'F011', 'F030', 'F073', 'F069', 'M041', 'F020', 'M054', 'F040', 'M024', 'M034', 'M037', 'F025',
                    'F008', 'F060', 'F022', 'M045', 'M007', 'M015', 'M042', 'M016', 'M038', 'M057', 'F051', 'F009',
                    'F082', 'M049', 'M019', 'F047', 'F054', 'M028', 'F075', 'F042', 'M018', 'F017', 'M003', 'F055',
                    'F046', 'F077', 'F043', 'F065', 'F023', 'F052', 'M012', 'F005', 'M010', 'F013', 'M032', 'M004',
                    'F037', 'F045', 'M046', 'F021', 'M027', 'M051', 'F019', 'F003', 'F068', 'M029', 'M025', 'F028',
                    'M021', 'M020', 'F036', 'F031', 'M002', 'M011', 'F080', 'F066', 'F071', 'F038', 'M035', 'F063',
                    'F026', 'M033', 'F072', 'F049', 'F061', 'F006', 'F001', 'F039']
        self.AU_label2idx = {label: idx for idx, label in enumerate(self.AUs)}
        self.ID_label2idx = {label: idx for idx, label in enumerate(self.IDs)}

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
        AUs = self.data[idx]['AUs']  # e.g. [4, 10, 14]
        ID = self.data[idx]['img_path'][0:4]

        AU_labels = torch.zeros(len(self.AUs))  # 12 classes
        ID_labels = torch.zeros(len(self.IDs))  # 140 classes

        for au in AUs:
            AU_labels[self.AU_label2idx[au]] = 1

        ID_labels[self.ID_label2idx[ID]] = 1

        if self.transform:
            image = self.transform(image)

        return image, (AU_labels, ID_labels)