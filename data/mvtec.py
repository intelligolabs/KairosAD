#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import numpy as np

from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split


class MVTecDataset(Dataset):
    categories = ['bottle', 'cable', 'capsule', 'carpet', 'grid',
                  'hazelnut', 'leather', 'metal_nut', 'pill', 'screw',
                  'tile', 'toothbrush', 'transistor', 'wood', 'zipper']

    def __init__(self, root, transform=None, target_transform=None,
                 category='name', test_split=0.2, seed=123):
        self.data = []
        self.nb_classes = 2
        self.is_train = True
        self.transform = transform

        category_path = os.path.join(root, category)

        good_samples = os.listdir(path=f"{category_path}/{'train/good'}")
        good_samples = [os.path.join(f"{category_path}/{'train/good'}", path) for path in good_samples]
        self.data = self.data + list(zip(good_samples, [0 for i in range(len(good_samples))]))

        category_test_path = os.path.join(category_path, 'test')
        for folder in os.listdir(path=category_test_path):
            fault_path = os.path.join(category_test_path, folder)
            samples = [os.path.join(fault_path, path) for path in os.listdir(fault_path)]
            class_label = 0 if folder == 'good' else 1
            self.data = self.data + list(zip(samples, [class_label for i in range(len(samples))]))

        train_data, test_data = train_test_split(self.data, test_size=test_split,
                                                 random_state=seed)
        self.train_data = train_data
        self.test_data = test_data

    def get_dataset(self, is_train = True):
        self.is_train = is_train
        return self

    def __len__(self):
        return len(self.train_data if self.is_train else self.test_data)

    def __getitem__(self, index):
        img_path, label = self.train_data[index] if self.is_train else self.test_data[index]

        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        return np.array(image), np.array(label)
