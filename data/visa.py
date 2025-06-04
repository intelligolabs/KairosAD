#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

from PIL import Image
from torch.utils.data import Dataset


class ViSADataset(Dataset):
    categories = ['candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
                  'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
                  'pipe_fryum']

    def __init__(self, path: str, category: str, type:str='train', shot:str='fewshot'):
        super().__init__()

        data = pd.read_csv(f"{path}/split_csv/2cls_{shot}.csv")
        data.label = data.label.apply(lambda x: 0 if x == 'normal' else 1)
        self.data = data[(data.object == category) & (data.split == type)]
        self.path = path

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, index):
        row = self.data.iloc[index]
        img_path, label = row.image, row.label
        image = Image.open(f'{self.path}/{img_path}').convert('RGB')

        return np.array(image), np.array(label)
