#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

from torch import nn


class ClassificationBlock(nn.Module):
    def __init__(self, in_features, out_features, activation):
        super().__init__()
        self.activation = activation
        self.batchNormalization = nn.BatchNorm1d(256)
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x):
        x = self.linear(x)
        x = self.batchNormalization(x)
        # x = nn.functional.dropout(x, p=0.3) # For ViSA dataset.
        x = nn.functional.dropout(x, p=0.1)   # For MVTec-AD dataset.

        return self.activation(x)


class KairosAD(nn.Module):
    def __init__(self, sam_model, number_of_blocks=5, divider=2):
        super().__init__()
        self.mobile_sam = sam_model
        self.embeding_dim = [1, 256, 64, 64]

        in_features = (self.embeding_dim[2] * self.embeding_dim[3])

        self.LinearBlocks = nn.ModuleList()
        for _ in range(number_of_blocks-1):
            layer = ClassificationBlock(in_features, in_features//divider, nn.functional.relu)
            self.LinearBlocks.append(layer)
            in_features //= divider
        self.LinearBlocks.append(nn.Linear(in_features, 1))

    def batch_embedings(self, images):
        embedings = []
        for img in images:
            embedings.append(self.mobile_sam(img))

        return torch.cat(embedings, dim=0)

    def forward(self, batch_images):
        x = self.batch_embedings(batch_images)
        x = x.reshape(-1, x.shape[1], x.shape[2] * x.shape[3])

        for block in self.LinearBlocks:
            x = block(x)
        x = x.mean(dim=1)

        return x
