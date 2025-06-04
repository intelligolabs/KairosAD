#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

import numpy as np

from torch import no_grad
from MobileSAM.mobile_sam.predictor import Sam
from MobileSAM.mobile_sam import sam_model_registry
from torchvision.transforms.functional import resize
from MobileSAM.mobile_sam.utils.transforms import ResizeLongestSide


class MSAM(torch.nn.Module):
    def __init__(self, model_type, checkpoint_path, device):
        super().__init__()
        self.mobile_sam = sam_model_registry[model_type](checkpoint_path)
        self.mobile_sam.to(device)
        self.mobile_sam.eval()

        self.device = device
        self.transformation = ResizeLongestSide(self.mobile_sam.image_encoder.img_size)
    
    def transform(self, img, img_format):
        if img_format != Sam.image_format:
            img = img[..., ::-1]

        input_image = self.transformation.apply_image(img)
        input_image_torch = input_image.to(device=self.device).contiguous()[None, :, :, :]

        return input_image_torch

    @no_grad
    def forward(self, img, img_format='RGB'):
        transformed_img  = self.transform(img, img_format)
        input_image = self.mobile_sam.preprocess(transformed_img)

        return self.mobile_sam.image_encoder(input_image)
