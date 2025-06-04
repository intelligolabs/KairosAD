#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import warnings


class PtToOnnx:
    def __init__(self, model):
        self.model = model
        image_size = self.model.mobile_sam.mobile_sam.image_encoder.img_size

        self.dummy_input = {
            'batch_images': torch.randn(
                (1, image_size, image_size, 3), dtype=torch.float32
            )
        }
        self.dynamic_axes = {
            'batch_images': {0: 'batch_size',
                             1: 'image_height',
                             2: 'image_width'}
        }

        _ = self.model(**self.dummy_input)
        self.output_names = ['logits']

    def export(self, model_path):
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=torch.jit.TracerWarning)
            warnings.filterwarnings('ignore', category=UserWarning)
            print(f'Exporting onnx model to {model_path} [...]')

            with open(model_path, 'wb') as fp:
                torch.onnx.export(
                    self.model,
                    tuple(self.dummy_input.values()),
                    fp,
                    export_params=True,
                    verbose=True,
                    opset_version=15,
                    do_constant_folding=False,
                    input_names=list(self.dummy_input.keys()),
                    output_names=self.output_names,
                    dynamic_axes=self.dynamic_axes,
                    operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK
                )
        print(f'{model_path} created sucessfuly!')
