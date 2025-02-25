'''
TransNeXt: Robust Foveal Visual Perception for Vision Transformers
Paper: https://arxiv.org/abs/2311.17132
Code: https://github.com/DaiShiResearch/TransNeXt

Author: Dai Shi
Github: https://github.com/DaiShiResearch
Email: daishiresearch@gmail.com

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
'''
from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np
from functools import partial

import torch.nn as nn

from mmseg.models.backbones.mamba.mambaWapper import parse_model, intersect_dicts
from mmseg.models.backbones.mamba.uitils1 import yaml_model_load
from mmseg.registry import MODELS

@MODELS.register_module()
class Mambackbone(nn.Module):

    def __init__(self ,pretrainded):
        super().__init__()
        yaml =yaml_model_load("./mamba-backbone.yaml")
        self.model, self.save = parse_model(deepcopy(yaml), ch=3, verbose=False)  # model, savelist
        da = torch.load(pretrainded,map_location="cpu")
        start_dicts = {k.replace("model.",""):v for k,v in da.items()}
        csd = intersect_dicts(start_dicts, self.model.state_dict(), exclude=[])  # intersect

        self.model.load_state_dict(csd)
        self.model.to("cuda")

    def forward(self, x):
        result = [ ]
        with torch.no_grad():
            y, dt, embeddings = [], [], []  # outputs
            for ix,m in enumerate(self.model):
                if m.f != -1:  # if not from previous layer
                    x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
                x = m(x)  # run
                if ix in [14, 17, 20]:
                    result.append(x)
                y.append(x if m.i in self.save else None)  # save output
        return   result

