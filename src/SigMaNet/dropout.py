from typing import Callable, List, Optional, Tuple
import math

import torch
import torch.nn.functional as F
from torch.nn.modules import Module

import .functional as cF
Tensor = torch.Tensor

class _DropoutNd(Module):
    __constants__ = ['p', 'inplace']
    p: float
    inplace: bool

    def __init__(self, p: float = 0.5, inplace: bool = False) -> None:
        super(_DropoutNd, self).__init__()
        if p < 0 or p > 1:
            raise ValueError("dropout probability has to be between 0 and 1, "
                             "but got {}".format(p))
        self.p = p
        self.inplace = inplace

    def extra_repr(self) -> str:
        return 'p={}, inplace={}'.format(self.p, self.inplace)


class Dropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return cF.complex_fcaller(F.dropout, input, self.p, self.training, self.inplace)


class Dropout2d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return cF.complex_fcaller(F.dropout2d, input, self.p, self.training, self.inplace)


class Dropout3d(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return cF.complex_fcaller(F.dropout3d, input, self.p, self.training, self.inplace)


class AlphaDropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return cF.complex_fcaller(F.alpha_dropout, input, self.p, self.training)


class FeatureAlphaDropout(_DropoutNd):
    def forward(self, input: Tensor) -> Tensor:
        return cF.complex_fcaller(F.feature_alpha_dropoutinput, input, self.p, self.training)