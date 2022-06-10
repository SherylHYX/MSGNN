from typing import Callable, List, Optional, Tuple
import math
import warnings
import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import ParameterList
from torch.nn import _reduction as _Reduction
from torch.overrides import (
    has_torch_function, has_torch_function_unary, has_torch_function_variadic,
    handle_torch_function)

Tensor = torch.Tensor

def complex_fcaller(funtional_handle, *args):
    return torch.complex(funtional_handle(args[0].real, *args[1:]), funtional_handle(args[0].imag, *args[1:]))

def c_sigmoid(input: Tensor):
    if input.is_complex():
        return torch.complex(F.sigmoid(input.real), F.sigmoid(input.imag))
    else:
        return F.sigmoid(input)

def c_tanh(input: Tensor):
    if input.is_complex():
        return torch.complex(F.tanh(input.real), F.tanh(input.imag))
    else:
        return F.tanh(input)

def mod_tanh(input: Tensor) -> Tensor:
    if input.is_complex():
        magnitude = torch.abs(input)
        euler_phase = torch.div(input, magnitude)
        return torch.mul(F.tanh(magnitude).type(input.type()), euler_phase)
    else:
        return F.tanh(input)

def siglog(input: Tensor):
    return torch.div(input, 1 + torch.abs(input))

def c_cardioid(input: Tensor):
    phase = torch.angle(input)
    return 0.5 * torch.mm(1 + torch.cos(phase), input)

def c_relu(input: Tensor, inplace: bool = False) -> Tensor:
    if input.is_complex():
        return torch.complex(F.relu(input.real, inplace=inplace), F.relu(input.imag, inplace=inplace))
    else:
        return F.relu(input, inplace=inplace)

def mod_relu(input: Tensor, bias: float = -math.sqrt(2), inplace: bool = False) -> Tensor:
    if input.is_complex():
        magnitude = torch.abs(input)
        euler_phase = torch.div(input, magnitude)
        if inplace:
            input = torch.mul(F.relu(magnitude + bias, inplace=False).type(input.type()), euler_phase)
            return input
        else:
            mod_relu = torch.mul(F.relu(magnitude + bias, inplace=inplace).type(input.type()), euler_phase)
            return mod_relu
    else:
        return F.relu(input, inplace=inplace)

def z_relu(input: Tensor, inplace: bool = False) -> Tensor:
    if input.is_complex():
        if inplace:
            mask = torch.zeros_like(input)
            input = torch.where(torch.angle(input) < 0, mask, input)
            input = torch.where(torch.angle(input) > (math.pi / 2), mask, input)
            return input
        else:
            mask = torch.zeros_like(input)
            z_relu = torch.where(torch.angle(input) < 0, mask, input)
            z_relu = torch.where(torch.angle(z_relu) > (math.pi / 2), mask, z_relu)
            return z_relu
    else:
        return F.relu(input, inplace=inplace)

def c_leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    if input.is_complex():
        return torch.complex(F.leaky_relu(input=input.real, negative_slope=negative_slope, inplace=inplace), \
                            F.leaky_relu(input=input.imag, negative_slope=negative_slope, inplace=inplace))
    else:
        return F.leaky_relu(input=input, negative_slope=negative_slope, inplace=inplace)

def mod_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        mag = torch.abs(input)
        input_ = torch.where(input.real < 0, -mag, mag)
        return F.softmax(input_, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    else:
        return F.softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def mod_log_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        mag = torch.abs(input)
        input_ = torch.where(input.real < 0, -mag, mag)
        return F.log_softmax(input_, dim=dim, _stacklevel=_stacklevel, dtype=dtype)
    else:
        return F.log_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def c_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        return torch.complex(F.softmax(input.real, dim=dim, _stacklevel=_stacklevel, dtype=dtype), F.softmax(input.imag, dim=dim, _stacklevel=_stacklevel, dtype=dtype))
    else:
        return F.softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)

def c_log_softmax(input: Tensor, dim: Optional[int] = None, _stacklevel: int = 3, dtype: Optional[int] = None) -> Tensor:
    if input.is_complex():
        return torch.complex(F.log_softmax(input.real, dim=dim, _stacklevel=_stacklevel, dtype=dtype), F.log_softmax(input.imag, dim=dim, _stacklevel=_stacklevel, dtype=dtype))
    else:
        return F.log_softmax(input, dim=dim, _stacklevel=_stacklevel, dtype=dtype)