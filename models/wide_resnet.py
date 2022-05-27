# Based on: objax.zoo.wide_resnet
# Modified to use flax instead of objax.

# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Module with WideResNet implementation.

See https://arxiv.org/abs/1605.07146 for detail.
"""

__all__ = ['WRNBlock', 'WideResNetGeneral', 'WideResNet']

from functools import partial
from typing import Callable, Tuple

from flax import linen as nn
from jax import numpy as jnp

BN_MOM = 0.9
BN_EPS = 1e-5


def conv_args(kernel_size: int, nout: int):
    """Returns list of arguments which are common to all convolutions.

    Args:
        kernel_size: size of convolution kernel (single number).
        nout: number of output filters.

    Returns:
        Dictionary with common convoltion arguments.
    """
    stddev = (0.5 * kernel_size * kernel_size * nout) ** -0.5
    return dict(kernel_init=nn.initializers.normal(stddev),
                use_bias=False,
                padding='SAME')


class WRNBlock(nn.Module):
    """WideResNet block."""
    nin: int
    nout: int
    stride: int = 1
    bn: Callable = partial(nn.BatchNorm, momentum=BN_MOM, epsilon=BN_EPS)

    def setup(self):
        if self.nin != self.nout or self.stride > 1:
            self.proj_conv = nn.Conv(self.nout, (1, 1), strides=self.stride, **conv_args(1, self.nout))
        else:
            self.proj_conv = None

        self.norm_1 = self.bn()
        self.conv_1 = nn.Conv(self.nout, (3, 3), strides=self.stride, **conv_args(3, self.nout))
        self.norm_2 = self.bn()
        self.conv_2 = nn.Conv(self.nout, (3, 3), strides=1, **conv_args(3, self.nout))

    def __call__(self, x, train: bool):
        o1 = nn.relu(self.norm_1(x, use_running_average=not train))
        y = self.conv_1(o1)
        o2 = nn.relu(self.norm_2(y, use_running_average=not train))
        z = self.conv_2(o2)
        return z + self.proj_conv(o1) if self.proj_conv else z + x


class WideResNetGeneral(nn.Module):
    """Base WideResNet implementation."""
    nclass: int
    blocks_per_group: Tuple[int]
    width: int
    bn: Callable = partial(nn.BatchNorm, momentum=BN_MOM, epsilon=BN_EPS)

    @nn.compact
    def __call__(self, x, train: bool = True):
        widths = [int(v * self.width) for v in [16 * (2 ** i) for i in range(len(self.blocks_per_group))]]
        n = 16
        x = nn.Conv(n, (3, 3), **conv_args(3, n))(x)
        for i, (block, width) in enumerate(zip(self.blocks_per_group, widths)):
            stride = 2 if i > 0 else 1
            x = WRNBlock(n, width, stride, self.bn)(x, train)
            for b in range(1, block):
                x = WRNBlock(width, width, 1, self.bn)(x, train)
            n = width
        x = self.bn()(x, use_running_average=not train)
        x = nn.relu(x)
        x = jnp.mean(x, axis=(-3, -2))
        x = nn.Dense(self.nclass, kernel_init=nn.initializers.glorot_normal())(x)
        return x


def WideResNet(
        num_classes: int,
        depth: int = 28,
        width: int = 2,
        bn: Callable = partial(nn.BatchNorm, momentum=BN_MOM, epsilon=BN_EPS)):
    """Creates WideResNet instance.

    Args:
        num_classes: number of output classes.
        depth: number of convolution layers. (depth-4) should be divisible by 6
        width: multiplier to the number of convolution filters.
        bn: module which used as batch norm function.
    """
    assert (depth - 4) % 6 == 0, 'depth should be 6n+4'
    n = (depth - 4) // 6
    blocks_per_group = (n,) * 3
    return WideResNetGeneral(num_classes, blocks_per_group, width, bn)
