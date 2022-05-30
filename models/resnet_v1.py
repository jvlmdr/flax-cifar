# Based on:
# https://github.com/google/flax/blob/43b358c/examples/imagenet/models.py
#
# With CIFAR variant from: https://github.com/kuangliu/pytorch-cifar

# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Flax implementation of ResNet V1."""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Callable, Sequence, Tuple

from flax import linen as nn
import jax.numpy as jnp

ModuleDef = Any


class ResNetBlock(nn.Module):
    """ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef  # May already have `use_running_average` set.
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (3, 3), self.strides)(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class BottleneckResNetBlock(nn.Module):
    """Bottleneck ResNet block."""
    filters: int
    conv: ModuleDef
    norm: ModuleDef  # May already have `use_running_average` set.
    act: Callable
    strides: Tuple[int, int] = (1, 1)

    @nn.compact
    def __call__(self, x):
        residual = x
        y = self.conv(self.filters, (1, 1))(x)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters, (3, 3), self.strides)(y)
        y = self.norm()(y)
        y = self.act(y)
        y = self.conv(self.filters * 4, (1, 1))(y)
        y = self.norm(scale_init=nn.initializers.zeros)(y)

        if residual.shape != y.shape:
            residual = self.conv(self.filters * 4, (1, 1),
                                 self.strides, name='conv_proj')(residual)
            residual = self.norm(name='norm_proj')(residual)

        return self.act(residual + y)


class ResNet(nn.Module):
    """ResNetV1."""
    stage_sizes: Sequence[int]
    block_cls: ModuleDef
    num_classes: int
    num_filters: int = 64
    dtype: Any = jnp.float32
    act: Callable = nn.relu
    conv: ModuleDef = nn.Conv
    norm: ModuleDef = nn.BatchNorm  # Must support `use_running_average`.
    stem_variant: bool = False

    @nn.compact
    def __call__(self, x, train: bool = True):
        assert self.stem_variant in ('imagenet', 'cifar')
        conv = partial(self.conv, use_bias=False, dtype=self.dtype)
        norm = partial(self.norm, use_running_average=not train)

        if self.stem_variant == 'cifar':
            # 3x3 conv with stride 1
            x = conv(self.num_filters, (3, 3), (1, 1), padding=[(1, 1), (1, 1)], name='conv_init')(x)
            x = norm(name='bn_init')(x)
            x = nn.relu(x)
        else:
            # 7x7 conv with stride 2, 3x3 max pool with stride 2
            x = conv(self.num_filters, (7, 7), (2, 2), padding=[(3, 3), (3, 3)], name='conv_init')(x)
            x = norm(name='bn_init')(x)
            x = nn.relu(x)
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

        for i, block_size in enumerate(self.stage_sizes):
            for j in range(block_size):
                strides = (2, 2) if i > 0 and j == 0 else (1, 1)
                x = self.block_cls(self.num_filters * 2 ** i,
                                   strides=strides,
                                   conv=conv,
                                   norm=norm,
                                   act=self.act)(x)
        x = jnp.mean(x, axis=(1, 2))
        x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
        x = jnp.asarray(x, self.dtype)
        return x


ResNet18 = partial(ResNet, [2, 2, 2, 2], ResNetBlock)
ResNet34 = partial(ResNet, [3, 4, 6, 3], ResNetBlock)
ResNet50 = partial(ResNet, [3, 4, 6, 3], BottleneckResNetBlock)
ResNet101 = partial(ResNet, [3, 4, 23, 3], BottleneckResNetBlock)
ResNet152 = partial(ResNet, [3, 8, 36, 3], BottleneckResNetBlock)
ResNet200 = partial(ResNet, [3, 24, 36, 3], BottleneckResNetBlock)

ResNet18Local = partial(ResNet, [2, 2, 2, 2], ResNetBlock, conv=nn.ConvLocal)

# Used for testing only.
_ResNet1 = partial(ResNet, [1], ResNetBlock)
_ResNet1Local = partial(ResNet, [1], ResNetBlock, conv=nn.ConvLocal)
