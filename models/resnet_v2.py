# Based on: objax.zoo.resnet_v2
# Modified to use flax instead of objax.
#
# With CIFAR variant from: https://github.com/kuangliu/pytorch-cifar

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

"""Module with ResNet-v2 implementation.

See https://arxiv.org/abs/1603.05027 for detail.
"""

from functools import partial
from typing import Callable, Sequence, Union

from flax import linen as nn
from jax import numpy as jnp

__all__ = ['ResNetV2', 'ResNet18', 'ResNet34', 'ResNet50', 'ResNet101', 'ResNet152', 'ResNet200']

BN_MOM = 0.9
BN_EPS = 1e-5


def conv_args(kernel_size: int,
              nout: int,
              padding: nn.linear.PaddingLike = 'VALID'):
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
                padding=padding)


class ResNetV2Block(nn.Module):
    """ResNet v2 block with optional bottleneck.

    Args:
        nout: number of output filters.
        stride: stride for 3x3 convolution and projection convolution in this block.
        use_projection: if True then include projection convolution into this block.
        bottleneck: if True then make bottleneck block.
        normalization_fn: module which used as normalization function.
        activation_fn: activation function.
    """
    nout: int
    stride: Union[int, Sequence[int]]
    use_projection: bool
    bottleneck: bool
    normalization_fn: Callable[..., nn.Module] = partial(nn.BatchNorm, momentum=BN_MOM, epsilon=BN_EPS)
    activation_fn: Callable = nn.relu

    def setup(self):
        """Creates ResNetV2Block instance."""
        if self.use_projection:
            self.proj_conv = nn.Conv(self.nout, (1, 1), strides=self.stride, **conv_args(1, self.nout))

        if self.bottleneck:
            self.norm_0 = self.normalization_fn()
            self.conv_0 = nn.Conv(self.nout // 4, (1, 1), strides=1, **conv_args(1, self.nout // 4))
            self.norm_1 = self.normalization_fn()
            self.conv_1 = nn.Conv(self.nout // 4, (3, 3), strides=self.stride, **conv_args(3, self.nout // 4, (1, 1)))
            self.norm_2 = self.normalization_fn()
            self.conv_2 = nn.Conv(self.nout, (1, 1), strides=1, **conv_args(1, self.nout))
        else:
            self.norm_0 = self.normalization_fn()
            self.conv_0 = nn.Conv(self.nout, (3, 3), strides=1, **conv_args(3, self.nout, (1, 1)))
            self.norm_1 = self.normalization_fn()
            self.conv_1 = nn.Conv(self.nout, (3, 3), strides=self.stride, **conv_args(3, self.nout, (1, 1)))

    def __call__(self, x, train: bool):
        if self.stride > 1:
            shortcut = nn.max_pool(x, (1, 1), strides=(self.stride, self.stride))
        else:
            shortcut = x

        if self.bottleneck:
            layers = ((self.norm_0, self.conv_0), (self.norm_1, self.conv_1), (self.norm_2, self.conv_2))
        else:
            layers = ((self.norm_0, self.conv_0), (self.norm_1, self.conv_1))

        for i, (bn_i, conv_i) in enumerate(layers):
            x = bn_i(x, use_running_average=not train)
            x = self.activation_fn(x)
            if i == 0 and self.use_projection:
                shortcut = self.proj_conv(x)
            x = conv_i(x)

        return x + shortcut


class ResNetV2BlockGroup(nn.Module):
    """Group of ResNet v2 Blocks.

    Args:
        nout: number of output filters.
        num_blocks: number of Resnet blocks in this group.
        stride: stride for 3x3 convolutions and projection convolutions in Resnet blocks.
        use_projection: if True then include projection convolution into each Resnet blocks.
        bottleneck: if True then make bottleneck blocks.
        normalization_fn: module which used as normalization function.
        activation_fn: activation function.
    """
    nout: int
    num_blocks: int
    stride: Union[int, Sequence[int]]
    use_projection: bool
    bottleneck: bool
    normalization_fn: Callable[..., nn.Module] = partial(nn.BatchNorm, momentum=BN_MOM, epsilon=BN_EPS)
    activation_fn: Callable = nn.relu

    def setup(self):
        """Creates ResNetV2BlockGroup instance."""
        blocks = []
        for i in range(self.num_blocks):
            blocks.append(
                ResNetV2Block(
                    nout=self.nout,
                    stride=(self.stride if i == self.num_blocks - 1 else 1),
                    use_projection=(i == 0 and self.use_projection),
                    bottleneck=self.bottleneck,
                    normalization_fn=self.normalization_fn,
                    activation_fn=self.activation_fn))
        self.blocks = blocks

    def __call__(self, x, train: bool):
        # Cannot use nn.Sequential because must pass train.
        # Could use @nn.compact, but it may be convenient to have blocks later.
        for block in self.blocks:
            x = block(x, train)
        return x


class ResNetV2(nn.Module):
    """Base implementation of ResNet v2 from https://arxiv.org/abs/1603.05027.

    Args:
        in_channels: number of channels in the input image.
        num_classes: number of output classes.
        blocks_per_group: number of blocks in each block group.
        bottleneck: if True then use bottleneck blocks.
        channels_per_group: number of output channels for each block group.
        group_strides: strides for each block group.
        normalization_fn: module which used as normalization function.
        activation_fn: activation function.
        stem_variant: 'imagenet' or 'cifar'
    """
    num_classes: int
    blocks_per_group: Sequence[int]
    bottleneck: bool = True
    channels_per_group: Sequence[int] = (256, 512, 1024, 2048)
    group_strides: Sequence[int] = (1, 2, 2, 2)
    group_use_projection: Sequence[bool] = (True, True, True, True)
    normalization_fn: Callable[..., nn.Module] = partial(nn.BatchNorm, momentum=BN_MOM, epsilon=BN_EPS)
    activation_fn: Callable = nn.relu
    stem_variant: str = 'cifar'

    @nn.compact
    def __call__(self, x, train: bool = True):
        """Creates ResNetV2 instance."""
        assert len(self.channels_per_group) == len(self.blocks_per_group)
        assert len(self.group_strides) == len(self.blocks_per_group)
        assert len(self.group_use_projection) == len(self.blocks_per_group)
        assert self.stem_variant in ('imagenet', 'cifar')

        if self.stem_variant == 'cifar':
            # 3x3 conv with stride 1
            x = nn.Conv(64, (3, 3), strides=1, **conv_args(3, 64, (1, 1)))(x)
        else:
            # 7x7 conv with stride 2, 3x3 max pool with stride 2
            x = nn.Conv(64, (7, 7), strides=2, **conv_args(7, 64, (3, 3)))(x)
            x = jnp.pad(x, pad_width=((0, 0), (1, 1), (1, 1), (0, 0)))
            x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='VALID')

        for i in range(len(self.blocks_per_group)):
            x = ResNetV2BlockGroup(
                nout=self.channels_per_group[i],
                num_blocks=self.blocks_per_group[i],
                stride=self.group_strides[i],
                bottleneck=self.bottleneck,
                use_projection=self.group_use_projection[i],
                normalization_fn=self.normalization_fn,
                activation_fn=self.activation_fn,
            )(x, train)

        x = self.normalization_fn()(x, use_running_average=not train)
        x = self.activation_fn(x)
        x = jnp.mean(x, axis=(-3, -2))
        x = nn.Dense(self.num_classes, kernel_init=nn.initializers.glorot_normal())(x)
        return x


ResNet18 = partial(ResNetV2,
                   blocks_per_group=(2, 2, 2, 2),
                   bottleneck=False,
                   channels_per_group=(64, 128, 256, 512),
                   group_use_projection=(False, True, True, True))

ResNet34 = partial(ResNetV2,
                   blocks_per_group=(3, 4, 6, 3),
                   bottleneck=False,
                   channels_per_group=(64, 128, 256, 512),
                   group_use_projection=(False, True, True, True))

ResNet50 = partial(ResNetV2,
                   blocks_per_group=(3, 4, 6, 3),
                   group_strides=(2, 2, 2, 1),
                   bottleneck=True)

ResNet101 = partial(ResNetV2,
                    blocks_per_group=(3, 4, 23, 3),
                    group_strides=(2, 2, 2, 1),
                    bottleneck=True)

ResNet152 = partial(ResNetV2,
                    blocks_per_group=(3, 8, 36, 3),
                    group_strides=(2, 2, 2, 1),
                    bottleneck=True)

ResNet200 = partial(ResNetV2,
                    blocks_per_group=(3, 24, 36, 3),
                    group_strides=(2, 2, 2, 1),
                    bottleneck=True)
