# Modified from: github.com/kuangliu/pytorch-cifar (models/densenet.py)

'''DenseNet in PyTorch.'''

from functools import partial
from typing import Callable, Sequence

from flax import linen as nn
from jax import numpy as jnp

ModuleDef = Callable[..., nn.Module]


class Bottleneck(nn.Module):
    growth_rate: int
    norm: ModuleDef = nn.BatchNorm

    def setup(self):
        self.bn1 = self.norm()
        self.conv1 = nn.Conv(4 * self.growth_rate, (1, 1), use_bias=False)
        self.bn2 = self.norm()
        self.conv2 = nn.Conv(self.growth_rate, (3, 3), padding=(1, 1), use_bias=False)

    def __call__(self, x, norm_kwargs=None):
        norm_kwargs = norm_kwargs or {}
        out = self.conv1(nn.relu(self.bn1(x, **norm_kwargs)))
        out = self.conv2(nn.relu(self.bn2(out, **norm_kwargs)))
        out = jnp.concatenate([out, x], -1)
        return out


class Transition(nn.Module):
    out_planes: int
    norm: ModuleDef = nn.BatchNorm

    def setup(self):
        self.bn = self.norm()
        self.conv = nn.Conv(self.out_planes, (1, 1), use_bias=False)

    def __call__(self, x, norm_kwargs=None):
        norm_kwargs = norm_kwargs or {}
        out = self.conv(nn.relu(self.bn(x, **norm_kwargs)))
        out = nn.avg_pool(out, (2, 2), strides=(2, 2), padding='VALID')
        return out


class BlockSequence(nn.Module):
    block: ModuleDef
    growth_rate: int
    nblock: int
    norm: ModuleDef = nn.BatchNorm

    @nn.compact
    def __call__(self, x, norm_kwargs=None):
        for i in range(self.nblock):
            block = self.block(self.growth_rate, norm=self.norm, name='block{:d}'.format(i + 1))
            x = block(x, norm_kwargs=norm_kwargs)
        return x


class DenseNet(nn.Module):
    block: ModuleDef
    nblocks: Sequence[int]
    num_classes: int
    growth_rate: int
    reduction: float = 0.5
    norm: ModuleDef = nn.BatchNorm

    def setup(self):
        make_dense_layers = partial(BlockSequence, self.block, self.growth_rate, norm=self.norm)

        num_planes = 2 * self.growth_rate
        self.conv1 = nn.Conv(num_planes, (3, 3), padding=(1, 1), use_bias=False)

        self.dense1 = make_dense_layers(self.nblocks[0])
        num_planes += self.nblocks[0] * self.growth_rate
        out_planes = int(num_planes * self.reduction)
        self.trans1 = Transition(out_planes, norm=self.norm)
        num_planes = out_planes

        self.dense2 = make_dense_layers(self.nblocks[1])
        num_planes += self.nblocks[1] * self.growth_rate
        out_planes = int(num_planes * self.reduction)
        self.trans2 = Transition(out_planes, norm=self.norm)
        num_planes = out_planes

        self.dense3 = make_dense_layers(self.nblocks[2])
        num_planes += self.nblocks[2] * self.growth_rate
        out_planes = int(num_planes * self.reduction)
        self.trans3 = Transition(out_planes, norm=self.norm)
        num_planes = out_planes

        self.dense4 = make_dense_layers(self.nblocks[3])
        num_planes += self.nblocks[3] * self.growth_rate

        self.bn = self.norm()
        self.linear = nn.Dense(self.num_classes)

    def __call__(self, x, norm_kwargs=None):
        norm_kwargs = norm_kwargs or {}
        out = self.conv1(x)
        out = self.trans1(self.dense1(out, norm_kwargs=norm_kwargs), norm_kwargs=norm_kwargs)
        out = self.trans2(self.dense2(out, norm_kwargs=norm_kwargs), norm_kwargs=norm_kwargs)
        out = self.trans3(self.dense3(out, norm_kwargs=norm_kwargs), norm_kwargs=norm_kwargs)
        out = self.dense4(out, norm_kwargs=norm_kwargs)
        out = nn.relu(self.bn(out, **norm_kwargs))
        out = nn.avg_pool(out, (4, 4), strides=(4, 4), padding='VALID')
        out = jnp.reshape(out, (*out.shape[:-3], -1))
        out = self.linear(out)
        return out


DenseNet121 = partial(DenseNet, Bottleneck, (6, 12, 24, 16), growth_rate=32)
DenseNet169 = partial(DenseNet, Bottleneck, (6, 12, 32, 32), growth_rate=32)
DenseNet201 = partial(DenseNet, Bottleneck, (6, 12, 48, 32), growth_rate=32)
DenseNet161 = partial(DenseNet, Bottleneck, (6, 12, 36, 24), growth_rate=48)
densenet_cifar = partial(DenseNet, Bottleneck, (6, 12, 24, 16), growth_rate=12)
