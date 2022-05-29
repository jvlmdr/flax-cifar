# Modified from: github.com/kuangliu/pytorch-cifar (models/densenet.py)

'''DenseNet in PyTorch.'''

from functools import partial
from typing import Callable, Sequence

from flax import linen as nn
from jax import numpy as jnp

ModuleDef = Callable[..., nn.Module]


class Bottleneck(nn.Module):
    growth_rate: int

    def setup(self):
        self.bn1 = nn.BatchNorm()
        self.conv1 = nn.Conv(4 * self.growth_rate, (1, 1), use_bias=False)
        self.bn2 = nn.BatchNorm()
        self.conv2 = nn.Conv(self.growth_rate, (3, 3), padding=(1, 1), use_bias=False)

    def __call__(self, x, train: bool):
        out = self.conv1(nn.relu(self.bn1(x, use_running_average=not train)))
        out = self.conv2(nn.relu(self.bn2(out, use_running_average=not train)))
        out = jnp.concatenate([out, x], -1)
        return out


class Transition(nn.Module):
    out_planes: int

    def setup(self):
        self.bn = nn.BatchNorm()
        self.conv = nn.Conv(self.out_planes, (1, 1), use_bias=False)

    def __call__(self, x, train: bool):
        out = self.conv(nn.relu(self.bn(x, use_running_average=not train)))
        out = nn.avg_pool(out, (2, 2), strides=(2, 2), padding='VALID')
        return out


class Sequential(nn.Module):
    layers: Sequence[ModuleDef]

    def __call__(self, x, *args, **kwargs):
        for layer in self.layers:
            x = layer(x, *args, **kwargs)
        return x


class DenseNet(nn.Module):
    block: ModuleDef
    nblocks: int
    num_classes: int
    growth_rate: int
    reduction: float = 0.5

    def setup(self):
        num_planes = 2 * self.growth_rate
        self.conv1 = nn.Conv(num_planes, (3, 3), padding=(1, 1), use_bias=False)

        self.dense1 = self._make_dense_layers(self.block, self.nblocks[0])
        num_planes += self.nblocks[0] * self.growth_rate
        out_planes = int(num_planes * self.reduction)
        self.trans1 = Transition(out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(self.block, self.nblocks[1])
        num_planes += self.nblocks[1] * self.growth_rate
        out_planes = int(num_planes * self.reduction)
        self.trans2 = Transition(out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(self.block, self.nblocks[2])
        num_planes += self.nblocks[2] * self.growth_rate
        out_planes = int(num_planes * self.reduction)
        self.trans3 = Transition(out_planes)
        num_planes = out_planes

        self.dense4 = self._make_dense_layers(self.block, self.nblocks[3])
        num_planes += self.nblocks[3] * self.growth_rate

        self.bn = nn.BatchNorm()
        self.linear = nn.Dense(self.num_classes)

    @nn.nowrap
    def _make_dense_layers(self, block, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(self.growth_rate))
        return Sequential(layers)

    def __call__(self, x, train: bool = True):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out, train), train)
        out = self.trans2(self.dense2(out, train), train)
        out = self.trans3(self.dense3(out, train), train)
        out = self.dense4(out, train)
        out = nn.relu(self.bn(out, use_running_average=not train))
        out = nn.avg_pool(out, (4, 4), strides=(4, 4), padding='VALID')
        out = jnp.reshape(out, (out.shape[0], -1))
        out = self.linear(out)
        return out


DenseNet121 = partial(DenseNet, Bottleneck, [6, 12, 24, 16], growth_rate=32)
DenseNet169 = partial(DenseNet, Bottleneck, [6, 12, 32, 32], growth_rate=32)
DenseNet201 = partial(DenseNet, Bottleneck, [6, 12, 48, 32], growth_rate=32)
DenseNet161 = partial(DenseNet, Bottleneck, [6, 12, 36, 24], growth_rate=48)
densenet_cifar = partial(DenseNet, Bottleneck, [6, 12, 24, 16], growth_rate=12)
