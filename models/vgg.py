from functools import partial
from typing import Callable, Tuple

from flax import linen as nn
from jax import numpy as jnp

ModuleDef = Callable[..., nn.Module]


class Backbone(nn.Module):
    stages: Tuple[Tuple[int, ...], ...]
    norm: ModuleDef = nn.BatchNorm

    @nn.compact
    def __call__(self, x, train: bool = True):
        for i, stage in enumerate(self.stages):
            for j, dim in enumerate(stage):
                suffix = '{:d}_{:d}'.format(i + 1, j + 1)
                x = nn.Conv(dim, (3, 3), padding=1, use_bias=False, name='conv' + suffix)(x)
                x = self.norm(use_running_average=not train, name='norm' + suffix)(x)
                x = nn.relu(x)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
        # x = nn.avg_pool(x, (1, 1), strides=1)
        return x


class MLP(nn.Module):
    dims: Tuple[int, ...]
    norm: ModuleDef = nn.BatchNorm

    @nn.compact
    def __call__(self, x, train: bool = True):
        for dim in self.dims:
            x = nn.Dense(dim, use_bias=False)(x)
            x = self.norm(use_running_average=not train)(x)
            x = nn.relu(x)
        return x


class VGG(nn.Module):
    conv_stages: Tuple[Tuple[int, ...], ...]
    mlp_dims: Tuple[int, ...]
    num_classes: int
    norm: ModuleDef = nn.BatchNorm

    def setup(self):
        self.backbone = Backbone(stages=self.conv_stages, norm=self.norm)
        if self.mlp_dims:
            self.mlp = MLP(self.mlp_dims, norm=self.norm)
        self.classifier = nn.Dense(self.num_classes)

    def __call__(self, x, train: bool = True):
        x = self.backbone(x, train)
        x = jnp.reshape(x, (x.shape[0], -1))
        if self.mlp_dims:
            x = self.mlp(x, train)
        x = self.classifier(x)
        return x


LiteVGG11 = partial(
    VGG, ((64,), (128,), (256, 256), (512, 512), (512, 512)), ())
LiteVGG13 = partial(
    VGG, ((64, 64), (128, 128), (256, 256), (512, 512), (512, 512)), ())
LiteVGG16 = partial(
    VGG, ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512)), ())
LiteVGG19 = partial(
    VGG, ((64, 64), (128, 128), (256, 256, 256, 256), (512, 512, 512, 512), (512, 512, 512, 512)), ())

VGG11 = partial(
    VGG, ((64,), (128,), (256, 256), (512, 512), (512, 512)), (4096, 4096))
VGG13 = partial(
    VGG, ((64, 64), (128, 128), (256, 256), (512, 512), (512, 512)), (4096, 4096))
VGG16 = partial(
    VGG, ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512)), (4096, 4096))
VGG19 = partial(
    VGG, ((64, 64), (128, 128), (256, 256, 256, 256), (512, 512, 512, 512), (512, 512, 512, 512)), (4096, 4096))
