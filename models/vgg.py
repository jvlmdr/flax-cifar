# Modified from: github.com/kuangliu/pytorch-cifar (models/vgg.py)

from functools import partial
from typing import Callable, Tuple

from flax import linen as nn
from jax import numpy as jnp

ModuleDef = Callable[..., nn.Module]


class Backbone(nn.Module):
    stages: Tuple[Tuple[int, ...], ...]
    norm: ModuleDef = nn.BatchNorm

    @nn.compact
    def __call__(self, x, norm_kwargs=None):
        norm_kwargs = norm_kwargs or {}
        for i, stage in enumerate(self.stages):
            for j, dim in enumerate(stage):
                suffix = '{:d}_{:d}'.format(i + 1, j + 1)
                x = nn.Conv(dim, (3, 3), padding=1, use_bias=False, name='conv' + suffix)(x)
                x = self.norm(name='norm' + suffix)(x, **norm_kwargs)
                x = nn.relu(x)
            x = nn.max_pool(x, (2, 2), strides=(2, 2))
        # x = nn.avg_pool(x, (1, 1), strides=1)
        return x


class MLP(nn.Module):
    dims: Tuple[int, ...]
    norm: ModuleDef = nn.BatchNorm

    @nn.compact
    def __call__(self, x, norm_kwargs=None):
        norm_kwargs = norm_kwargs or {}
        for dim in self.dims:
            x = nn.Dense(dim, use_bias=False)(x)
            x = self.norm()(x, **norm_kwargs)
            x = nn.relu(x)
        return x


class VGG(nn.Module):
    conv_stages: Tuple[Tuple[int, ...], ...]
    num_classes: int
    mlp_dims: Tuple[int, ...] = ()
    norm: ModuleDef = nn.BatchNorm

    def setup(self):
        self.backbone = Backbone(stages=self.conv_stages, norm=self.norm)
        if self.mlp_dims:
            self.mlp = MLP(self.mlp_dims, norm=self.norm)
        self.classifier = nn.Dense(self.num_classes)

    def __call__(self, x, norm_kwargs=None):
        norm_kwargs = norm_kwargs or {}
        x = self.backbone(x, norm_kwargs=norm_kwargs)
        x = jnp.reshape(x, (*x.shape[:-3], -1))
        if self.mlp_dims:
            x = self.mlp(x, norm_kwargs=norm_kwargs)
        x = self.classifier(x)
        return x


VGG11Backbone = partial(
    VGG, ((64,), (128,), (256, 256), (512, 512), (512, 512)))
VGG13Backbone = partial(
    VGG, ((64, 64), (128, 128), (256, 256), (512, 512), (512, 512)))
VGG16Backbone = partial(
    VGG, ((64, 64), (128, 128), (256, 256, 256), (512, 512, 512), (512, 512, 512)))
VGG19Backbone = partial(
    VGG, ((64, 64), (128, 128), (256, 256, 256, 256), (512, 512, 512, 512), (512, 512, 512, 512)))

VGG11 = partial(VGG11Backbone, mlp_dims=(4096, 4096))
VGG13 = partial(VGG13Backbone, mlp_dims=(4096, 4096))
VGG16 = partial(VGG16Backbone, mlp_dims=(4096, 4096))
VGG19 = partial(VGG19Backbone, mlp_dims=(4096, 4096))
