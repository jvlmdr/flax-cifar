This is intended to be a flax version of the [pytorch-cifar repo](https://github.com/kuangliu/pytorch-cifar) that I've often used as a reference.

The ResNet models have 'imagenet' and 'cifar' variants, which are intended for 224x224 and 32x32 images respectively.


### Credit

The WideResNet and ResNetV2 implementations were based on the [objax.zoo package](https://objax.readthedocs.io/en/latest/objax/zoo.html).

The ResNetV1 implementation was based on the [flax imagenet example](https://github.com/google/flax/tree/main/examples/imagenet).

The DenseNet and VGG implementations were based on the [pytorch-cifar repo](https://github.com/kuangliu/pytorch-cifar).


### Results

All results using default config (see `configs/default.py`).

Refer to the [wandb project](https://wandb.ai/jvlmdr/flax-cifar).

| Model | Params | Acc. |
| :--   |    --: |  --: |
| VGG-11 backbone | 9,228,362 | 90.6% |
| VGG-13 backbone | 9,413,066 | 92.7% |
| VGG-16 backbone | 14,724,042 | 92.1% |
| VGG-19 backbone | 20,035,018 | 92.3% |
| VGG-11 | 28,154,954 | 89.7% |
| VGG-13 | 28,339,658 | 91.9% |
| VGG-16 | 33,650,634 | 92.2% |
| VGG-19 | 38,961,610 | 90.3% |
| ResNetV1-18 | 11,173,962 | 94.1% |
| ResNetV1-50 | 23,520,842 | 94.0% |
| ResNetV2-18 | 11,172,170 | 94.7% |
| ResNetV2-50 | 23,513,162 | 93.8% |
| WideResNet-28-2 | 1,467,610 | 93.3% |
| WideResNet-28-8 | 23,354,842 | 95.1% |
| DenseNet121-12 | 1,000,618 | 93.6% |
| DenseNet121-32 | 6,956,298 | 94.8% |
| DenseNet169-32 | 12,493,322 | 94.2% |


### Usage

```bash
python main.py --dataset_root=path/to/cifar10 --config=configs/default.py --config.model.arch=resnet_v1_18
```


### Dependencies

```bash
absl-py
flax
jax[cuda]
ml_collections
numpy 
torch
torchvision
tqdm
wandb
```
