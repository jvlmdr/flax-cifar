This is intended to be a flax version of the [pytorch-cifar repo](https://github.com/kuangliu/pytorch-cifar) that I've often used as a reference.

The ResNet models have 'imagenet' and 'cifar' variants, which are intended for 224x224 and 32x32 images respectively.


### Credit

The WideResNet and ResNetV2 implementations were based on the [objax.zoo package](https://objax.readthedocs.io/en/latest/objax/zoo.html).

The ResNetV1 implementation was based on the [flax imagenet example](https://github.com/google/flax/tree/main/examples/imagenet).

These were generally modified as little as possible.


### Results

All results using default config (see `configs/default.py`).

Refer to the [wandb project](https://wandb.ai/jvlmdr/flax-cifar).

| Model | Params | Acc. |
| :--   |    --: |  --: |
| ResNetV1-18 | 11,173,962 | 94.1% |
| ResNetV1-50 | 23,520,842 | 94.3% |
| ResNetV2-18 | 11,172,170 | 94.6% |
| ResNetV2-50 | 23,513,162 | 93.6% |
| WideResNet-28-2 | 1,467,610 | 93.2% |
| WideResNet-28-8 | 23,354,842 | 95.0% |
| DenseNet | 1,000,618 | 93.6% |


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
