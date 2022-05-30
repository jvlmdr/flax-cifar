import collections
from functools import partial
import sys
from typing import Tuple

from absl import app
from absl import flags
from flax import linen as nn
import jax
from jax import numpy as jnp
from jax import random
from jax import tree_util
import jaxopt
import ml_collections
from ml_collections import config_flags
import numpy as np
import optax
import torch.utils.data
from torchvision import datasets
from torchvision import transforms
import tqdm
import wandb

import models.densenet
import models.resnet_v1
import models.resnet_v2
import models.vgg
import models.wide_resnet
import util

flags.DEFINE_string('dataset_root', None, 'Path to data.', required=True)
flags.DEFINE_bool('download', False, 'Download dataset.')
flags.DEFINE_integer('eval_batch_size', 128, 'Batch size to use during evaluation.')
flags.DEFINE_integer('loader_num_workers', 4, 'num_workers for DataLoader')
flags.DEFINE_integer('loader_prefetch_factor', 2, 'prefetch_factor for DataLoader')
config_flags.DEFINE_config_file('config')

FLAGS = flags.FLAGS

Dataset = torch.utils.data.Dataset


def main(_):
    config = ml_collections.ConfigDict(FLAGS.config)

    wandb.init(project='flax-cifar')
    wandb.config.update(config.to_dict())

    num_classes, input_shape, train_dataset, val_dataset = setup_data()
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        pin_memory=False,
        num_workers=FLAGS.loader_num_workers,
        prefetch_factor=FLAGS.loader_prefetch_factor)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=FLAGS.eval_batch_size,
        shuffle=False,
        pin_memory=False,
        num_workers=FLAGS.loader_num_workers,
        prefetch_factor=FLAGS.loader_prefetch_factor)

    model = make_model(config, num_classes, input_shape)
    rng_init, _ = random.split(random.PRNGKey(0))
    init_vars = model.init(rng_init, jnp.zeros((1,) + input_shape))
    params, batch_stats = init_vars['params'], init_vars['batch_stats']

    print('params:')
    sys.stdout.writelines(x + '\n' for x in util.dict_tree_format(util.tree_shape(params)))
    print('batch_stats:')
    sys.stdout.writelines(x + '\n' for x in util.dict_tree_format(util.tree_shape(batch_stats)))
    sys.stdout.flush()

    def filter_kernel_params(tree):
        return [x for path, x in util.dict_tree_items(tree) if path[-1] == 'kernel']

    print('total number of params:',
          tree_util.tree_reduce(np.add, tree_util.tree_map(lambda x: np.prod(x.shape), params)))
    print('number of linear layers:', sum(1 for _ in filter_kernel_params(params)))

    total_steps = config.train.num_epochs * len(train_loader)
    schedule = optax.cosine_decay_schedule(config.train.base_learning_rate, total_steps)
    tx = optax.sgd(schedule, momentum=0.9)
    opt_state = tx.init(params)

    loss_with_logits = jax.vmap(jaxopt.loss.multiclass_logistic_loss)

    def objective_fn(params, mutable_vars, data):
        # Designed for use with jax.value_and_grad(..., has_aux=True).
        # Params are a separate arg (arg 0).
        # Returns scalar loss and one auxiliary output.
        inputs, labels = data
        model_vars = {'params': params, **mutable_vars}
        outputs, mutated_vars = model.apply(model_vars, inputs, train=True, mutable=list(mutable_vars.keys()))
        example_loss = loss_with_logits(labels, outputs)
        data_loss = jnp.mean(example_loss)
        if config.train.weight_decay_vars == 'all':
            wd_vars = list(tree_util.tree_leaves(params))
        elif config.train.weight_decay_vars == 'kernel':
            wd_vars = filter_kernel_params(params)
        else:
            raise ValueError('unknown variable collection', config.train.weight_decay_vars)
        wd_loss = 0.5 * sum(jnp.sum(jnp.square(x)) for x in wd_vars)
        objective = data_loss + config.train.weight_decay * wd_loss
        return objective, (outputs, mutated_vars)

    @jax.jit
    def train_step(opt_state, params, mutable_vars, data):
        objective_and_grad_fn = jax.value_and_grad(objective_fn, has_aux=True)
        (objective, aux), grads = objective_and_grad_fn(params, mutable_vars, data)
        outputs, mutated_vars = aux
        updates, opt_state = tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return opt_state, params, mutated_vars, objective, outputs

    @jax.jit
    def apply_model(params, batch_stats, inputs):
        return model.apply({'params': params, 'batch_stats': batch_stats}, inputs, train=False)

    for epoch in range(config.train.num_epochs + 1):
        metrics = {}

        if epoch > 0:
            train_outputs = collections.defaultdict(list)
            for inputs, labels in tqdm.tqdm(train_loader, f'train epoch {epoch}'):
                inputs, labels = jnp.asarray(inputs.numpy()), jnp.asarray(labels.numpy())
                inputs = jnp.moveaxis(inputs, -3, -1)
                opt_state, params, mutated_vars, objective, logits = train_step(
                    opt_state, params, {'batch_stats': batch_stats}, (inputs, labels))
                batch_stats = mutated_vars['batch_stats']
                loss = loss_with_logits(labels, logits)
                pred = jnp.argmax(logits, axis=-1)
                acc = (pred == labels)
                train_outputs['acc'].append(acc)
                train_outputs['loss'].append(loss)
                train_outputs['objective'].append([objective])
            train_outputs = {k: np.concatenate(v) for k, v in train_outputs.items()}
            metrics.update({
                'train_loss': np.mean(train_outputs['loss']),
                'train_acc': np.mean(train_outputs['acc']),
                'train_objective': np.mean(train_outputs['objective']),
            })

        val_outputs = collections.defaultdict(list)
        for inputs, labels in tqdm.tqdm(val_loader, f'val epoch {epoch}'):
            inputs, labels = jnp.asarray(inputs.numpy()), jnp.asarray(labels.numpy())
            inputs = jnp.moveaxis(inputs, -3, -1)
            logits = apply_model(params, batch_stats, inputs)
            loss = loss_with_logits(labels, logits)
            pred = jnp.argmax(logits, axis=-1)
            acc = (pred == labels)
            val_outputs['acc'].append(acc)
            val_outputs['loss'].append(loss)
        val_outputs = {k: np.concatenate(v) for k, v in val_outputs.items()}
        metrics.update({
            'val_loss': np.mean(val_outputs['loss']),
            'val_acc': np.mean(val_outputs['acc']),
        })

        wandb.log(metrics)
        if epoch == 0:
            print('epoch {:d}: val_acc {:.2%}'.format(epoch, metrics['val_acc']))
        else:
            print('epoch {:d}: val_acc {:.2%}, train_objective {:.6g}'.format(
                epoch, metrics['val_acc'], metrics['train_objective']))


def setup_data() -> Tuple[int, Tuple[int, int, int], Dataset, Dataset]:
    num_classes = 10
    input_shape = (32, 32, 3)
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1.0),
    ])
    transform_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.5, 1.0),
    ])
    train_dataset = datasets.CIFAR10(
        FLAGS.dataset_root, train=True, download=FLAGS.download, transform=transform_train)
    val_dataset = datasets.CIFAR10(
        FLAGS.dataset_root, train=False, download=FLAGS.download, transform=transform_eval)
    return num_classes, input_shape, train_dataset, val_dataset


def make_model(
        config: ml_collections.ConfigDict,
        num_classes: int,
        input_shape: Tuple[int, int, int]) -> nn.Module:
    try:
        model_fn = {
            'resnet_v1_18': partial(models.resnet_v1.ResNet18, stem_variant='cifar'),
            'resnet_v1_34': partial(models.resnet_v1.ResNet34, stem_variant='cifar'),
            'resnet_v1_50': partial(models.resnet_v1.ResNet50, stem_variant='cifar'),
            'resnet_v2_18': partial(models.resnet_v2.ResNet18, stem_variant='cifar'),
            'resnet_v2_34': partial(models.resnet_v2.ResNet34, stem_variant='cifar'),
            'resnet_v2_50': partial(models.resnet_v2.ResNet50, stem_variant='cifar'),
            'wrn28_2': partial(models.wide_resnet.WideResNet, depth=28, width=2),
            'wrn28_8': partial(models.wide_resnet.WideResNet, depth=28, width=8),
            'densenet121_12': models.densenet.densenet_cifar,
            'densenet121_32': models.densenet.DenseNet121,
            'densenet169_32': models.densenet.DenseNet169,
            'densenet201_32': models.densenet.DenseNet201,
            'densenet161_48': models.densenet.DenseNet161,
            'vgg11_lite': models.vgg.LiteVGG11,
            'vgg13_lite': models.vgg.LiteVGG13,
            'vgg16_lite': models.vgg.LiteVGG16,
            'vgg19_lite': models.vgg.LiteVGG19,
            'vgg11': models.vgg.VGG11,
            'vgg13': models.vgg.VGG13,
            'vgg16': models.vgg.VGG16,
            'vgg19': models.vgg.VGG19,
        }[config.arch]
    except KeyError as ex:
        raise ValueError('unknown architecture', ex)
    return  model_fn(num_classes=num_classes)


if __name__ == '__main__':
    app.run(main)
