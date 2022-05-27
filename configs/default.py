def get_config():
    return {
        'model': {
            'arch': 'wide_resnet',
            'resnet': {
                'stem_variant': 'cifar',
            },
            'wrn': {
                'depth': 28,
                'width': 2,
            },
        },
        'train': {
            'base_learning_rate': 0.1,
            'num_epochs': 50,
            'batch_size': 128,
            'weight_decay': 0.0003,
            'weight_decay_vars': 'all',
        },
    }
