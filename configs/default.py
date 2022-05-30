def get_config():
    return {
        'arch': 'wrn28_2',
        'train': {
            'base_learning_rate': 0.1,
            'num_epochs': 50,
            'batch_size': 128,
            'weight_decay': 0.0003,
            'weight_decay_vars': 'all',
        },
    }
