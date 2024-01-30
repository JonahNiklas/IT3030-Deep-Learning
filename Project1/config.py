DATASET_FILE = 'dataset.csv'

GLOBAL_CONFIG = {
    'loss': "mse",
    'lrate': 0.01,
    'wrt': "L1",
    'wreg': 0.001,
    'epochs': 100,
}

LAYER_CONFIG = {
    'input': 784,
    'layers': [
    {
        'size': 100,
        'activation': 'sigmoid',
        'weight_range': [-0.1, 0.1],
        'lrate': 0.01
    },
    {
        'size': 100,
        'activation': 'sigmoid',
        'weight_range': [-0.1, 0.1],
        'lrate': 0.01
    },
    {
        'size': 100,
        'activation': 'sigmoid',
        'weight_range': [-0.1, 0.1],
        'lrate': 0.01
    },
    
    ],
    'output_function': 'softmax',
}


