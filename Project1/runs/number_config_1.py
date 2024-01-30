DATASET_FILE = 'dataset.csv'

GENERATION = {
    'cases':100,
}

GLOBAL_CONFIG = {
    'loss': "mse",
    'lrate': 0.01,
    'wrt': "L1",
    'wreg': 0.001,
    'epochs': 100,
    'batch_size': 10,
}

LAYER_CONFIG = {
    'input': 2500,
    'layers': [
    {
        'size': 20,
        'activation': 'relu',
        'weight_range': [-0.1, 0.1],
        'lrate': 0.01
    },
    {
        'size': 40,
        'activation': 'relu',
        'weight_range': [-0.1, 0.1],
        'lrate': 0.01
    },
    {
        'size': 9,
        'activation': 'sigmoid',
        'weight_range': [-0.1, 0.1],
        'lrate': 0.01
    },    
    ],
    'output_function': 'softmax',
}


