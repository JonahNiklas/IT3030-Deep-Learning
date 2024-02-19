DATASET_FILE = './Project1/core/doodle_cases.pkl'
RUN_NAME = "Wide"


GLOBAL_CONFIG = {
    'loss': "cross_entropy",
    'lrate': 0.1,
    'wrt': "L1",
    'wreg': 0.00001,
    'epochs': 20,
    'batch_size': 10,
}

LAYER_CONFIG = {
    'input': 2500,
    'layers': [
    {
        'size': 200,
        'activation': 'relu',
        'weight_range': [-1, 1],
        'lrate': 0.1
    },    
    {
        'size': 200,
        'activation': 'sigmoid',
        'weight_range': [-1, 1],
        'lrate': 0.1
    },
    {
        'size': 9,
        'activation': 'sigmoid',
        'weight_range': [-1, 1],
        'lrate': 0.1
    },    
    ],
    'output_function': 'softmax',
}


