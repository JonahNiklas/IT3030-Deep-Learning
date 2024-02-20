DATASET_FILE = './Project1/core/doodle_cases.pkl'
RUN_NAME = "Deep"

GLOBAL_CONFIG = {
    'loss': "cross_entropy",
    'lrate': 0.1,
    'wrt': None,
    'wreg': 0.00001,
    'epochs': 50,
    'batch_size': 10,
}

LAYER_CONFIG = {
    'input': 2500,
    'layers': [
    {
        'size': 10,
        'activation': 'relu',
        'weight_range': [-0.5, 0.5],
        'lrate': 0.1
    },  
    {
        'size': 20,
        'activation': 'relu',
        'weight_range': [-0.5, 0.5],
        'lrate': 0.1
    },  
    {
        'size': 50,
        'activation': 'relu',
        'weight_range': [-0.5, 0.5],
        'lrate': 0.1
    },    
    {
        'size': 100,
        'activation': 'relu',
        'weight_range': [-0.5, 0.5],
        'lrate': 0.1
    },
    {
        'size': 20,
        'activation': 'relu',
        'weight_range': [-0.5, 0.5],
        'lrate': 0.1
    },  
    {
        'size': 9,
        'activation': 'sigmoid',
        'weight_range': [-0.5, 0.5],
        'lrate': 0.1
    },    
    ],
    'output_function': 'softmax',
}


