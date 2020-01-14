from copy import deepcopy

default_sstage_conf = {
    'dataset': 'default',
    'types': ['copter','sphero','youbot'],
    # converged after 15 / 10 / 5 epochs
    # 10 epochs after convergance for evaluation phase
    'epochs_cat': 25,
    #'epochs_reg': 20,
    'epochs_reg': 200,
    'epochs_bin': 15,
    'optimizer': 'adam',
    'learning_rate': 3e-4,
    'dropout': 0,
    'alpha': 0.5,
    'img_size': 35,
    'separate_cat_ori': True,
    'cat_weight': 1.0,
    'reg_weight': 1.0,
    'bin_weight': 1.0,
    'enable_reg': False,
    'enable_bin': True,
    'repetions': 4,
}

def create_all_sstage_experiments():
    configs = []
    configs.extend(create_sstage_default())

    return configs

def create_sstage_default():
    config = []
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_default"
    modified['enable_reg'] = True
    config.append(deepcopy(modified))
    return config
