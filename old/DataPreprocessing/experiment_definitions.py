"""
Experiment config
* Compositor background diversity experiment
    * Train: swarmlab vs COCO vs both
    * Eval: all
* First stage resolution experiment
    * Train: 200x150 vs 400x300 vs 800x600
    * Eval: all
* Compositor amount generated experiment
    * Train: 1,000 vs 5,000 vs 20,000
    * Eval: all
* Automatic/Manual crop experiment
    * Train: automatic vs manual vs both
    * Eval: all
* Crop Position experiment
    * Train: 4,5 vs 1,2,3,4,5
    * Eval: all
* Lighting experiment
    * Train: artificial vs natural vs both vs artificial augmented
    * Eval: all
* Ground color experiment
    * Train: green vs yellow vs both for spheros natural lighting
    * Eval: spheros natural lighting
* ID transfer experiment
    * Only artifical lighting
    * First Stage Train: youbot all + copter letter
    * Second Stage Train: youbot mp1, youbot mp2, copter letter, all
    * Eval both: youbot mp1 + youbot mp2 + copter letter
"""

from copy import deepcopy

exp_default = {
    "sphero": {
        "scl_min": 1.0,
        "scl_max": 1.0
    },
    "copter": {
        "scl_min": 1.0,
        "scl_max": 2.2
    },
    "youbot": {
        "scl_min": 1.0,
        "scl_max": 1.0
    },
    "skip_compositing": False,
    "robot_type": ["copter","sphero","youbot"],
    "arena_position": ["top_left","top_right","bottom_left","bottom_right","center"],
    "height": ["0","60"],
    "background_color": ["green","yellow"],
    "crop_method": ["manual"],
    "lighting": ["artificial","natural"],
    "background_set": ["Swarmlab","COCO2"],
    "compositor_amount": 5000,
    # (width, height)
    "first_stage_resolution": (400,300),
    # (min, max)
    "crop_brightness_augmentation": (1.0,1.0),
    "eval": ['by_al_copter_led', 'by_nl_copter_led',
             'by_al_copter_letter', 'by_nl_copter_letter',
             'by_nl_sphero', 'bg_al_sphero', 'bg_nl_sphero',
             'bg_al_youbot_mp1', 'bg_al_youbot_mp2', 'bg_nl_youbot_mp2'],
    "sstage_out_var": 0.15,
    "sstage_in_var": 0.1,

}

def create_all_experiments():
    exps = []
    exps.append(create_default_experiment())
    exps.append(create_background_diversity_experiment())
    exps.append(create_first_stage_resolution_experiment())
    exps.append(create_compositor_amount_experiment())
    exps.append(create_crop_method_experiment())
    exps.append(create_crop_position_experiment())
    exps.append(create_lighting_experiment())
    # exps.append(create_crop_ground_color_experiment())
    # exps.append(create_id_transfer_experiment())
    return exps

def create_default_experiment():
    config = []
    swarmlab = deepcopy(exp_default)
    swarmlab['name'] = "default"
    config.append(swarmlab)
    return config

def create_no_crop_var_experiment():
    config = []
    modified = deepcopy(exp_default)
    modified['name'] = "no_crop_var"
    modified['sstage_out_var'] = 0
    modified['sstage_in_var'] = 0
    config.append(deepcopy(modified))
    return config

def create_youbot_test_dataset():
    config = []
    modified = deepcopy(exp_default)
    modified['name'] = "youbot_test_dataset"
    modified['sstage_out_var'] = 0
    modified['sstage_in_var'] = 0
    modified['compositor_amount'] = 5000*2
    modified['robot_type'] = ['youbot']
    modified['crop_brightness_augmentation'] = (0.8,1.4)
    config.append(deepcopy(modified))
    return config

def create_background_diversity_experiment():
    config = []
    swarmlab = deepcopy(exp_default)
    swarmlab['name'] = "background_diversity_swarmlab"
    swarmlab['background_set'] = ["Swarmlab"]
    config.append(swarmlab)
    coco = deepcopy(exp_default)
    coco['name'] = "background_diversity_coco"
    coco['background_set'] = ["COCO2"]
    config.append(coco)
    return config

def create_first_stage_resolution_experiment():
    config = []
    modified = deepcopy(exp_default)
    modified['name'] = "first_stage_resolution_200_150"
    modified['first_stage_resolution'] = (200,150)
    modified['skip_compositing'] = True
    config.append(deepcopy(modified))
    modified = deepcopy(exp_default)
    modified['name'] = "first_stage_resolution_800_600"
    modified['first_stage_resolution'] = (800,600)
    modified['skip_compositing'] = True
    config.append(deepcopy(modified))
    return config

def create_compositor_amount_experiment():
    config = []
    modified = deepcopy(exp_default)
    modified['name'] = "compositor_amount_1000"
    modified['compositor_amount'] = 1000
    config.append(deepcopy(modified))
    modified = deepcopy(exp_default)
    modified['name'] = "compositor_amount_20000"
    modified['compositor_amount'] = 20000
    config.append(deepcopy(modified))
    return config

def create_crop_method_experiment():
    config = []
    modified = deepcopy(exp_default)
    modified['name'] = "crop_method_automatic"
    modified['crop_method'] = ["automatic"]
    modified['height'] = ['0']
    config.append(deepcopy(modified))
    modified = deepcopy(exp_default)
    modified['name'] = "crop_method_manual"
    modified['crop_method'] = ["manual"]
    modified['height'] = ['0']
    config.append(deepcopy(modified))
    modified = deepcopy(exp_default)
    modified['name'] = "crop_method_both"
    modified['crop_method'] = ["automatic","manual"]
    modified['height'] = ['0']
    config.append(deepcopy(modified))
    return config

def create_crop_position_experiment():
    config = []
    modified = deepcopy(exp_default)
    modified['name'] = "crop_position_4"
    modified['arena_position'] = ["center"]
    modified['height'] = ['0']
    config.append(deepcopy(modified))
    modified = deepcopy(exp_default)
    modified['name'] = "crop_position_245"
    modified['arena_position'] = ["top_right","center","bottom_left"]
    modified['height'] = ['0']
    config.append(deepcopy(modified))
    #modified = deepcopy(exp_default)
    #modified['name'] = "crop_position_12345"
    #modified['arena_position'] = ["top_left","top_right","bottom_left","bottom_right","center"]
    #modified['height'] = ['0']
    #config.append(deepcopy(modified))
    return config

def create_lighting_experiment():
    config = []
    modified = deepcopy(exp_default)
    modified['name'] = "lighting_artificial"
    modified['lighting'] = ["artificial"]
    config.append(deepcopy(modified))
    modified = deepcopy(exp_default)
    modified['name'] = "lighting_natural"
    modified['lighting'] = ["natural"]
    config.append(deepcopy(modified))
    modified = deepcopy(exp_default)
    modified['name'] = "lighting_artificial_augmented"
    modified['lighting'] = ["artificial"]
    # TODO: Check parameters
    modified['crop_brightness_augmentation'] = (1.0,1.4)
    config.append(deepcopy(modified))
    return config

def create_crop_ground_color_experiment():
    config = []
    modified = deepcopy(exp_default)
    modified['name'] = "crop_ground_color_green"
    modified['background_color'] = ["green"]
    modified['robot_type'] = ["sphero"]
    modified['eval'] = ['by_nl_sphero', 'bg_nl_sphero']
    config.append(deepcopy(modified))
    modified = deepcopy(exp_default)
    modified['name'] = "crop_ground_color_yellow"
    modified['background_color'] = ["yellow"]
    modified['robot_type'] = ["sphero"]
    modified['eval'] = ['by_nl_sphero', 'bg_nl_sphero']
    config.append(deepcopy(modified))
    modified = deepcopy(exp_default)
    modified['name'] = "crop_ground_color_both"
    modified['background_color'] = ["green","yellow"]
    modified['robot_type'] = ["sphero"]
    modified['eval'] = ['by_nl_sphero', 'bg_nl_sphero']
    config.append(deepcopy(modified))
    return config

def create_id_transfer_experiment():
    config = []
    modified = deepcopy(exp_default)
    modified['name'] = "id_transfer_first_stage"
    modified['robot_type'] = ["youbot","copter"]
    # TODO: Only copter letter
    modified['lighting'] = ["artificial"]
    modified['eval'] = ['by_al_copter_letter',
                        'bg_al_youbot_mp1', 'bg_al_youbot_mp2']
    config.append(deepcopy(modified))
    # TODO: Second stage config
    return config

def create_sstage_augmented_data():
    config = []
    modified = deepcopy(exp_default)
    modified['name'] = "sstage_augmented_5000"
    # TODO: check parameters
    modified['crop_brightness_augmentation'] = (1.0,1.4)
    config.append(deepcopy(modified))
    return config

# Second stage config separately with reference to experiment dataset
default_sstage_conf = {
    'dataset': 'default',
    'types': ['copter','sphero','youbot'],
    # converged after 15 / 10 / 5 epochs
    # 10 epochs after convergance for evaluation phase
    'epochs_cat': 25,
    'epochs_reg': 20,
    'epochs_bin': 15,
    'optimizer': 'adam',
    'learning_rate': 3e-4,
    'dropout': 0,
    'alpha': 0.5,
    'img_size': 128,
    'separate_cat_ori': True,
    'cat_weight': 1.0,
    'reg_weight': 1.0,
    'bin_weight': 1.0,
    'enable_reg': False,
    'enable_bin': True,
    'repetions': 4,
}

# TODO: Don't repeat standard config
def create_all_sstage_experiments():
    configs = []
    configs.extend(create_sstage_default())
    # configs.extend(create_sstage_no_crop_var())
    # configs.extend(create_sstage_youbot_test())
    # configs.extend(create_sstage_sphero_test())
    # configs.extend(create_sstage_learning_rate_experiment())
    # configs.extend(create_sstage_dropout_experiment())
    # configs.extend(create_sstage_augmented_experiment())
    # configs.extend(create_sstage_adam_experiment())
    # configs.extend(create_sstage_loss_weight_experiment())
    configs.extend(create_sstage_background_experiment())
    configs.extend(create_sstage_amount_experiment())
    configs.extend(create_sstage_crop_method_experiment())
    configs.extend(create_sstage_alpha_experiment())
    configs.extend(create_sstage_crop_position_experiment())
    configs.extend(create_sstage_lighting_experiment())

    return configs

def create_sstage_default():
    config = []
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_default"
    modified['enable_reg'] = True
    config.append(deepcopy(modified))
    return config

def create_sstage_alpha_experiment():
    config = []
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_alpha_25"
    modified['alpha'] = 0.25
    config.append(deepcopy(modified))
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_alpha_100"
    modified['alpha'] = 1.0
    config.append(deepcopy(modified))
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_alpha_75"
    modified['alpha'] = 0.75
    config.append(deepcopy(modified))
    return config

def create_sstage_background_experiment():
    config = []
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_background_swarmlab"
    modified['dataset'] = "background_diversity_swarmlab"
    config.append(deepcopy(modified))
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_background_coco"
    modified['dataset'] = "background_diversity_coco"
    config.append(deepcopy(modified))
    return config

def create_sstage_amount_experiment():
    config = []
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_amount_1000"
    modified['dataset'] = "compositor_amount_1000"
    config.append(deepcopy(modified))
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_amount_20000"
    modified['dataset'] = "compositor_amount_20000"
    config.append(deepcopy(modified))
    return config

def create_sstage_crop_method_experiment():
    config = []
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_crop_method_automatic"
    modified['dataset'] = "crop_method_automatic"
    config.append(deepcopy(modified))
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_crop_method_both"
    modified['dataset'] = "crop_method_both"
    config.append(deepcopy(modified))
    return config

def create_sstage_crop_position_experiment():
    config = []
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_crop_position_4"
    modified['dataset'] = "crop_position_4"
    config.append(deepcopy(modified))
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_crop_position_245"
    modified['dataset'] = "crop_position_245"
    config.append(deepcopy(modified))
    return config

def create_sstage_lighting_experiment():
    config = []
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_lighting_artificial"
    modified['dataset'] = "lighting_artificial"
    config.append(deepcopy(modified))
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_lighting_natural"
    modified['dataset'] = "lighting_natural"
    config.append(deepcopy(modified))
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_lighting_artificial_augmented"
    modified['dataset'] = "lighting_artificial_augmented"
    config.append(deepcopy(modified))
    return config

# def create_sstage_no_crop_var():
#     config = []
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_no_crop_var"
#     modified['dataset'] = "no_crop_var"
#     config.append(deepcopy(modified))
#     return config
#
# def create_sstage_youbot_test():
#     config = []
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_youbot_test"
#     modified['dataset'] = "youbot_test_dataset"
#     modified['types'] = ['youbot']
#     config.append(deepcopy(modified))
#     return config
#
# def create_sstage_sphero_test():
#     config = []
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_sphero_test"
#     modified['types'] = ['sphero']
#     config.append(deepcopy(modified))
#     return config

# def create_sstage_learning_rate_experiment():
#     config = []
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_learning_rate_2e-3"
#     modified['learning_rate'] = 2e-3
#     config.append(deepcopy(modified))
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_learning_rate_6e-3"
#     modified['learning_rate'] = 6e-3
#     config.append(deepcopy(modified))
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_learning_rate_2e-4"
#     modified['learning_rate'] = 2e-4
#     config.append(deepcopy(modified))
#     return config
#
# def create_sstage_adam_experiment():
#     config = []
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_adam"
#     modified['dataset'] = "sstage_augmented_5000"
#     modified['optimizer'] = 'adam'
#     modified['learning_rate'] = 3e-4
#     config.append(deepcopy(modified))
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_adam_drop_1e-3"
#     modified['dataset'] = "sstage_augmented_5000"
#     modified['optimizer'] = 'adam'
#     modified['learning_rate'] = 3e-4
#     modified['dropout'] = 1e-3
#     config.append(deepcopy(modified))
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_adam_drop_1e-4"
#     modified['dataset'] = "sstage_augmented_5000"
#     modified['optimizer'] = 'adam'
#     modified['learning_rate'] = 3e-4
#     modified['dropout'] = 1e-4
#     config.append(deepcopy(modified))
#     return config
#
# def create_sstage_dropout_experiment():
#     config = []
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_dropout_1e-5"
#     modified['dropout'] = 1e-5
#     modified['types'] = ['copter']
#     config.append(deepcopy(modified))
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_dropout_1e-4"
#     modified['dropout'] = 1e-4
#     modified['types'] = ['copter']
#     config.append(deepcopy(modified))
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_dropout_1e-3"
#     modified['learning_rate'] = 1e-3
#     modified['types'] = ['copter']
#     config.append(deepcopy(modified))
#     return config
#
# def create_sstage_augmented_experiment():
#     config = []
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_augmented_learning_rate_2e-3"
#     modified['dataset'] = "sstage_augmented_5000"
#     modified['learning_rate'] = 2e-3
#     config.append(deepcopy(modified))
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_augmented_learning_rate_2e-4"
#     modified['dataset'] = "sstage_augmented_5000"
#     modified['learning_rate'] = 2e-4
#     config.append(deepcopy(modified))
#     return config
#
# def create_sstage_loss_weight_experiment():
#     config = []
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_loss_weight_1e-1"
#     modified['dataset'] = "sstage_augmented_5000"
#     modified['optimizer'] = 'adam'
#     modified['learning_rate'] = 3e-4
#     modified['ori_weight'] = 1e-1
#     config.append(deepcopy(modified))
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_loss_weight_1e-2"
#     modified['dataset'] = "sstage_augmented_5000"
#     modified['optimizer'] = 'adam'
#     modified['learning_rate'] = 3e-4
#     modified['ori_weight'] = 1e-2
#     config.append(deepcopy(modified))
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_loss_weight_1e-3"
#     modified['dataset'] = "sstage_augmented_5000"
#     modified['optimizer'] = 'adam'
#     modified['learning_rate'] = 3e-4
#     modified['ori_weight'] = 1e-3
#     config.append(deepcopy(modified))
#     modified = deepcopy(default_sstage_conf)
#     modified['name'] = "sstage_loss_weight_1e-4"
#     modified['dataset'] = "sstage_augmented_5000"
#     modified['optimizer'] = 'adam'
#     modified['learning_rate'] = 3e-4
#     modified['ori_weight'] = 1e-4
#     config.append(deepcopy(modified))
#     return config
