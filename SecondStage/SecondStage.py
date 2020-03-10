from tqdm import tqdm
import re
from datetime import datetime
from multiprocessing import Process, Queue
from copy import deepcopy
from keras import applications, optimizers, backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Reshape, Conv2D, Activation
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, LambdaCallback
from keras.utils import np_utils
from keras.layers.noise import GaussianNoise
from keras.regularizers import l1
import pandas as pd
import numpy as np
from numpy.random import randint
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
import io
import os
import sys
import csv, json, pickle
from lxml import etree

####################################################################

import sys
#path des dir in dem rs_nn_training liegt
sys.path.insert(0, "/home/josi/OvGU/Rolling Swarm/")

from object_detection.utils import label_map_util
from rs_nn_training.Utils.file_utils import *
from rs_nn_training.SecondStage.second_stage_utils import *

####################################################################

TRAIN_RECORD = '/home/josi/OvGU/Rolling Swarm/rs_nn_training/SecondStage/data/10_label/training_rot9_10colors.record'
EVAL_RECORD = '/home/josi/OvGU/Rolling Swarm/rs_nn_training/SecondStage/data/10_label/test_rot3_10colors.record'
OUT_NAME = '10_colors'

LOG_PATH = '/home/josi/OvGU/Rolling Swarm/rs_nn_training/SecondStage/output/'
LABEL_MAP_PATH = '/home/josi/OvGU/Rolling Swarm/rs_nn_training/SecondStage/label_map.pbtxt'

#TYPE = '_bin'    
TYPE = '_cat'
#TYPE = '_reg'

GPU = False

if not GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

###################################################################

default_sstage_conf = {
    'dataset': 'default',
    'epochs_cat': 100,
    'epochs_reg': 60,
    'epochs_bin': 70,
    'optimizer': 'adam',
    'learning_rate': 1e-4,
    'dropout': 0.001,
    'alpha': 0.5,
    'img_size': 35,
    'repetions': 1,
    'batch_size': 4096,
}

def create_all_sstage_experiments():
    configs = []
    #configs.extend(create_sstage_default())
    configs.extend(create_sstage_dropouts_alphas())
    return configs

def create_sstage_dropouts_alphas():
    config = []
    dropouts = [0.001] #, 0.1, 0.2, 0.3
    alphas = [0.75]
    batches = [1024]
    for drop in dropouts:
      modified = deepcopy(default_sstage_conf)
      modified['name'] = "ssdef_drop" + str(drop)
      modified['dropout'] = drop
      for alp in alphas:
        modifiedB = deepcopy(modified)
        modifiedB['name'] = modifiedB['name'] + "_alpha" + str(alp)
        modifiedB['alpha'] = alp
        for batch in batches:
          modifiedC = deepcopy(modifiedB)
          modifiedC['name'] = modifiedC['name'] + "_batch_size" + str(batch)
          modifiedC['batch_size'] = batch
          config.append(deepcopy(modifiedC))
    return config

def create_sstage_default():
    config = []
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_default"
    config.append(deepcopy(modified))
    return config

###################################################################

def train(train_record, conf, type, rep=1):

    out = str(type)

    timestamp = "{:%Y-%m-%d-%H-%M}".format(datetime.now())
    log_path = LOG_PATH + conf['name'] + out + '/' + timestamp + '-r' + str(rep) + '/'
    os.makedirs(log_path, exist_ok=True)
    save_json(log_path + '/experiment_config.json', conf)

    label_map = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)
    num_classes = label_map_util.get_max_label_map_index(label_map_util.load_labelmap(LABEL_MAP_PATH)) + 1

    X,Y,Z,_ = tf_record_load_crops([train_record])
    X_train, Y_train, Z_train = data_to_keras(X,Y,Z,num_classes,conf['img_size'])
    Z2_train = angle_to_bin(Z_train)

    eval_records = EVAL_RECORD

    X,Y,Z,_ = tf_record_extract_crops([eval_records], 1, 0.0, 0.0)
    X_val, Y_val, Z_val = data_to_keras(X,Y,Z,num_classes,conf['img_size'])
    Z2_val = angle_to_bin(Z_val)

    assert len(X_val) > 0 and len(Y_val) > 0 and len(Z_val) > 0, '{} is incomplete'.format(eval_records)

    outputs = None
    model_final = None
    summary = TensorBoardCustom(log_dir=log_path, label_map=label_map, AddCustomMetrics=(out == '_cat'))
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)

    filepath=log_path+"model-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, period=20)
    if conf['optimizer'] == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=conf['learning_rate'])
    elif conf['optimizer'] == 'adam':
        optimizer = optimizers.Adam(lr=conf['learning_rate'])
    else:
        raise Error('Unknown optimizer: ' + conf['optimizer'])

    loss_weights = [1.0 if out == '_cat' else 0.0, 1.0 if out == '_reg' else 0.0, 1.0 if out == '_bin' else 0.0]
    epochs = 0
    if out == '_cat': 
      epochs = conf['epochs_cat']
    elif out == '_reg': 
      epochs = conf['epochs_reg']
    else:
      epochs = conf['epochs_bin']

    mobilenet_base = applications.mobilenet.MobileNet(alpha = conf['alpha'],
                                                      weights = "imagenet",
                                                      include_top=False,
                                                      dropout = conf['dropout'],
                                                      input_shape = (
                                                      conf['img_size'],
                                                      conf['img_size'],
                                                        3
                                                      ))
    shape = (1, 1, int(1024 * conf['alpha']))
    x = GlobalAveragePooling2D()(mobilenet_base.output)
    x = Reshape(shape, name='reshape_1')(x)
    # Branch regression
    reg = Conv2D(1, (1, 1), padding='same', name='conv_reg')(x)
    reg = Activation('linear', name='act_linear')(reg)
    reg = Reshape((1,), name='reg_out')(reg)
    # Branch orientation classification with bins
    bin = Conv2D(NUM_ORI_BINS, (1, 1), padding='same', name='conv_bin')(x)
    bin = Activation('softmax', name='act_bin')(bin)
    bin = Reshape((NUM_ORI_BINS,), name='bin_out')(bin)
    # Branch classification
    x = GaussianNoise(stddev=0.2)(x, training = True) #0.1
    cat = Conv2D(num_classes, (1, 1), padding='same', name='conv_cat')(x) 
    cat = GaussianNoise(stddev=0.01)(cat, training = True)
    cat = Activation('softmax', name='act_softmax')(cat)
    cat = Reshape((num_classes,), name='cat_out')(cat)

    model_final = Model(inputs = mobilenet_base.input, outputs = [cat,reg,bin])
    model_final.compile(optimizer = optimizer,
                  loss={'cat_out': 'categorical_crossentropy',
                        'reg_out': angle_mse,
                        'bin_out': 'categorical_crossentropy',
                  },
                  loss_weights={'cat_out': loss_weights[0],
                                'reg_out': loss_weights[1],
                                'bin_out': loss_weights[2]},
                  metrics ={'cat_out': 'accuracy',
                            'reg_out': angle_mae,
                            'bin_out': angle_bin_rmse})
    model_final.fit(
        X_train,
        {'cat_out': Y_train, 'reg_out': Z_train, 'bin_out': Z2_train},
        validation_data=(X_val, [Y_val, Z_val, Z2_val]),
        batch_size=conf['batch_size'], epochs=epochs, verbose=1,
        callbacks=[summary, checkpoint, earlyStop],
        shuffle=True
    )

    model_final.save(log_path+"model-final.h5")
    print(log_path)
    print("Finished training for {}".format(conf['name']))

###################################################################

exp = create_all_sstage_experiments()

for config in tqdm(exp):
    print("or_conf", config)
    for r in range(config['repetions']):    
        conf = deepcopy(config)
        print("Start training:  " + conf['name'])
              
        if not GPU:
            p = Process(
                target=train,
                args=(TRAIN_RECORD, conf, TYPE, r)
            )
            p.start() 
        else:
            train(TRAIN_RECORD, conf, TYPE, r)
