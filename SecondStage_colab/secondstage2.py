import csv
import io
import json
import os
import pickle
import re
import sys
from copy import deepcopy
from datetime import datetime
from multiprocessing import Process, Queue

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras import applications
from keras import backend as K
from keras import optimizers
from keras.callbacks import (Callback, EarlyStopping, LambdaCallback,
                             LearningRateScheduler, ModelCheckpoint,
                             TensorBoard)
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          GlobalAveragePooling2D, Reshape)
from keras.models import Model, Sequential, load_model
from keras.utils import np_utils
from lxml import etree
from numpy.random import randint
from object_detection.utils import dataset_util, label_map_util
from PIL import Image
from tqdm import tqdm

import file_utils
import pandas as pd
import seaborn as sns
import second_stage_utils
from file_utils import save_json
from second_stage_utils import *

#NAME =
#example: cat_exptype_dataset(training)_dataset(eval)_conf(batch size etc) 

LOG_PATH = 'output/'
LABEL_MAP_PATH = 'label_map.pbtxt'
EVAL_RECORD = 'data/validation_rot3_13colors.record'

GPU = True


def train(train_record, conf, out, rep=1):
    timestamp = "{:%Y-%m-%d-%H-%M}".format(datetime.now())
    log_path = LOG_PATH + conf['name'] + '_' + out + '/' + timestamp + '-r' + str(rep) + '/'
    os.makedirs(log_path, exist_ok=True)
    save_json(log_path + '/experiment_config.json', conf)

    label_map = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)

    num_classes = label_map_util.get_max_label_map_index(label_map_util.load_labelmap(LABEL_MAP_PATH)) + 1

    X,Y,Z,_ = tf_record_load_crops([train_record])

    X_train, Y_train, Z_train = data_to_keras(X,Y,Z,num_classes,conf['img_size'])
    Z2_train = angle_to_bin(Z_train)
    print("´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´")
    print(Y_train.shape)
    print(num_classes)
    print(Z2_train.shape)
    print(NUM_ORI_BINS)
    print("´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´")

    eval_records = EVAL_RECORD

    X,Y,Z,_ = tf_record_extract_crops([eval_records], 1, 0.0, 0.0)
    X_val, Y_val, Z_val = data_to_keras(X,Y,Z,num_classes,conf['img_size'])
    Z2_val = angle_to_bin(Z_val)

    assert len(X_val) > 0 and len(Y_val) > 0 and len(Z_val) > 0, '{} is incomplete'.format(eval_records)

    outputs = None
    model_final = None
    summary = TensorBoardCustom(log_dir=log_path, label_map=label_map, AddCustomMetrics=(out == '_cat'))

    filepath=log_path+"model-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, period=20)
    if conf['optimizer'] == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=conf['learning_rate'])
    elif conf['optimizer'] == 'adam':
        optimizer = optimizers.Adam(lr=conf['learning_rate'])
    else:
        raise Error('Unknown optimizer: ' + conf['optimizer'])


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
    #x = Dropout(conf['dropout'], name='dropout')(x)

    if out == '_cat':
      # Branch classification
      cat = Conv2D(num_classes, (1, 1),
                padding='same', name='conv_cat')(x)
      cat = Activation('softmax', name='act_softmax')(cat)
      cat = Reshape((num_classes,), name='cat_out')(cat)
      outputs = cat
      model_final = Model(inputs = mobilenet_base.input, outputs = outputs)
      model_final.compile(optimizer = optimizer,
                          loss={'cat_out': 'categorical_crossentropy'},
                          metrics ={'cat_out': 'accuracy'})
      model_final.fit(
        X_train,
        {'cat_out': Y_train},
        validation_data=(X_val, Y_val),
        batch_size=conf['batch_size'], epochs = conf['epochs_cat'], verbose=1,
        callbacks=[summary,checkpoint],
        shuffle=True
      )
      
    elif out == '_reg':
      # Branch regression
      reg = Conv2D(1, (1, 1),
                padding='same', name='conv_reg')(x)
      reg = Activation('linear', name='act_linear')(reg)
      reg = Reshape((1,), name='reg_out')(reg)
      outputs = reg
      model_final = Model(inputs = mobilenet_base.input, outputs = outputs)
      model_final.compile(optimizer = optimizer,
                          loss={'reg_out': angle_mse},
                          metrics ={'reg_out': angle_mae})
      model_final.fit(
        X_train,
        {'reg_out': Z_train},
        validation_data=(X_val, Z_val),
        batch_size=conf['batch_size'], epochs = conf['epochs_reg'], verbose=1,
        callbacks=[summary,checkpoint],
        shuffle=True
      )
            
    elif out == '_bin':
      # Branch orientation classification with bins
      bin = Conv2D(NUM_ORI_BINS, (1, 1),
                padding='same', name='conv_bin')(x)
      bin = Activation('softmax', name='act_bin')(bin)
      bin = Reshape((NUM_ORI_BINS,), name='bin_out')(bin)
      outputs = bin
      model_final = Model(inputs = mobilenet_base.input, outputs = outputs)
      model_final.compile(optimizer = optimizer,
                          loss= {'bin_out': 'categorical_crossentropy'},
                          metrics = [angle_bin_mae, angle_bin_rmse])
      model_final.fit(
        X_train,
        {'bin_out': Z2_train},
        validation_data=(X_val, Z2_val),
        batch_size=conf['batch_size'], epochs = conf['epochs_bin'], verbose=1,
        callbacks=[summary,checkpoint],
        shuffle=True
      )
    else:
        print('UNKNOWN OUTPUT CONFIG {}'.format(out))

    model_final.save(log_path+"model-final.h5")
    print(log_path)
    print("Finished training for {}".format(conf['name']))

default_sstage_conf = {
    'dataset': 'default',
    'epochs_cat': 40,
    'epochs_reg': 40,
    'epochs_bin': 40,
    'optimizer': 'adam',
    'learning_rate': 3e-4,
    'dropout': 0.001,
    'alpha': 0.5,
    'img_size': 35,
    'repetions': 1,
    'batch_size': 7020,
}

def create_all_sstage_experiments():
    configs = []
    #configs.extend(create_sstage_default())
    #configs.extend(create_sstage_dropouts())
    #configs.extend(create_sstage_alphas())
    #configs.extend(create_sstage_dropouts_alphas())
    configs.extend(create_sstage_batch_sizes())
    return configs

def create_sstage_batch_sizes():
    config = []
    batchsizes = [3510, 2106, 1053]
    for size in batchsizes:
      modified = deepcopy(default_sstage_conf)
      modified['name'] = "sstage_default_bsize" + str(size)
      modified['batch_size'] = size
      config.append(deepcopy(modified))
    return config

def create_sstage_dropouts_alphas():
    config = []
    dropouts = [0.001, 0.3, 0.4, 0.5, 0.6] #
    alphas = [0.25] # 0.5, 0.75, 1.0
    for drop in dropouts:
      modified = deepcopy(default_sstage_conf)
      modified['name'] = "ssdef_drop" + str(drop)
      modified['dropout'] = drop
      for alp in alphas:
        modifiedB = deepcopy(modified)
        modifiedB['name'] = modifiedB['name'] + "_alpha" + str(alp)
        modifiedB['alpha'] = alp
        config.append(deepcopy(modifiedB))
    return config

def create_sstage_dropouts():
    config = []
    dropouts = [0.25]
    for drop in dropouts:
      modified = deepcopy(default_sstage_conf)
      modified['name'] = "sstage_default_drop" + str(drop)
      modified['dropout'] = drop
      config.append(deepcopy(modified))
    return config

def create_sstage_alphas():
    config = []
    alphas = [0.5, 0.75, 1.0]
    for alp in alphas:
      modified = deepcopy(default_sstage_conf)
      modified['name'] = "sstage_default_alpha" + str(alp)
      modified['alpha'] = alp
      config.append(deepcopy(modified))
    return config

def create_sstage_default():
    config = []
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "sstage_default"
    config.append(deepcopy(modified))
    return config

exp = create_all_sstage_experiments()
TORUN = []
TORUN.append('_bin')
TORUN.append('_cat')
train_records = [ 'training_rot9_13colors.record']
for config in tqdm(exp):
    print("or_conf", config)
    for r in range(config['repetions']):     
        for out in TORUN: # 
            for train_record in train_records:
              conf = deepcopy(config)
              conf['name'] = conf['name']+'_'+out+"_"+train_record
              print("Start training:  " + conf['name'])
              
              if not GPU:
                  p = Process(
                    target=train,
                    args=("data/" + train_record, conf, out, r)
                  )
                  p.start() 
              else:
                train("data/" + train_record, conf, out)
