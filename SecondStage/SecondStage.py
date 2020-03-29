
#!pip install tensorflow-gpu==1.15
#!pip install object-detection-core
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util

from tqdm import tqdm
import re
from datetime import datetime
from multiprocessing import Process, Queue

from copy import deepcopy

from keras import applications, optimizers, backend as K
from keras.models import Sequential, Model, load_model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Reshape, Conv2D, Activation, BatchNormalization, MaxPool2D, Input, GaussianNoise
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping, Callback, LambdaCallback
from keras.utils import np_utils
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


os.chdir('/content/gdrive/My Drive/')
from SecondStage_colab import second_stage_utils
from SecondStage_colab import file_utils

sys.path.append('/content/gdrive/My Drive/SecondStage_colab/')
os.environ['PYTHONPATH'] += ':/content/gdrive/My Drive/SecondStage_colab/'

print(tf.__version__)

LOG_PATH = '/content/gdrive/My Drive/SecondStage_colab/output/'
LABEL_MAP_PATH = '/content/gdrive/My Drive/SecondStage_colab/robot_label_map_komplett.pbtxt'
RECORD_PATH = '/content/gdrive/My Drive/data/generated/SecondStage_X/'

GPU = True

from file_utils import save_json
from second_stage_utils import *

# you can not set these here, change them in the second_stage_utils
print(ANGLES_PER_BIN)
print(NUM_ORI_BINS)


def train(train_record, eval_record, conf, out, rep=1, useMobileNet=True):
    timestamp = "{:%Y-%m-%d-%H-%M}".format(datetime.now())
    log_path = LOG_PATH + conf['name'] + '_' + out + '/' + timestamp + '-r' + str(rep) + '/'
    os.makedirs(log_path, exist_ok=True)
    save_json(log_path + '/experiment_config.json', conf)

    label_map = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)
    num_classes = label_map_util.get_max_label_map_index(label_map_util.load_labelmap(LABEL_MAP_PATH)) + 1
    
    X,Y,Z,_ = tf_record_load_crops([train_record])
    print("--- crops loaded ---")
    num_classes = label_map_util.get_max_label_map_index(label_map_util.load_labelmap(LABEL_MAP_PATH)) + 1
    X_train, Y_train, Z_train = data_to_keras(X,Y,Z, num_classes, conf['img_size'])
    Z2_train = angle_to_bin(Z_train)
    print("´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´")
    print(Y_train.shape)
    print(num_classes)
    print(Z2_train.shape)
    print(NUM_ORI_BINS)
    print("´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´")

    X,Y,Z,_ = tf_record_extract_crops([eval_record], 1, 0.0, 0.0)
    X_val, Y_val, Z_val = data_to_keras(X,Y,Z,num_classes,conf['img_size'])
    Z2_val = angle_to_bin(Z_val)

    assert len(X_val) > 0 and len(Y_val) > 0 and len(Z_val) > 0, '{} is incomplete'.format(eval_record)

    outputs = None
    model_final = None
    summary = TensorBoardCustom(log_dir=log_path, label_map=label_map, AddCustomMetrics=(out == '_cat'))
    earlyStop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

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
    model_final = None
    if useMobileNet:
      mobilenet_base = applications.mobilenet.MobileNet(alpha = conf['alpha'],
                                                        weights = "imagenet",
                                                        include_top=False,
                                                        dropout = conf['dropout'],
                                                        input_shape = (
                                                        conf['img_size'],
                                                        conf['img_size'],
                                                          3
                                                        ))
      
      x = GlobalAveragePooling2D()(mobilenet_base.output)
      x = Dense(128, activation='relu')(x)
      # Branch regression
      reg = Dense(1, activation='linear', name='dense_reg')(x)
      reg = Reshape((1,), name='reg_out')(reg)
      # Branch orientation classification with bins
      bin = Dense(NUM_ORI_BINS, activation='softmax', name='dense_bin')(x)
      bin = Reshape((NUM_ORI_BINS,), name='bin_out')(bin)
      # Branch classification
      cat = Dense(num_classes, activation='softmax', name='dense_cat')(x)
      cat = Reshape((num_classes,), name='cat_out')(cat)

      model_final = Model(inputs = mobilenet_base.input, outputs = [cat,reg,bin])

    else:
      inputs = Input(shape=(35,35,3), name = 'input_1')
      x = GaussianNoise(3.0)(inputs)
      x = Conv2D(32, (3,3), padding='same', activation='relu', input_shape=(35, 35, 3))(x)
      x = BatchNormalization()(x)
      x = MaxPool2D()(x)
      x = Conv2D(64, (3,3), padding='same', activation='relu')(x)
      x = BatchNormalization()(x)
      x = MaxPool2D()(x)
      filters = 128
      x = Conv2D(filters, (3,3), padding='same', activation='relu')(x)
      x = BatchNormalization()(x)
      x = MaxPool2D()(x)
      x = Reshape((filters*16,))(x)
      #x = Flatten()(x)
      x = Dropout(conf['dropout'])(x)
      x = Dense(512, activation='relu')(x)
      x = Dropout(conf['dropout'])(x)
      x = Dense(512, activation='relu')(x)
      # Branch regression
      reg = Dense(1, activation='linear', name='dense_reg')(x)
      reg = Reshape((1,), name='reg_out')(reg)
      # Branch orientation classification with bins
      bin = Dense(NUM_ORI_BINS, activation='softmax', name='dense_bin')(x)
      bin = Reshape((NUM_ORI_BINS,), name='bin_out')(bin)
      # Branch classification
      cat = Dense(num_classes, activation='softmax', name='dense_cat')(x)
      cat = Reshape((num_classes,), name='cat_out')(cat)

      model_final = Model(inputs = inputs, outputs = [cat,reg,bin])

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
    orig_stdout = sys.stdout
    with open(log_path + 'model_summary.txt', 'w') as f:
      sys.stdout = f 
      print(model_final.summary())
      f.close()
    sys.stdout = orig_stdout

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

default_sstage_conf = {
    'dataset': 'default',
    'epochs_cat': 70,
    'epochs_reg': 70,
    'epochs_bin': 70,
    'optimizer': 'adam',
    'learning_rate': 1e-3,
    'dropout': 0.001,
    'alpha': 0.5,
    'img_size': 35,
    'repetions': 3,
    'batch_size': 4096*2,
}

def create_all_sstage_experiments():
    configs = []
    #configs.extend(create_sstage_default())
    configs.extend(create_sstage_dropouts())
    #configs.extend(create_sstage_alphas())
    return configs

def create_sstage_alphas():
    config = []
    dropouts = [0.5]
    for drop in dropouts:
      modified = deepcopy(default_sstage_conf)
      modified['name'] = "ssdef_alpha" + str(drop)
      modified['dropout'] = drop
      config.append(modified)
    return config

def create_sstage_dropouts():
    config = []
    dropouts = [0.5, 0.6]
    for drop in dropouts:
      modified = deepcopy(default_sstage_conf)
      modified['name'] = "ssdef_drop" + str(drop)
      modified['dropout'] = drop
      config.append(modified)
    return config

def create_sstage_default():
    config = []
    modified = deepcopy(default_sstage_conf)
    modified['name'] = "ssdef" + "_"
    config.append(deepcopy(modified))
    return config


exp = create_all_sstage_experiments()
TORUN = []
#TORUN.append('_bin')
TORUN.append('_cat')
#TORUN.append('_reg')
useMobileNet = False
#RECORDS = [('training_rot9_13colors_neu_noflip.record', 'validation_rot6_13colors_neu.record')]
#RECORDS = [('training_rot12_10colors.record','validation_rot12_10colors.record')]
#RECORDS = [('training_rot12_13colors.record','validation_rot12_13colors.record')]
#RECORDS = [('training_test.record', 'training_test.record')]

for out in TORUN:
  for config in tqdm(exp):
    print("or_conf", config)
    for train_record, eval_records in RECORDS:
      for r in range(config['repetions']):
        conf = deepcopy(config)
        conf['name'] = conf['name']+'_'+out+"_"+train_record
        print("Start training:  " + conf['name'])
        
        if not GPU:
            p = Process(
              target=train,
              args=(RECORD_PATH + train_record, RECORD_PATH + eval_records, conf, out, r, useMobileNet)
            )
            p.start()
        else:
          train(RECORD_PATH + train_record, RECORD_PATH + eval_records, conf, out, useMobileNet)