import os
import re
import threading
from tqdm import tqdm
from multiprocessing import Process, Queue
from argparse import ArgumentParser
from datetime import datetime
from keras import applications
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Reshape, Conv2D, Activation
from keras import backend as K
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, TensorBoard, EarlyStopping
from keras.models import load_model

"""import sys
#path des dir in dem rs_nn_training liegt
sys.path.insert(0, "/home/josi/OvGU/Rolling Swarm/")
from object_detection.utils import label_map_util
from rs_nn_training.SecondStage.exp_def import *
from rs_nn_training.Utils.file_utils import *
from rs_nn_training.SecondStage.second_stage_utils import *
"""
################################################################################

GPU = True
"""
LOG_PATH = "/home/josi/OvGU/Rolling Swarm/output/second_stage/09-01-rot9-"
LABEL_MAP_PATH = '/home/josi/OvGU/Rolling Swarm/rs_nn_training/SecondStage/label_map.pbtxt'
TRAIN_DIR = "/home/josi/OvGU/Rolling Swarm/data/train"
EVAL_DIR = "/home/josi/OvGU/Rolling Swarm/data/test"
"""
#BATCH_SIZE = 32
BATCH_SIZE = 1000

if not GPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

################################################################################

def train(train_record, conf, types, out, rep=1):
    timestamp = "{:%Y-%m-%d-%H-%M}".format(datetime.now())
    log_path = LOG_PATH + conf['name'] + '_' + \
                ''.join(types) + out + '/' + \
                timestamp + '-r' + str(rep) + '/'
    os.makedirs(log_path, exist_ok=True)
    save_json(log_path + '/experiment_config.json', conf)
    label_map = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)
    num_classes = label_map_util.get_max_label_map_index(
                            label_map_util.load_labelmap(LABEL_MAP_PATH)) + 1

    X,Y,Z,_ = tf_record_load_crops([train_record])
    X_train, Y_train, Z_train = data_to_keras(X,Y,Z,num_classes,conf['img_size'])
    Z2_train = angle_to_bin(Z_train)
    print("´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´")
    print(Y_train.shape)
    print(num_classes)
    print(Z2_train.shape)
    print(NUM_ORI_BINS)
    print("´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´´")

    eval_records = get_recursive_file_list(EVAL_DIR, file_matchers=types)

    #X,Y,Z,_ = tf_record_extract_crops(eval_records, 1, 0.0, 0.0, class_filters=types)
    X,Y,Z,_ = tf_record_extract_crops(eval_records, 1, 0.0, 0.0, class_filters="sphero")
    X_val, Y_val, Z_val = data_to_keras(X,Y,Z,num_classes,conf['img_size'])
    Z2_val = angle_to_bin(Z_val)
    assert len(X_val) > 0 and len(Y_val) > 0 and len(Z_val) > 0, \
        '{} is incomplete'.format(eval_records)
    mobilenet_base = applications.mobilenet.MobileNet(alpha = conf['alpha'],
                                                      weights = "imagenet",
                                                      include_top=False,
                                                      input_shape = (
                                                      conf['img_size'],
                                                      conf['img_size'],
                                                        3
                                                      ))
    shape = (1, 1, int(1024 * conf['alpha']))
    x = GlobalAveragePooling2D()(mobilenet_base.output)
    x = Reshape(shape, name='reshape_1')(x)
    x = Dropout(conf['dropout'], name='dropout')(x)
    # Branch regression
    reg = Conv2D(1, (1, 1),
               padding='same', name='conv_reg')(x)
    reg = Activation('linear', name='act_linear')(reg)
    reg = Reshape((1,), name='reg_out')(reg)
    # Branch orientation classification with bins
    bin = Conv2D(NUM_ORI_BINS, (1, 1),
               padding='same', name='conv_bin')(x)
    bin = Activation('softmax', name='act_bin')(bin)
    bin = Reshape((NUM_ORI_BINS,), name='bin_out')(bin)
    # Branch classification
    cat = Conv2D(num_classes, (1, 1),
               padding='same', name='conv_cat')(x)
    cat = Activation('softmax', name='act_softmax')(cat)
    cat = Reshape((num_classes,), name='cat_out')(cat)

    # creating the final model
    model_final = None
    model_final = Model(inputs = mobilenet_base.input, outputs = [cat,reg,bin])

    if conf['optimizer'] == 'rmsprop':
        optimizer = optimizers.RMSprop(lr=conf['learning_rate'])
    elif conf['optimizer'] == 'adam':
        optimizer = optimizers.Adam(lr=conf['learning_rate'])
    else:
        raise Error('Unknown optimizer: ' + conf['optimizer'])

    model_final.compile(optimizer = optimizer,
                        loss={'cat_out': 'categorical_crossentropy',
                              'reg_out': angle_mse,
                              'bin_out': 'categorical_crossentropy',
                        },
                        loss_weights={'cat_out': conf['cat_weight'],
                                      'reg_out': conf['reg_weight'],
                                      'bin_out': conf['bin_weight']},
                        metrics ={'cat_out': 'accuracy',
                                  'reg_out': angle_mae,
                                  'bin_out': angle_bin_error})

    summary = TensorBoardCustom(log_dir=log_path,label_map=label_map)
    filepath=log_path+"model-{epoch:02d}.h5"
    checkpoint = ModelCheckpoint(filepath, verbose=1, period=5)

    

    model_final.fit(
        X_train,
        {'cat_out': Y_train, 'reg_out': Z_train, 'bin_out': Z2_train},
        validation_data=(X_val,[Y_val,Z_val, Z2_val]),
        batch_size=BATCH_SIZE, epochs=conf['epochs'], verbose=0,
        callbacks=[summary,checkpoint],
        shuffle=True
    )
    model_final.save(log_path+"model-final.h5")
    print("Finished training for {}_{}".format(conf['name'],t))

def read_command_line_args():
    parser = ArgumentParser()
    parser.add_argument("-e", "--experiment", dest="exp_name",
                        help='Filter experiment instance. Python regex are supported. \
                        E.g. ".*2e-4.*(sphero|youbot)"',
                        default=None)
    return parser.parse_args()

exp = create_all_sstage_experiments()
args = read_command_line_args()
for or_conf in tqdm(exp):
    print("or_conf", or_conf)
    for r in range(or_conf['repetions']): 
        #train_records = get_recursive_file_list(TRAIN_DIR,file_matchers=[or_conf['dataset']])
        train_records = get_recursive_file_list(TRAIN_DIR)
        for t in or_conf['types']:
            for out in ['_cat','_reg','_bin','']:
                conf = deepcopy(or_conf)
                train_instance_name = conf['name']+'_'+t+out
                if not conf['separate_cat_ori'] and out != '':
                    continue
                if conf['separate_cat_ori']:
                    if out == '':
                        continue
                    elif out == '_cat':
                        conf['reg_weight'] = 0.0
                        conf['bin_weight'] = 0.0
                        conf['epochs'] = conf['epochs_cat']
                    elif out == '_reg' and conf['enable_reg']:
                        conf['cat_weight'] = 0.0
                        conf['bin_weight'] = 0.0
                        conf['epochs'] = conf['epochs_reg']
                    elif out == '_bin' and conf['enable_bin']:
                        conf['cat_weight'] = 0.0
                        conf['reg_weight'] = 0.0
                        conf['epochs'] = conf['epochs_bin']
                    else:
                        print('UNKNOWN OUTPUT CONFIG {}'.format(out))
                        continue
                #in str() gecastet - warum hat das bei Lukas funktioniert?
                #if not re.match(args.exp_name, train_instance_name):
                if not re.match(str(args.exp_name), train_instance_name): 
                    #print('Skip '+train_instance_name)
                    continue            
                record_for_type = [e for e in train_records if t in e]
                print("record_for_type: ", len(record_for_type))
                assert len(record_for_type) == 1, \
                       "{} for {} has not one item".format(record_for_type, train_instance_name)
                print("Start training {} for {}".format(train_instance_name,
                                                        record_for_type))
                if not GPU:
                    p = Process(
                      target=train,
                      args=(record_for_type[0], conf, [t], out, r)
                    )
                    p.start()
                else:
                    train(record_for_type[0], conf, [t], out)
