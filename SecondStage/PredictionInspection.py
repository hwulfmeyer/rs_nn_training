import os
from datetime import datetime
from keras.applications import mobilenet_v2 as mobilenet
from keras.models import load_model
from keras import backend as K
from PIL import Image

import numpy as np
from keras.layers import DepthwiseConv2D, ReLU
from keras.losses import CategoricalCrossentropy
from keras.metrics import Accuracy
#from keras.utils.generic_utils import CustomObjectScope

import sys
#path des dir in dem rs_nn_training liegt
sys.path.insert(0, "/home/josi/OvGU/Rolling Swarm/")

from object_detection.utils import label_map_util

from rs_nn_training.Utils.file_utils import *
from second_stage_utils import *

# disable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

#MODEL = '/home/josi/OvGU/Rolling Swarm/output/5_pos/reg/rot6/b=4096/2020-01-16-08-05-r1/model-final.h5'
MODEL = '/home/josi/OvGU/Rolling Swarm/rs_nn_training/SecondStage/pb_models/old_second_stage/new_labels/cat/model-final.h5'
OUT_PATH = '/home/josi/OvGU/Rolling Swarm/output/inference/13_colours_lab/'
TRAIN_RECORD = '/home/josi/OvGU/Rolling Swarm/data/13_colours_lab/training_rot9_9colors.record'
EVAL_DIR = '/home/josi/OvGU/Rolling Swarm/data/13_colours_lab/eval/'
LABEL_MAP_PATH = '/home/josi/OvGU/Rolling Swarm/rs_nn_training/SecondStage/robot_label_map_komplett.pbtxt'
TRAIN_OR_VAL = 'val'
#MODE = "regression"
MODE = "classification"
TYPES = ['sphero']
IMG_SIZE = 35

SAVE_RESULTS=True
TIMESTAMP = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
OUT_PATH += TIMESTAMP+'/'
os.makedirs(OUT_PATH, exist_ok=True)

label_map = label_map_util.create_category_index_from_labelmap(LABEL_MAP_PATH)
num_classes = label_map_util.get_max_label_map_index(
                        label_map_util.load_labelmap(LABEL_MAP_PATH)) + 1

if TRAIN_OR_VAL == 'train':
    X,Y,Z,_ = tf_record_load_crops([TRAIN_RECORD])
    X_val, Y_val, Z_val = data_to_keras(X,Y,Z,num_classes,IMG_SIZE)
else:
    eval_records = get_recursive_file_list(EVAL_DIR) 
    print(eval_records)
    X,Y,Z,D = tf_record_extract_crops(eval_records, 1, 0.0, 0.0) 
    X_val, Y_val, Z_val = data_to_keras(X,Y,Z,num_classes,IMG_SIZE)
#X_val

print('********************************************************************')
print(X_val)
print(Y_val)
print(Z_val)

#X_val
second_stage_model = load_model(MODEL,
                   custom_objects={
                   'relu6': ReLU,
                   'DepthwiseConv2D': DepthwiseConv2D,
                   'angle_mse': angle_mse,
                   'angle_mae': angle_mae,
                   'angle_bin_error': angle_bin_error})
                   #loss: CategoricalCrossentropy,
                   #metrics: Accuracy})

predictions = second_stage_model.predict(np.array(X_val))
#predictions = second_stage_model.predict(X_val)
cat = np.argmax(predictions[0],axis=1)
cat_score = np.max(predictions[0],axis=1)
ori = predictions[1]

if MODE == "classification":
    for i in range(len(X_val)):
        if cat[i] == Y[i]:
            #print(i)
            continue
        img = Image.fromarray(X_val[i])
        pred_label = label_map[cat[i]]['name']
        gt_label = label_map[Y[i]]['name']

        img.save(OUT_PATH+'{:03d}-pred-{}-gt-{}.png'.format(i,pred_label,gt_label))
if MODE == "regression":
    num_in_tol = 0
    for i in range(len(X_val)):
        img = Image.fromarray(X_val[i])
        pred = ori[i]
        gt = Z[i]
        diff = np_angle_diff2(gt,pred)
        if (abs(diff) < 1.5): num_in_tol += 1
        if (abs(diff) > 3):
            print('Image {} diff of {} and {} is {}'.format(i, pred, gt, diff))
            img.save(OUT_PATH+'{:03d}-pred-{}-gt-{}.png'.format(i,pred,gt))
    print(num_in_tol / len(X_val))
    # Doesn't work.. Why?
    #print('MAE: {}'.format(np.mean(np.abs(np_angle_diff2(Z,ori)))))
