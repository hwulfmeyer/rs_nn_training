import os, datetime, csv
from enum import Enum
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
from copy import deepcopy
import shutil
import tarfile
import tensorflow as tf
from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from CNNRobotLocalisation.Utils.file_utils import *
from compositor import *
from experiment_definitions import *

################################################################################

OUT_PATH = "/home/lhoyer/cnn_robot_localization/output/compositor2/"
PATH_TO_LABELS = "/home/lhoyer/cnn_robot_localization/CNNRobotLocalisation/LabelMaps/robot_label_map.pbtxt"
OBJECT_TAR_PATH = "/home/lhoyer/cnn_robot_localization/data/CompositorInputData/RobotCrops/TrainData1"
BACKGROUND_TAR_PATH = "/home/lhoyer/cnn_robot_localization/data/CompositorInputData/Backgrounds"
OBJECT_PATH = '/home/lhoyer/cnn_robot_localization/data/tmp/RobotCrops'
BACKGROUND_PATH = '/home/lhoyer/cnn_robot_localization/data/tmp/Backgrounds'
STAGES = ['SecondStage']
labelMapDict = label_map_util.get_label_map_dict(PATH_TO_LABELS)

################################################################################

timestamp = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.datetime.now())

def untar():
    # Untar to tmp dir
    assert 'tmp' in BACKGROUND_PATH
    assert 'tmp' in OBJECT_PATH
    shutil.rmtree(BACKGROUND_PATH,ignore_errors=True)
    shutil.rmtree(OBJECT_PATH,ignore_errors=True)
    for f in tqdm(get_recursive_file_list(OBJECT_TAR_PATH)):
        with tarfile.open(f) as tar:
            tar.extractall(f.replace(OBJECT_TAR_PATH,OBJECT_PATH).rsplit('/',1)[0])
    for f in tqdm(get_recursive_file_list(BACKGROUND_TAR_PATH,file_excludes=["COCO2.tar"])):
        with tarfile.open(f) as tar:
            tar.extractall(f.replace(BACKGROUND_TAR_PATH,BACKGROUND_PATH).rsplit('/',1)[0])

def compile(set_name, amount, conf, export_arena=True, export_crop=False,
            out_size_w = 1600, out_size_h = 1200):
    typeObjectList = generate_balanced_object_list(OBJECT_PATH,conf)
    backgroundFileList = generate_balanced_background_list(BACKGROUND_PATH,conf)
    tf_examples = []
    IMG_OUT_PATH = OUT_PATH+set_name+'-imgs/'
    os.makedirs(IMG_OUT_PATH, exist_ok=True)

    # about 3 min for 1000 images
    for i in tqdm(range(amount)):
        tf_example = composite(set_name,i,
                               backgroundFileList,
                               typeObjectList,
                               conf,
                               labelMapDict,
                               IMG_OUT_PATH,
                               out_size_w = out_size_w,
                               out_size_h = out_size_h,
                               export_crop = export_crop,
                               export_arena = export_arena)
        tf_examples.extend(tf_example)
    writer = tf.python_io.TFRecordWriter(OUT_PATH+set_name+".record")
    for example in tf_examples:
        writer.write(example.SerializeToString())
    writer.close()

#untar()
# exp_configs = create_all_experiments()
exp_configs = [create_lighting_experiment()]
for exp_config in exp_configs:
    for conf in exp_config:
        set_name = conf['name']
        print(set_name)
        if conf['skip_compositing']:
            print('Skipped compositing')
            continue
        if 'FirstStage' in STAGES:
            compile(set_name,conf['compositor_amount'],conf)
        if 'SecondStage' in STAGES:
            for t in  conf['robot_type']:
                sconf = deepcopy(conf)
                sconf['robot_type'] = t

                if t == "copter": num_sub = 15+3
                elif t == "sphero": num_sub = 8
                elif t == "youbot": num_sub = 6
                else: raise Exception("type not supported")
                # 1000 per category for amount == 5000
                # and each image has 2 crops
                sstage_amount = round(sconf['compositor_amount'] / 5 * num_sub)
                compile(set_name+"_crop_"+t,sstage_amount,sconf, False, True)
