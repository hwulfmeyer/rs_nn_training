import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/site-packages')

import os
#os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import cv2
import numpy as np
import time
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from threading import Thread
from multiprocessing import Process
from keras.models import load_model
from object_detection.utils import label_map_util

from CNNRobotLocalisation.SecondStage.second_stage_utils import *
from CNNRobotLocalisation.Utils.file_utils import *
from inference_utils import *

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DIR = '/home/lhoyer/cnn_robot_localization/benchmark'

for width, height in [(200,150),(400,300),(800,600)]:
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile('{}/{}_{}_frozen_inference_graph.pb'.format(DIR,width, height), 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')

    with detection_graph.as_default():
        sess = tf.Session()
        tensor_dict = create_tensor_dict()

    times_first_stage = []
    for i in tqdm(range(100)):
        image_np = np.random.rand(1600, 1200, 3) * 255
        start = time.time()
        # The pipeline already contains resizing, but it's slower
        image_np = cv2.resize(image_np, (width,height))
        output_dict = inference(sess,detection_graph,tensor_dict,image_np)
        time_after_fs = time.time()
        # Skip first two inferences which are significantly slower
        if (i > 2): times_first_stage.append(1000*(time_after_fs - start))
    print('First stage average exec time in ms: {} +- {}'.format(np.mean(times_first_stage), np.std(times_first_stage)))
