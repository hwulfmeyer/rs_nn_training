import numpy as np
import time
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from threading import Thread
from multiprocessing import Process
from object_detection.utils import label_map_util

from CNNRobotLocalisation.SecondStage.second_stage_utils import *
from CNNRobotLocalisation.Utils.file_utils import *
from inference_utils import *

##################################
####### IMPORTANT ################
##################################
# This script is deprecated.
# Use jupyter notebook instead
# as loading the networks takes
# a long time.
##################################


PATH_TO_CKPT = '/home/lhoyer/cnn_robot_localization/training/first_stage_2/ssd_mobilenet2/exp0/default/deploy_00/frozen_inference_graph.pb'
COPTER_ID_NET = '/home/lhoyer/cnn_robot_localization/training_result_models/second_stage_2/sstage_default_copter_cat/2018-06-12-01-27/model-final.h5'
SPHERO_ID_NET = '/home/lhoyer/cnn_robot_localization/training_result_models/second_stage_2/sstage_default_sphero_cat/2018-06-12-04-54/model-final.h5'
YOUBOT_ID_NET = '/home/lhoyer/cnn_robot_localization/training_result_models/second_stage_2/sstage_default_youbot_cat/2018-06-12-06-28/model-final.h5'
COPTER_ROT_NET = '/home/lhoyer/cnn_robot_localization/training_result_models/second_stage_2/sstage_default_copter_bin/2018-06-12-03-45/model-05-0.02.h5'
SPHERO_ROT_NET = '/home/lhoyer/cnn_robot_localization/training_result_models/second_stage_2/sstage_default_sphero_bin/2018-06-12-05-56/model-05-0.00.h5'
YOUBOT_ROT_NET = '/home/lhoyer/cnn_robot_localization/training_result_models/second_stage_2/sstage_default_youbot_bin/2018-06-12-07-14/model-05-0.02.h5'

PATH_TO_LABELS = "/home/lhoyer/cnn_robot_localization/CNNRobotLocalisation/LabelMaps/robot_label_map.pbtxt"
PATH_TO_TEST_IMAGES_DIR = '/home/lhoyer/cnn_robot_localization/data/InferenceTest'
TEST_IMAGE_PATHS = get_recursive_file_list(PATH_TO_TEST_IMAGES_DIR, file_extensions=".jpg")
SAVE_RESULTS=True

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
num_classes = label_map_util.get_max_label_map_index(
                        label_map_util.load_labelmap(PATH_TO_LABELS)) + 1
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
category_index = label_map_util.create_category_index(categories)

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')
sstage_id_model = {
    1: load_sstage_model(COPTER_ID_NET),
    2: load_sstage_model(SPHERO_ID_NET),
    5: load_sstage_model(YOUBOT_ID_NET),
}
sstage_rot_model = {
    1: load_sstage_model(COPTER_ROT_NET),
    2: load_sstage_model(SPHERO_ROT_NET),
    5: load_sstage_model(YOUBOT_ROT_NET),
}

TIMESTAMP = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
if SAVE_RESULTS:
    os.makedirs('/home/lhoyer/cnn_robot_localization/output/Inference/'+TIMESTAMP, exist_ok=True)
i = 0
with detection_graph.as_default():
    sess = tf.Session()
    tensor_dict = create_tensor_dict()
for image_path in tqdm(TEST_IMAGE_PATHS):
    start = time.time()
    image_hr, image_np = import_jpg(image_path)
    time_after_imread = time.time()
    output_dict = inference(sess,detection_graph,tensor_dict,image_np)
    time_after_fs = time.time()

    objects, output_dict = extractFirstStagePredictions(image_hr, output_dict)
    output_dict['sub_classes'] = []
    output_dict['sub_scores'] = []
    output_dict['poses'] = []
    time_after_fs_extr = time.time()
    if len(objects) > 0:
        assert len(objects) == len(output_dict['detection_classes'])
        for o, t in zip(objects, output_dict['detection_classes']):
            prediction = sstage_id_model[t].predict(np.asarray([o]))[0]
            output_dict['sub_classes'].extend(np.argmax(prediction, axis=1))
            output_dict['sub_scores'].extend(np.max(prediction, axis=1))
            # [2] -> use bin orientation estimation
            output_dict['poses'].append(np.argmax(sstage_rot_model[t].predict(np.asarray([o]))[2]))
    else:
        print('NO OBJECT RECOGNIZED')
    time_after_ss = time.time()

    print(output_dict)
    # output_dict = {'poses': [359], 'detection_classes': [1], 'sub_scores': [[0.9999536]], 'detection_scores': [0.7944848], 'sub_classes': [[30]], 'num_detections': 100, 'detection_boxes': [[0.70404077, 0.67665005, 0.78040874, 0.7340714 ]]}

    if SAVE_RESULTS:
        thread = Thread(
            target=save_visualization,
            args=(image_hr, output_dict, category_index,i, TIMESTAMP)
        )
        thread.start()
    i += 1
    # print('Imread exec in ms: {}'.format(1000*(time_after_imread - start)))
    # print('First stage exec in ms: {}'.format(1000*(time_after_fs - time_after_imread)))
    # print('First stage extract in ms: {}'.format(1000*(time_after_fs_extr - time_after_fs)))
    # print('Second stage exec in ms: {}'.format(1000*(time_after_ss - time_after_fs_extr)))
    # print('Total exec in ms: {}'.format(1000*(time.time() - start)))

# Evaluation: see ObjectDetectionEvaluator
