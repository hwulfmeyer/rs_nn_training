import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    print('cv2 dist-packages not in PATH')

import numpy as np
import time
from PIL import Image
import tensorflow as tf

from inference_utils import *

class TwoStageFramework():
    def __init__(self,network_files):
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(network_files['FIRST_STAGE'], 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')
            self.sess = tf.Session()
            self.tensor_dict = create_tensor_dict()
        self.sstage_id_model = {
            1: load_sstage_model(network_files['COPTER_ID_NET']),
            2: load_sstage_model(network_files['SPHERO_ID_NET']),
            5: load_sstage_model(network_files['YOUBOT_ID_NET']),
        }
        self.sstage_rot_model = {
            1: load_sstage_model(network_files['COPTER_ROT_NET']),
            2: load_sstage_model(network_files['SPHERO_ROT_NET']),
            5: load_sstage_model(network_files['YOUBOT_ROT_NET']),
        }


    def infer(self,image_hr, min_score_thresh):
        image_np = load_image_into_numpy_array(image_hr.resize((400,300)))
        output_dict = inference(self.sess,self.detection_graph,self.tensor_dict,image_np)

        objects, output_dict = extractFirstStagePredictions(image_hr, output_dict, min_score_thresh)
        output_dict['sub_classes'] = []
        output_dict['sub_scores'] = []
        output_dict['poses'] = []
        if len(objects) > 0:
            assert len(objects) == len(output_dict['super_classes'])
            for o, t in zip(objects, output_dict['super_classes']):
                prediction = self.sstage_id_model[t].predict(np.asarray([o]))[0]
                output_dict['sub_classes'].extend(np.argmax(prediction, axis=1))
                output_dict['sub_scores'].extend(np.max(prediction, axis=1))
                # [2] -> use bin orientation estimation
                output_dict['poses'].append(np.argmax(self.sstage_rot_model[t].predict(np.asarray([o]))[2]))
        else:
            print('NO OBJECT RECOGNIZED')

        return output_dict
