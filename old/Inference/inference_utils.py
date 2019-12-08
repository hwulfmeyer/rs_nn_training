import sys
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
except:
    print('cv2 dist-packages not in PATH')

import tensorflow as tf
import numpy as np
import math
import cv2
from keras.models import load_model
from keras.applications.mobilenet import relu6
from object_detection.utils import visualization_utils as vis_util
from CNNRobotLocalisation.SecondStage.second_stage_utils import *

def load_sstage_model(file):
    return load_model(file,
                      custom_objects={
                      'relu6': relu6,
                      #'DepthwiseConv2D': mobilenet.DepthwiseConv2D,
                      'angle_mse': angle_mse,
                      'angle_mae': angle_mae,
                      'angle_bin_error': angle_bin_error})

def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)

def import_jpg(image_path):
    image_hr = Image.open(image_path).convert('RGB')
    image = image_hr.resize((400,300))
    image_np = load_image_into_numpy_array(image)
    return image_hr, image_np

def create_tensor_dict():
    # Get handles to input and output tensors
    ops = tf.get_default_graph().get_operations()
    all_tensor_names = {output.name for op in ops for output in op.outputs}
    tensor_dict = {}
    for key in [
            'num_detections', 'detection_boxes', 'detection_scores',
            'detection_classes', 'detection_masks'
    ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
    return tensor_dict

class PredictionRowDistribution:
    def __init__(self):
        self.pos_cnt, self.neg_cnt = 0,0
        self.pos_in_row, self.neg_in_row = [],[]
    def add_prediction(self,correct):
        if correct:
            if self.neg_cnt != 0:
                self.neg_in_row.append(self.neg_cnt)
            self.neg_cnt = 0
            self.pos_cnt += 1
        else:
            if self.pos_cnt != 0:
                self.pos_in_row.append(self.pos_cnt)
            self.pos_cnt = 0
            self.neg_cnt += 1
        print('Right classification {}, pos in row {}, neg in row {}'.format(
            correct, self.pos_in_row, self.neg_in_row))
    def finalize(self):
        if self.neg_cnt != 0:
            self.neg_in_row.append(self.neg_cnt)
            self.neg_cnt = 0
        if self.pos_cnt != 0:
            self.pos_in_row.append(self.pos_cnt)
            self.pos_cnt = 0

def inference(sess, graph, tensor_dict, image):
    image_tensor = graph.get_tensor_by_name('image_tensor:0')

    # Run inference
    output_dict = sess.run(tensor_dict,
       feed_dict={image_tensor: np.expand_dims(image, 0)})

    # all outputs are float32 numpy arrays, so convert types as appropriate
    output_dict['num_detections'] = int(output_dict['num_detections'][0])
    output_dict['detection_classes'] = output_dict[
            'detection_classes'][0].astype(np.uint8)
    output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
    output_dict['detection_scores'] = output_dict['detection_scores'][0]
    if 'detection_masks' in output_dict:
        output_dict['detection_masks'] = output_dict['detection_masks'][0]
    return output_dict

def extractFirstStagePredictions(image, output_dict, min_score_thresh=0.5):
    boxes = output_dict['detection_boxes']
    classes = output_dict['detection_classes']
    scores = output_dict['detection_scores']
    extracted_boxes, extracted_classes, extracted_scores = [],[],[]
    objects = []
    height, width = image.height, image.width
    for i in range(boxes.shape[0]):
        if scores[i] > min_score_thresh:
            box = tuple(boxes[i].tolist())
            y0=int(box[0]*height)
            x0=int(box[1]*width)
            y1=int(box[2]*height)
            x1=int(box[3]*width)
            obj = image.crop((x0, y0, x1, y1))
            obj = make_square(obj, 128)
            obj = np.asarray(obj)
            objects.append(obj)
            extracted_boxes.append(boxes[i])
            extracted_classes.append(classes[i])
            extracted_scores.append(scores[i])
    output_dict['detection_boxes'] = np.asarray(extracted_boxes)
    output_dict['super_classes'] = np.asarray(extracted_classes)
    output_dict['super_scores'] = np.asarray(extracted_scores)
    return objects, output_dict

def visualize_poses(image, boxes, poses, arrow_length = 100, arrow_width = 4):
    for i in range(boxes.shape[0]):
        box = tuple(boxes[i].tolist())
        cy = int((box[0]+box[2])/2 * image.shape[0])
        cx = int((box[1]+box[3])/2 * image.shape[1])
        theta = poses[i]
        xOff = math.sin(theta*np.pi/180)*arrow_length
        yOff = math.cos(theta*np.pi/180)*arrow_length
        cv2.arrowedLine(image,
                    (int(cx),int(cy)),
                    (int(cx-xOff),int(cy-yOff)),
                    (255,0,0),arrow_width)
        #draw = ImageDraw.Draw(image)
        #draw.line((cx,cy, int(cx-xOff),int(cy-yOff)), fill=128)

def save_visualization(image, output_dict, category_index, i, TIMESTAMP):
    image_np = np.array(image)
    #image_np = load_image_into_numpy_array(image)
    vis_util.visualize_boxes_and_labels_on_image_array(
               image_np,
               output_dict['detection_boxes'],
               output_dict['sub_classes'],
               output_dict['sub_scores'],
               category_index,
               use_normalized_coordinates=True,
               line_thickness=2)
    if 'poses' in output_dict:
        visualize_poses(image_np, output_dict['detection_boxes'], output_dict['poses'])
    image_pil = Image.fromarray(np.uint8(image_np)).convert('RGB')
    with tf.gfile.Open('/home/lhoyer/cnn_robot_localization/output/Inference/'+TIMESTAMP+'/'+str(i)+'.jpg', 'w') as fid:
        image_pil.save(fid, 'JPEG')
