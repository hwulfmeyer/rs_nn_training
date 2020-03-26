import numpy as np
import tensorflow as tf
import glob

from PIL import Image

import cv2

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = '/.label_map.pbtxt'

NUM_CLASSES = 1

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

with detection_graph.as_default():
    with tf.Session(graph=detection_graph) as sess:
        filelist = glob.glob('images/*.png')
        for file in filelist:
            image_np = np.array(Image.open(file))
            # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            scores = detection_graph.get_tensor_by_name('detection_scores:0')
            classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # Actual detection.
            (boxes, scores, classes, num_detections) = sess.run(
                [boxes, scores, classes, num_detections],
                feed_dict={image_tensor: image_np_expanded})
            boxes_sq = np.squeeze(boxes)
            scores_sq = np.squeeze(scores)
            num_detections_sq = np.squeeze(num_detections)

            # Draw bounding boxes
            color = np.array([0, 255, 0], dtype=np.uint8)
            for i in range(0, int(num_detections_sq) - 1):
                if scores_sq[i] > 0.5:
                    box = boxes_sq[i, 0]
                    bounding_box = (int(boxes_sq[i, 1] * image_np.shape[1]),
                                    int(boxes_sq[i, 0] * image_np.shape[0]),
                                    int((boxes_sq[i, 3] - boxes_sq[i, 1]) * image_np.shape[1]),
                                    int((boxes_sq[i, 2] - boxes_sq[i, 0]) * image_np.shape[0]))

                    image_np[bounding_box[1], bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
                    image_np[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0]] = color

                    image_np[bounding_box[1] + bounding_box[3], bounding_box[0]:bounding_box[0] + bounding_box[2]] = color
                    image_np[bounding_box[1]:bounding_box[1] + bounding_box[3], bounding_box[0] + bounding_box[2]] = color

            cv2.imshow('object detection', cv2.resize(image_np, (800, 600)))
            cv2.waitKey()