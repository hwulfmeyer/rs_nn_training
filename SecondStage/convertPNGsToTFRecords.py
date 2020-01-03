import tensorflow as tf
import csv
import pandas
import numpy

import sys
sys.path.insert(0, "/home/josi/OvGU/Rolling Swarm/")

from rs_nn_training.Utils.file_utils import get_recursive_file_list

#train
"""
FILE_PATH = "/home/josi/OvGU/Rolling Swarm/data/train/sphero_data.tfrecords"
DATA = "/home/josi/OvGU/Rolling Swarm/SecondStage_X/training/" 
LABEL_FILE = "/home/josi/OvGU/Rolling Swarm/SecondStage_X/training/groundtruth.csv"

"""
#test
FILE_PATH = "/home/josi/OvGU/Rolling Swarm/data/test/sphero_data.tfrecords"
DATA = "/home/josi/OvGU/Rolling Swarm/SecondStage_X/test/" 
LABEL_FILE = "/home/josi/OvGU/Rolling Swarm/SecondStage_X/test/groundtruth.csv"
#"""

def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _float_feature(value):
  """Returns a float_list from a float / double."""
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def _int64_feature(value):
  """Returns an int64_list from a bool / enum / int / uint."""
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
 
files = get_recursive_file_list(DATA, file_extensions=".png")

col_names = ['image_name', 'image_class_str', 'image_class', 'rot', 'scl', 'brtn', 'pos0', 'pos1']
data = pandas.read_csv(LABEL_FILE, names=col_names)

image_name = data.image_name.tolist()
image_class_str = data.image_class_str.tolist()
image_class = data.image_class.tolist()
rot = data.rot.tolist()
scl = data.scl.tolist()
brtn = data.brtn.tolist()
pos0 = data.pos0.tolist()
pos1 = data.pos1.tolist()

writer = tf.io.TFRecordWriter(FILE_PATH)

for j in range(len(files)):

    feature = {'image/encoded': _bytes_feature(tf.compat.as_bytes(files[j])), 
               'image/filename': _bytes_feature(tf.compat.as_bytes(image_name[j])), 
               'image/object/class/text': _bytes_feature(tf.compat.as_bytes("sphero")),
               'image/object/subclass/text': _bytes_feature(tf.compat.as_bytes(image_class_str[j])),
                'image/object/subclass/label': _int64_feature(image_class[j]),
                'image/object/pose/orientation': _int64_feature(rot[j]),
                'scl': _float_feature(scl[j]),
                'brtn': _float_feature(brtn[j]),
                'pos0': _int64_feature(pos0[j]),
                'pos1': _int64_feature(pos1[j])
              }


    example = tf.train.Example(features = tf.train.Features(feature=feature))

    writer.write(example.SerializeToString())

writer.close()
         
         

         
    


