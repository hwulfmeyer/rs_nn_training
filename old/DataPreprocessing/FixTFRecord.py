import os
import tensorflow as tf

from CNNRobotLocalisation.Utils.file_utils import *

PATH = '/home/lhoyer/cnn_robot_localization/data'


for f in get_recursive_file_list(PATH, file_matchers=['exp'], file_extensions=['.record']):
    print(f)
    tf_examples = []
    record_iterator = tf.python_io.tf_record_iterator(f)
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        feats = example.features

        for i,t in enumerate(feats.feature["image/object/subclass/text"].bytes_list.value):
            if t.decode('utf8') == 'youbot_A':
                print(feats.feature["image/object/subclass/label"].int64_list.value[i])
                feats.feature["image/object/subclass/label"].int64_list.value[i] = 48
        tf_examples.append(tf.train.Example(features=feats))

    assert '.record' in f
    out_file = f
    with tf.python_io.TFRecordWriter(out_file) as writer:
        for example in tf_examples:
            writer.write(example.SerializeToString())
