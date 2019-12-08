import os
import PIL
from PIL import Image
from tqdm import tqdm
import tensorflow as tf

from CNNRobotLocalisation.Utils.file_utils import *

################################################################################

PATH = '/home/lhoyer/cnn_robot_localization/data/exp1'
TMP_PATH = '/tmp/TFRecordResize/'
TARGET_W = 400
TARGET_H = 300
os.makedirs(TMP_PATH, exist_ok=True)

################################################################################

for f in get_recursive_file_list(PATH, file_extensions=['.record'], file_excludes=['0x']):
    print(f)
    tf_examples = []
    record_iterator = tf.python_io.tf_record_iterator(f)
    for string_record in tqdm(record_iterator):
        example = tf.train.Example()
        example.ParseFromString(string_record)
        feats = example.features
        img_name = (feats.feature['image/filename'].bytes_list.value[0]).decode('utf8')
        img_enc = (feats.feature['image/encoded'].bytes_list.value[0])
        with tf.gfile.GFile(TMP_PATH+'tmp_img.jpg', 'wb') as fid:
            fid.write(img_enc)
        img = Image.open(TMP_PATH+'tmp_img.jpg').convert('RGB')
        img = img.resize((TARGET_W, TARGET_H), resample=PIL.Image.LANCZOS)
        img.save(TMP_PATH+'tmp_img_scaled.png')

        with tf.gfile.GFile(TMP_PATH+'tmp_img_scaled.png', 'rb') as fid:
            encoded_image_data = fid.read()
        feats.feature['image/height'].int64_list.value[0] = TARGET_H
        feats.feature['image/width'].int64_list.value[0] = TARGET_W
        feats.feature['image/encoded'].bytes_list.value[0] = encoded_image_data
        feats.feature['image/format'].bytes_list.value[0] = b'png'
        tf_examples.append(tf.train.Example(features=feats))

    assert '.record' in f
    out_file = f.replace('.record','_'+str(TARGET_W)+'x'+str(TARGET_H)+'.record')
    with tf.python_io.TFRecordWriter(out_file) as writer:
        for example in tf_examples:
            writer.write(example.SerializeToString())
