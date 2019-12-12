import os
import numpy as np
import tensorflow as tf
from PIL import Image
from numpy.random import randint
import io
from keras import backend as K
from keras.utils import np_utils

from Utils.file_utils import *

# ANGLES_PER_BIN = 4
# NUM_ORI_BINS = 90 # 360 / 4
ANGLES_PER_BIN = 1
NUM_ORI_BINS = 360


def make_square(im, size, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    scl = size/max(x, y)
    im = im.resize((int(x*scl),int(y*scl)))
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - im.size[0]) / 2), int((size - im.size[1]) / 2)))
    return new_im


# Doesn't converge well
# def angle_diff(y_true, y_pred):
#     a = y_true / 360.0 * np.pi
#     b = y_pred / 360.0 * np.pi
#     diff = tf.atan2(K.sin(a-b), K.cos(a-b))
#     return diff / np.pi * 360.0
# def np_angle_diff(y_true, y_pred):
#     a = y_true / 360.0 * np.pi
#     b = y_pred / 360.0 * np.pi
#     diff = np.arctan2(np.sin(a-b), np.cos(a-b))
#     return diff / np.pi * 360.0

def angle_diff2(y_true, y_pred):
    return tf.mod(( (y_true - y_pred) + 180 ), 360 ) - 180

def np_angle_diff2(y_true, y_pred):
    return np.mod(( (y_true - y_pred) + 180 ), 360 ) - 180

def angle_mae(y_true, y_pred):
    return K.mean(K.abs(angle_diff2(y_true, y_pred)), axis=-1)
def angle_mse(y_true, y_pred):
    return K.mean(K.square(angle_diff2(y_true, y_pred)), axis=-1)
def angle_bin_error(y_true, y_pred):
    diff = angle_diff2(K.argmax(y_true)*ANGLES_PER_BIN,
                       K.argmax(y_pred)*ANGLES_PER_BIN)
    return K.mean(K.cast(K.abs(diff), K.floatx()))

def data_to_keras(X,Y,Z, num_classes, size=128):
    X = [np.asarray(e) for e in X]

    X = np.asarray(X, dtype="uint8")
    X = X.reshape(X.shape[0], size, size, 3)
    X.astype('float32')
    #X /= 255
    Y = np.asarray(Y)
    Y = np_utils.to_categorical(Y, num_classes)
    Z = np.asarray(Z)
    return X,Y,Z

def angle_to_bin(Z):
    Z = np.mod(Z, 360)
    return np_utils.to_categorical(np.floor(Z/ANGLES_PER_BIN),NUM_ORI_BINS)

def tf_record_load_crops(files,num_per_record=-1,size=128):
    crops, classes, orientations = [],[],[]
    debug_infos = []
    for f in files:
        print(f)
        record_iterator = tf.python_io.tf_record_iterator(f)
        for l, string_record in enumerate(record_iterator):
            if num_per_record!=-1 and l > num_per_record: break
            example = tf.train.Example()
            example.ParseFromString(string_record)
            feats = example.features
            img_name = (feats.feature['image/filename'].bytes_list.value[0]).decode('utf8')
            img_enc = (feats.feature['image/encoded'].bytes_list.value[0])
            img = Image.open(io.BytesIO(img_enc))

            assert len(feats.feature["image/object/class/text"].bytes_list.value) == 1
            class_text = (feats.feature["image/object/subclass/text"].bytes_list.value[0]).decode('utf8')
            class_label = feats.feature["image/object/subclass/label"].int64_list.value[0]
            orientation = feats.feature["image/object/pose/orientation"].float_list.value[0]

            crops.append(make_square(img,size))
            classes.append(class_label)
            orientations.append(orientation)
            debug_infos.append({
                'filename': img_name,
                'src_record': f
            })

    assert len(crops) == len(classes) and len(crops) == len(orientations)
    print("Loaded {} crops from {}".format(len(crops),files))

    return crops, classes, orientations, debug_infos

def custom_randint(min, max):
    min = round(min)
    max = round(max)
    if min == max:
        return min
    return randint(min, max)

def tf_record_extract_crops(files, num_derivations,
                            out_var, in_var,
                            num_per_record=-1,
                            size=128,
                            class_filters=None):
    crops, classes, orientations = [],[],[]
    debug_infos = []
    for f in files:
        print(f)
        record_iterator = tf.python_io.tf_record_iterator(f)
        for l, string_record in enumerate(record_iterator):
            if num_per_record!=-1 and l > num_per_record: break
            example = tf.train.Example()
            example.ParseFromString(string_record)
            feats = example.features
            width  = feats.feature["image/width"].int64_list.value[0]
            height = feats.feature["image/height"].int64_list.value[0]
            img_name = (feats.feature['image/filename'].bytes_list.value[0]).decode('utf8')
            img_enc = (feats.feature['image/encoded'].bytes_list.value[0])
            img = Image.open(io.BytesIO(img_enc))

            for i,_ in enumerate(feats.feature["image/object/class/text"].bytes_list.value):
                #print(feats.feature.keys())
                class_text = (feats.feature["image/object/subclass/text"].bytes_list.value[i]).decode('utf8')
                if not (class_filters == None or any(m in class_text for m in class_filters)):
                    continue
                class_label = feats.feature["image/object/subclass/label"].int64_list.value[i]
                orientation = feats.feature["image/object/pose/orientation"].float_list.value[i]
                xmin = round(feats.feature["image/object/bbox/xmin"].float_list.value[i] * width)
                xmax = round(feats.feature["image/object/bbox/xmax"].float_list.value[i] * width)
                ymin = round(feats.feature["image/object/bbox/ymin"].float_list.value[i] * height)
                ymax = round(feats.feature["image/object/bbox/ymax"].float_list.value[i] * height)
                obj_w = xmax - xmin
                obj_h = ymax - ymin
                for j in range(num_derivations):
                    img_crop = img.crop((
                        xmin+custom_randint(-out_var*obj_w, +in_var*obj_w),
                        ymin+custom_randint(-out_var*obj_h, +in_var*obj_h),
                        xmax-custom_randint(-out_var*obj_w, +in_var*obj_w),
                        ymax-custom_randint(-out_var*obj_h, +in_var*obj_h)
                    ))
                    crops.append(make_square(img_crop,size))
                    classes.append(class_label)
                    orientations.append(orientation)
                    debug_infos.append({
                        'filename': img_name,
                        'src_record': f,
                        'crop_num': i*num_derivations + j,
                    })

    return crops, classes, orientations, debug_infos

from keras.callbacks import Callback
import sklearn.metrics as sklm
class TensorBoardCustom(Callback):
    def __init__(self, log_dir='./logs',label_map=''):
        super(Callback, self).__init__()
        global tf, projector
        try:
            import tensorflow as tf
            from tensorflow.contrib.tensorboard.plugins import projector
        except ImportError:
            raise ImportError('You need the TensorFlow module installed to use TensorBoard.')

        self.log_dir = log_dir
        self.label_map = label_map

    def set_model(self, model):
        self.model = model
        if K.backend() == 'tensorflow':
            self.sess = K.get_session()
        self.merged = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter(self.log_dir)

    def write_metric(self, name, value, epoch, namespace=''):
        if namespace != '':
            namespace += '/'
        summary = tf.Summary(value=[
            tf.Summary.Value(tag=namespace+name, simple_value=value),
        ])
        self.writer.add_summary(summary, epoch)

    def add_custom_metrics(self, epoch):
        y_pred, z_pred, z2_pred = self.model.predict(self.validation_data[0])
        # [1] -> Y; [2] -> Z
        y_targ = self.validation_data[1]
        y_targ_onehot = np.argmax(y_targ, axis=1)
        y_pred_onehot = np.argmax(y_pred, axis=1)

        class_prec, class_rec, class_f1, class_support = sklm.precision_recall_fscore_support(
            y_targ_onehot,
            y_pred_onehot,
            labels=[int(e) for e in self.label_map.keys()]
        )
        for i, (prec, rec, f1, sup) in enumerate(zip(class_prec,
                                                   class_rec,
                                                   class_f1,
                                                   class_support)):
            i+=1
            if self.label_map == '':
                label = str(i)
            elif i in self.label_map:
                label = str(i) +' '+ self.label_map[i]['name']
            else:
                label = str(i)
            if sup > 0:
                namespace = "Precision by Category/{}".format(label)
                self.write_metric('Precision', prec, epoch, namespace)
                namespace = "Recall by Category/{}".format(label)
                self.write_metric('Recall', rec, epoch, namespace)
                namespace = "F1 by Category/{}".format(label)
                self.write_metric('F1', f1, epoch, namespace)
            if epoch == 0:
                namespace = "Support by Category/{}".format(label)
                self.write_metric('Support', sup, epoch, namespace)
        # average='weighted'
        # Calculate metrics for each label, and find their average
        # -> balanced
        avg_prec, avg_rec, avg_f1, avg_support = sklm.precision_recall_fscore_support(
            y_targ_onehot,
            y_pred_onehot,
            average='weighted'
        )
        self.write_metric('Average Precision', avg_prec, epoch, 'Average Performance')
        self.write_metric('Average Recall', avg_rec, epoch, 'Average Performance')
        self.write_metric('Average F1', avg_f1, epoch, 'Average Performance')
        self.write_metric('Average Support', avg_support, epoch, 'Average Performance')

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for name, value in logs.items():
            if name in ['batch', 'size']:
                continue
            # summary = tf.Summary()
            # summary_value = summary.value.add()
            # summary_value.simple_value = value.item()
            # summary_value.tag = name
            if 'val' in name:
                group = 'Keras Validation/'
            else:
                group = 'Keras Training/'
            self.write_metric(name, value, epoch, group)

        # self.add_custom_metrics(epoch)

        self.writer.flush()

    def on_train_end(self, _):
        self.writer.close()


import unittest

class TestSecondStageUtils(unittest.TestCase):

    def test_np_angle_diff2(self):
        self.assertEqual(np_angle_diff2(359,52), -53)
        self.assertEqual(np_angle_diff2(52,359), 53)
        self.assertEqual(np_angle_diff2(0,720), 0)

    def test_angle_to_bin(self):
        self.assertEqual(len(angle_to_bin(4)), 90)
        self.assertEqual(angle_to_bin(4)[0], 0)
        self.assertEqual(angle_to_bin(4)[1], 1)
        for i in range(2,90):
            self.assertEqual(angle_to_bin(4)[i], 0)
        self.assertEqual(angle_to_bin(50)[12], 1)
        self.assertEqual(angle_to_bin(360)[0], 1)
        self.assertEqual(angle_to_bin(370)[2], 1)
        self.assertEqual(angle_to_bin(-10)[87], 1)

if __name__ == '__main__':
    unittest.main()
