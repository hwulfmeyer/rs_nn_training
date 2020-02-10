import os
import numpy as np
import tensorflow as tf
from PIL import Image
from numpy.random import randint
import io
from keras import backend as K
from keras.utils import np_utils
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

ANGLES_PER_BIN = 1
NUM_ORI_BINS = 360

def make_square(im, size, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    scl = size/max(x, y)
    im = im.resize((int(x*scl),int(y*scl)))
    new_im = Image.new('RGB', (size, size), fill_color)
    new_im.paste(im, (int((size - im.size[0]) / 2), int((size - im.size[1]) / 2)))
    return new_im

def angle_diff2(y_true, y_pred):
    return tf.mod(( (y_true - y_pred) + 180 ), 360 ) - 180

def np_angle_diff2(y_true, y_pred):
    return np.mod(( (y_true - y_pred) + 180 ), 360 ) - 180

def angle_mae(y_true, y_pred):
    return K.mean(K.abs(angle_diff2(y_true, y_pred)), axis=-1)

def angle_mse(y_true, y_pred):
    return K.mean(K.square(angle_diff2(y_true, y_pred)), axis=-1)

def angle_bin_mae(y_true, y_pred):
    diff = angle_diff2(K.argmax(y_true)*ANGLES_PER_BIN,
                       K.argmax(y_pred)*ANGLES_PER_BIN)
    return K.mean(K.cast(K.abs(diff), K.floatx()))

def angle_bin_rmse(y_true, y_pred):
    diff = angle_diff2(K.argmax(y_true)*ANGLES_PER_BIN,
                       K.argmax(y_pred)*ANGLES_PER_BIN)
    return K.sqrt(K.mean(K.square(K.cast(K.abs(diff), K.floatx()))))

def data_to_keras(X,Y,Z, num_classes, size=35):
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

"""
img_enc = (feats.feature['image/encoded'].bytes_list.value[0]).decode('utf-8')
img_name = (feats.feature['image/filename'].bytes_list.value[0]).decode('utf-8')
filename = (feats.feature['image/filename'].bytes_list.value[0]).decode('utf-8')
source_id = (feats.feature['image/source_id'].bytes_list.value[0]).decode('utf-8')
width = feats.feature['image/width'].int64_list.value[0]
height = feats.feature['image/height'].int64_list.value[0]
class = (feats.feature['image/object/class/text'].bytes_list.value[0]).decode('utf-8')
class_label = feats.feature['image/object/class/label'].int64_list.value[0]
color = (feats.feature['image/object/subclass/text'].bytes_list.value[0]).decode('utf-8')
color_id = feats.feature['image/object/subclass/label'].int64_list.value[0]
orientation = feats.feature['image/object/pose/orientation'].int64_list.value[0]
"""

def tf_record_load_crops(files,num_per_record=-1,size=35):
    crops, classes, orientations = [],[],[]
    debug_infos = []
    for f in files:
        record_iterator = tf.python_io.tf_record_iterator(f)
        #records = tf.data.TFRecordDataset(f)
        for l, string_record in enumerate(record_iterator):
        #for string_record in records:
            if num_per_record!=-1 and l > num_per_record: break
            example = tf.train.Example()
            example.ParseFromString(string_record)#.numpy())
            feats = example.features

            img_enc = (feats.feature['image/encoded'].bytes_list.value[0])#.decode('utf-8')

            img_name = (feats.feature['image/filename'].bytes_list.value[0]).decode('utf-8')
            img_enc = (feats.feature['image/encoded'].bytes_list.value[0]) #.decode('utf-8')
            width = feats.feature['image/width'].int64_list.value[0]
            height = feats.feature['image/height'].int64_list.value[0]

            #img = Image.frombytes('RGB', (height, width), img_enc)
            img = Image.open(io.BytesIO(img_enc))

            color = (feats.feature["image/object/subclass/text"].bytes_list.value[0]).decode('utf-8') 
            color_id = feats.feature["image/object/subclass/label"].int64_list.value[0]
            orientation = feats.feature["image/object/pose/orientation"].int64_list.value[0]

            crops.append(make_square(img,size))
            classes.append(color_id)
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
                            size=35,
                            class_filters=None):
    crops, classes, orientations = [],[],[]
    debug_infos = []
    for f in files:
        record_iterator = tf.python_io.tf_record_iterator(f)
        for l, string_record in enumerate(record_iterator):
            if num_per_record!=-1 and l > num_per_record: break
            example = tf.train.Example()
            example.ParseFromString(string_record) #.numpy())
            feats = example.features
            width  = feats.feature["image/width"].int64_list.value[0]
            height = feats.feature["image/height"].int64_list.value[0]
            img_name = (feats.feature['image/filename'].bytes_list.value[0]).decode('utf8')
            img_enc = (feats.feature['image/encoded'].bytes_list.value[0])
            img = Image.open(io.BytesIO(img_enc))
            #img = Image.open(open(img_enc, "rb"))
            #img = Image.frombytes('RGB', (height, width), img_enc)

            for i,_ in enumerate(feats.feature["image/object/class/text"].bytes_list.value):
                class_text = (feats.feature["image/object/subclass/text"].bytes_list.value[i]).decode('utf8')
                if not (class_filters == None or any(m in class_text for m in class_filters)):
                    continue
                class_label = feats.feature["image/object/subclass/label"].int64_list.value[i]
                orientation = feats.feature["image/object/pose/orientation"].int64_list.value[i]
                width = feats.feature['image/width'].int64_list.value[0]
                height = feats.feature['image/height'].int64_list.value[0]                
 
                img_resized = img
                if (width != 35 or height != 35): 
                    img_resized = img.resize((35,35), Image.BICUBIC)

                crops.append(make_square(img_resized,size))
                classes.append(class_label)
                orientations.append(orientation)
                debug_infos.append({
                    'filename': img_name,
                    'src_record': f,
                    'crop_num': i*num_derivations,
                })
            img.close()

    return crops, classes, orientations, debug_infos

from keras.callbacks import Callback
import sklearn.metrics as sklm
class TensorBoardCustom(Callback):
    def __init__(self, log_dir='./logs',label_map='', AddCustomMetrics=False):
        super(Callback, self).__init__()
        global tf, projector
        try:
            import tensorflow as tf
            from tensorflow.contrib.tensorboard.plugins import projector
        except ImportError:
            raise ImportError('You need the TensorFlow module installed to use TensorBoard.')

        self.log_dir = log_dir
        self.label_map = label_map
        self.AddCustomMetrics = AddCustomMetrics

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
        y_pred = self.model.predict(self.validation_data[0])
        y_pred = y_pred[0]
        # [0]->cat, [1]->reg, [2]->bin
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

    def log_confusion_matrix(self, epoch):
        # from https://androidkt.com/keras-confusion-matrix-in-tensorboard/
        # and https://stackoverflow.com/questions/37902705/how-to-manually-create-a-tf-summary
        # and https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514#file-tensorboard_logging-py-L41
        y_pred = self.model.predict(self.validation_data[0])
        y_pred = y_pred[0]
        # [0]->cat, [1]->reg, [2]->bin
        y_targ = self.validation_data[1]
        y_targ_onehot = np.argmax(y_targ, axis=1)
        y_pred_onehot = np.argmax(y_pred, axis=1)
        con_mat = tf.Session().run(tf.math.confusion_matrix(labels=y_targ_onehot, predictions=y_pred_onehot))
        con_mat_norm = np.around(con_mat.astype('float') / con_mat.sum(axis=1)[:, np.newaxis], decimals=2)
        classes = ["None"]
        classes.extend([self.label_map[i]['name'] for i in self.label_map.keys()])

        con_mat_df = pd.DataFrame(con_mat_norm,
                            index = classes, 
                            columns = classes)

        figure = plt.figure(figsize=(8, 8))
        sns.heatmap(con_mat_df, annot=True, cmap=plt.get_cmap('Blues'))
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        plt.close(figure)
        img_sum = tf.Summary.Image(encoded_image_string=buf.getvalue())

        summary = tf.Summary(value=[
            tf.Summary.Value(tag="Confusion Matrix", image=img_sum),
        ])
        self.writer.add_summary(summary, epoch)

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
        if self.AddCustomMetrics:
          self.add_custom_metrics(epoch)
          self.log_confusion_matrix(epoch)

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

#if __name__ == '__main__':
#    unittest.main()
