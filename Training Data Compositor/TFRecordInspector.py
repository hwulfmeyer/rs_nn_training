import os
import io
import PIL
from PIL import Image
from datetime import datetime
import math
import numpy as np
import tensorflow as tf
from pprint import pprint

################################################################################

FILE = "output/SecondStage_X/training_rot3.record"
OUT_PATH = "output/inspector_"
TIMESTAMP = "X" #"{:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
OUT_PATH += TIMESTAMP+"/"
os.makedirs(OUT_PATH, exist_ok=True)
#####################################################################

raw_dataset = tf.data.TFRecordDataset(FILE)

statistics = {}
for raw_record in raw_dataset:
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    feats = example.features
    img_enc = (feats.feature['image/encoded'].bytes_list.value[0])
    img_width = feats.feature['image/width'].int64_list.value[0]
    img_height = feats.feature['image/height'].int64_list.value[0]
    filename = class_text = (feats.feature['image/filename'].bytes_list.value[0]).decode('utf8')
    class_text = (feats.feature['image/object/subclass/text'].bytes_list.value[0]).decode('utf8')
    #class_text = (feats.feature['image/object/bbox/xmin'].float_list.value[0])

    if class_text not in statistics:
        statistics[class_text] = 1
    else:
        statistics[class_text] +=1

    #img = Image.frombytes('RGB', (img_height, img_width), img_enc)
    #img = Image.open(open(img_enc, "rb"))    
    img = Image.open(io.BytesIO(img_enc))
    img.save(OUT_PATH + '/{}'.format(filename))

pprint(statistics)
