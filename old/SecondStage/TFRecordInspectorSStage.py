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

FILE = '/media/data/LocalizationDataNew/Remote/data/Validation/secondstage110balance3/eval_bg_al_sphero.record'
OUT_PATH = '/media/data/LocalizationDataNew/Output/TFRecordInspectorSStage/'
TIMESTAMP = "{:%Y-%m-%d-%H-%M-%S}".format(datetime.now())
OUT_PATH += TIMESTAMP+'/'

#####################################################################

record_iterator = tf.python_io.tf_record_iterator(FILE)

statistics = {}
for e_i, string_record in enumerate(record_iterator):
    example = tf.train.Example()
    example.ParseFromString(string_record)
    feats = example.features
    img_enc = (feats.feature['image/encoded'].bytes_list.value[0])
    class_text = (feats.feature["image/object/subclass/text"].bytes_list.value[0]).decode('utf8')

    if class_text not in statistics:
        statistics[class_text] = 1
    else:
        statistics[class_text] +=1

    os.makedirs(OUT_PATH + class_text, exist_ok=True)
    img = Image.open(io.BytesIO(img_enc))
    img.save(OUT_PATH + class_text + '/img{}.png'.format(e_i))

pprint(statistics)
