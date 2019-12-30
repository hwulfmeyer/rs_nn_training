"""
- der compositor benutzt alle crops im Pfad CROP_PATH um die trainingsdaten zu erstellen
    - zusätzlich zu den unterpfaden 'training' und 'test'
- die crops müssen einen runden cut mit alpha channel besitzen (als PNG Datei)
    => verhindert, dass das netz einfach die ränder der crops lernt
- der Dateiname der crops muss so aussehen: YX.png
    - Y = klasse aus 'sphereo_classes' dictionary X=nummerierung für die crops
    - z.B. bright_blue0.png, bright_blue1.png, ...
"""

import csv
import datetime
from os import listdir, makedirs
from os.path import isfile, join
import io
import tensorflow as tf
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import decimal

np.random.seed(146324)

object_classes = {
    "sphero": 1
}

sphereo_classes = {
    "bright_blue": 1,
    "bright_red": 2,
    "bright_green": 3,
    "bright_white": 4,
    "dark_blue": 5,
    "dark_green": 6,
    "dark_red": 7
}

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

"""
- SHAPE = (height, width, 3)
- the images have a random brightness
"""
def generate_gaussiannoiseimg(SHAPE = (1200, 1600, 3), brtn = None):
    noise = np.random.randint(0, 255, SHAPE)
    noise = noise.astype(dtype=np.uint8)
    img = Image.fromarray(noise, mode='RGB')
    img = ImageEnhance.Color(img).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    # random brightness
    if brtn is None:
        brtn = np.random.uniform(0.1, 0.3)
    img = ImageEnhance.Brightness(img).enhance(brtn)
    return img


def generate_random_image(img, bg, rot=None):
    # random scale
    scl = round(np.random.uniform(0.9, 1.1), 2)
    img = img.resize((int(scl*img.width),int(scl*img.height)), resample=Image.LANCZOS)

    # random rotation
    if rot is None:
        rot = np.random.randint(360)
    img = img.rotate(rot, resample=Image.BICUBIC, expand=False)

    # random brightness
    brtn = round(np.random.uniform(0.5, 1.5), 2)
    img = ImageEnhance.Brightness(img).enhance(brtn)

    # random position
    # !position is the upper left corner of the crop in the picture!
    pos = (np.random.randint(0, bg.width-img.width), np.random.randint(0, bg.height-img.height))
    bg.paste(img, pos, img)
    return bg, img, pos


def writecsv(filename, csv_rows):
    with open(filename, 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            spamwriter.writerow(row)

def writetfrecord(filename, tf_examples):
    writer = tf.io.TFRecordWriter(filename)
    for example in tf_examples:
        writer.write(example.SerializeToString())
    writer.close()

"""
first stage: objection detection (to learn a bounding box for the objects)
- random rgb noise backgrounds mit random brightness in Größe 300x300
- auf die backgrounds werden die sphero crops superimposed
- die crops haben eine random scale, rotation, brightness und position im background
- die CSV datei enthält folgende spalten:
	[image_name, width, height, object_class, xmin, xmax, ymin, ymax, img_class_str, img_class, rot, scl, brtn]
	image_name: vollständige Name der Bilddatei
    width, height: Weite und Höhe des bilds
    object_class: objektklasse (z.b. 'sphero', 'copter', ...)
	xmin, xmax, ymin, ymax: sind die koordinaten der bounding box
	img_class_str: ist der name der klasse (siehe 'sphereo_classes' in compositor.py)
	img_class: ist die ID der klasse (siehe 'sphereo_classes' in compositor.py)
	rot: rotationswinkel
	scl: scalingfaktor
	brtn: brightnessfaktor
TODO:
    - mehrere objekte in einem bild ?
"""
def firststage(isTrainingData):
    TRAIN_SIZE = 300
    TEST_SIZE = 100
    BGHEIGHT = 300
    BGWIDTH = 300
    FOLDER = "evaluation"
    SIZE = TEST_SIZE
    if isTrainingData:
        FOLDER = "training"
        SIZE = TRAIN_SIZE
    OUT_PATH = "output/"
    CROP_PATH = "crops/" + FOLDER + "/"
    timestamp = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    timestamp = "X"
    IMG_OUT_PATH = OUT_PATH+"FirstStage_"+timestamp+"/"  + FOLDER + "/"
    TFREC_OUT_PATH = OUT_PATH+"FirstStage_"+timestamp+"/" + FOLDER
    makedirs(IMG_OUT_PATH, exist_ok=True)

    CROPS = [f for f in listdir(CROP_PATH) if isfile(join(CROP_PATH, f))]
    csv_rows = []
    tf_examples = []

    for i in tqdm(range(SIZE)):
        bg = generate_gaussiannoiseimg(SHAPE = (BGHEIGHT, BGWIDTH, 3))
        crop = np.random.choice(CROPS)
        img = Image.open(CROP_PATH + crop)

        bg, img, pos = generate_random_image(img, bg)

        obj_class_str = "sphero"
        obj_class = object_classes.get(obj_class_str)

        image_name = str(i) +  ".png"
        xmin = pos[0]
        xmax = xmin+img.width
        ymin = pos[1]
        ymax = ymin+img.height
        csv_rows.append([image_name, bg.width, bg.height, obj_class_str, obj_class, xmin, xmax, ymin, ymax])
        tf_example = tf.train.Example(features=tf.train.Features(feature={      
            'image/encoded': bytes_feature(bg.tobytes()),
            'image/format': bytes_feature(b'png'),
            'image/filename': bytes_feature(image_name.encode('utf8')),
            'image/source_id': bytes_feature(image_name.encode('utf8')),
            'image/width': int64_feature(bg.width),
            'image/height': int64_feature(bg.height),
            'image/object/class/text': bytes_feature(obj_class_str.encode('utf8')),
            'image/object/class/label': int64_feature(obj_class),
            'image/object/bbox/xmin': int64_feature(xmin),
            'image/object/bbox/xmax': int64_feature(xmax),
            'image/object/bbox/ymin': int64_feature(ymin),
            'image/object/bbox/ymax': int64_feature(ymax),
        }))
        tf_examples.append(tf_example)
        bg.save(IMG_OUT_PATH+image_name)

    writecsv(IMG_OUT_PATH+"labels.csv", csv_rows)
    writetfrecord(TFREC_OUT_PATH+".tfrecords", tf_examples)



"""
second stage: Identification CNN and Orientation CNN
=> Identification CNN: hohe vielfalt an farben, rotationen und helligkeiten
=> Orientation CNN: hohe vielfalt an rotationen mit unterschiedlichen farben und helligkeiten

- random rgb noise backgrounds mit random brightness in Größe 35x35
- die crops haben die größe 30x30
- jeder crop wird zwei mal mit jedem winkel erstellt
- zusätzlich erhalten die crops eine random scale, random helligkeit und random position im background
- die CSV datei enthält folgende spalten:
	[image_name, width, heigth, object_class, img_class_str, img_class, rot, scl, brtn, pos[0], pos[1]])
	image_name: vollständige Name der Bilddatei
    width, height: Weite und Höhe des bilds
    object_class: objektklasse (z.b. 'sphero')
	img_class_str: ist der name der klasse (siehe 'sphereo_classes' in compositor.py)
	img_class: ist die ID der klasse (siehe 'sphereo_classes' in compositor.py)
	rot: rotationswinkel
	scl: scalingfaktor
	brtn: brightnessfaktor
	pos[0] und pos[1]: position der crops im bg (ist der linke obere pixel vom crop im bg)

TODO:
    - create 5 different crops (e.g. each corner and the middle) per color for the training data set
    - create 2 different crops (different from the training) per color for the test data set
    - add noise to crops?
"""
def secondstage(isTrainingData):

    BGHEIGHT = 35
    BGWIDTH = 35
    FOLDER = "evaluation"
    if isTrainingData:
        FOLDER = "training"
    OUT_PATH = "output/"
    CROP_PATH = "crops/" + FOLDER + "/"
    timestamp = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    timestamp = "X"
    IMG_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/"  + FOLDER + "/"
    TFREC_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/" + FOLDER
    makedirs(IMG_OUT_PATH, exist_ok=True)
    
    CROPS = [f for f in listdir(CROP_PATH) if isfile(join(CROP_PATH, f))]
    csv_rows = []
    tf_examples = []

    for crop in tqdm(CROPS):
        for rot in range(360):
            for k in range (2):
                img = Image.open(CROP_PATH + crop)
                bg = generate_gaussiannoiseimg(SHAPE = (BGHEIGHT, BGWIDTH, 3))

                bg, img, pos = generate_random_image(img, bg, rot)

                # save image and csv
                obj_class_str = "sphero"
                obj_class = object_classes.get(obj_class_str)
                img_class_str = crop[:-5]
                img_class = sphereo_classes.get(img_class_str)
                image_name = img_class_str + crop[-5] + "_r" + str(rot) + "_" + str(k) + ".png"
                bg.save(IMG_OUT_PATH+image_name)
                csv_rows.append([image_name, bg.width, bg.height, img_class_str, img_class, rot, pos[0], pos[1]])

                tf_example = tf.train.Example(features=tf.train.Features(feature={      
                    'image/encoded': bytes_feature(bg.tobytes()),
                    'image/format': bytes_feature(b'png'),
                    'image/filename': bytes_feature(image_name.encode('utf8')),
                    'image/source_id': bytes_feature(image_name.encode('utf8')),
                    'image/width': int64_feature(bg.width),
                    'image/height': int64_feature(bg.height),
                    'image/object/class/text': bytes_feature(obj_class_str.encode('utf8')),
                    'image/object/class/label': int64_feature(obj_class),
                    'image/object/subclass/text': bytes_feature(img_class_str.encode('utf8')),
                    'image/object/subclass/label': int64_feature(img_class),
                    'image/object/pose/orientation': int64_feature(rot),

                }))
                tf_examples.append(tf_example)

    writecsv(IMG_OUT_PATH+"labels.csv", csv_rows)
    writetfrecord(TFREC_OUT_PATH+".tfrecords", tf_examples)


if __name__ == "__main__":
    img = generate_gaussiannoiseimg(SHAPE = (500, 500, 3), brtn=1.0)
    img.save("gaussian_noise.png")

    firststage(True)
    firststage(False)
    secondstage(True)
    secondstage(False)
