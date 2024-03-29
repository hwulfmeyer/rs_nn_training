"""
- der compositor benutzt alle crops im Pfad CROP_PATH um die trainingsdaten zu erstellen
    - zusätzlich zu den unterpfaden 'training' und 'validation'
- die crops müssen einen runden cut mit alpha channel besitzen (als PNG Datei)
    => verhindert, dass das netz einfach die ränder der crops lernt
- der Dateiname der crops muss so aussehen: Yxx.png
    - Y = klasse aus 'sphero_classes' dictionary X=nummerierung für die crops
    - z.B. bright_blue01.png, bright_blue02.png, ...
"""

# Commented out IPython magic to ensure Python compatibility.
# %tensorflow_version 1.x

import csv
import datetime
import decimal
import copy
import io
from os import listdir, makedirs
from os.path import isfile, join
import random
import numpy as np
import PIL
import tensorflow as tf
from PIL import Image, ImageEnhance
from tqdm import tqdm, trange
import math

np.random.seed(146324)

print(tf.__version__)

object_classes = {
    "sphero": 2
}


sphero_11_classes = {
    "red": 1,
    "orange": 2,
    "yellow": 3,
    "lime_green": 4,
    "magenta": 5, 
    "purple": 6,
    "green": 7,
    "light_green": 8,
    "blue_green": 9,
    "light_blue": 10,
    "blue": 11,
}

sphero_9_classes = {
    "red": 1,
    "yellow": 2,
    "lime_green": 3,
    "magenta": 4, 
    "purple": 5,
    "green": 6,
    "blue_green": 7,
    "light_blue": 8,
    "blue": 9,
}

sphero_13_classes = {
    "red": 1,
    "yellow": 2,
    "lime_green": 3,
    "magenta": 4, 
    "purple": 5,
    "green": 6,
    "blue_green": 7,
    "light_blue": 8,
    "blue": 9,
    "dark_blue": 10,
    "dark_red": 11,
    "dark_green": 12,
    "white": 13
}

sphero_13_classes_lab = {
    "red": 35,
    "yellow": 36,
    "lime_green": 37,
    "magenta": 38, 
    "purple": 39,
    "green": 40,
    "blue_green": 41,
    "light_blue": 42,
    "blue": 49,
    "dark_blue": 50,
    "dark_red": 51,
    "dark_green": 52,
    "white": 53
}

sphero_classes = sphero_13_classes_lab


def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

def float_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

def float_list_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))


"""
- SHAPE = (height, width, 3)
- the images have a random brightness
"""
def generate_gaussiannoiseimg(SHAPE = (1200, 1600, 3), brtn = None, out_width=1600):
    scl_factor = out_width/1600
    noise = np.random.randint(0, 5, SHAPE)
    noise = noise.astype(dtype=np.uint8)
    img = Image.fromarray(noise, mode='RGB')
    img = img.resize((int(math.ceil(img.width*scl_factor)),int(math.ceil(img.height*scl_factor))), resample=Image.LANCZOS)
    return img

def generate_random_image(img, bg, edge_distance=0, rot=None, out_width=1600, withRandomMirroring=True):
    # random horizontal flip
    if withRandomMirroring and bool(random.getrandbits(1)):
      img = img.transpose(PIL.Image.FLIP_LEFT_RIGHT)

    # random scale
    scl_factor = out_width/1600
    img = copy.deepcopy(img)
    bg = copy.deepcopy(bg)
    scl = round(np.random.uniform(0.8, 1.2), 2)
    img = img.resize((int(scl*img.width*scl_factor),int(scl*img.height*scl_factor)), resample=Image.NEAREST)

    # random rotation
    if rot is None:
        rot = np.random.randint(360)
    img = img.rotate(rot, resample=Image.BICUBIC, expand=False)

    # random brightness
    brtn = round(np.random.uniform(0.8, 1.2), 2)
    img = ImageEnhance.Brightness(img).enhance(brtn)

    # random position
    # !position is the upper left corner of the crop in the picture!
    assert bg.width-img.width - edge_distance >= 0, "Abstand zum Rand ist zu hoch eingestellt!"
    assert bg.height-img.height - edge_distance >= 0, "Abstand zum Rand ist zu hoch eingestellt!"
    pos = (np.random.randint(edge_distance, bg.width-img.width-edge_distance), np.random.randint(edge_distance, bg.height-img.height-edge_distance))
    bg.paste(img, pos, img)
    return bg, img, pos, brtn, rot, scl


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

#img = generate_gaussiannoiseimg(SHAPE = (500, 500, 3), brtn=1.0)
#img.save("gaussian_noise.png")

"""first stage: objection detection (to learn a bounding box for the objects)
- random rgb noise backgrounds mit random brightness in Größe AxB
- auf die backgrounds werden die sphero crops superimposed
- die crops haben eine random scale, rotation, brightness und position im background
- object_class, object_class_str: Id der objektklasse mit dem namen (siehe 'object_classes')
- xmin, xmax, ymin, ymax: sind die koordinaten der bounding box
"""

def firststage(isTrainingData, saveImages=False, out_width=1600, CROP_FOLDER="crops_8-24/"):
    TRAIN_SIZE = 3
    TEST_SIZE = 10
    FOLDER = "validation"
    SIZE = TEST_SIZE
    if isTrainingData:
        FOLDER = "training"
        SIZE = TRAIN_SIZE
    OUT_PATH = "output/"
    CROP_PATH = CROP_FOLDER + FOLDER + "/"
    timestamp = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    timestamp = "X"
    IMG_OUT_PATH = OUT_PATH+"FirstStage_"+timestamp+"/"  + FOLDER + "/"
    TFREC_OUT_PATH = OUT_PATH+"FirstStage_"+timestamp+"/" + FOLDER
    makedirs(IMG_OUT_PATH, exist_ok=True)

    CROPS = [f for f in listdir(CROP_PATH) if isfile(join(CROP_PATH, f))]
    tf_examples = []
    tfrec_writer = tf.io.TFRecordWriter(TFREC_OUT_PATH+"_width"+str(out_width)+".record")
    for i in trange(SIZE):
        bg = generate_gaussiannoiseimg(out_width=out_width)
        xmins, xmaxs, ymins, ymaxs, obj_class_str, obj_class, img_class_str, img_class = [], [], [], [], [], [], [], []
        for k in range(random.randint(1,4)):
            crop = np.random.choice(CROPS)
            img = Image.open(CROP_PATH + crop, 'r')

            # repeat as long as we have a sphero that overlaps on another sphero
            repeat = True
            while repeat:
                addedsize = int(math.ceil(5*(out_width/1600)))
                new_bg, new_img, pos, _, rot,_ = generate_random_image(img, bg, edge_distance=addedsize, out_width=out_width)
                xmin = pos[0] - addedsize
                xmax = pos[0] + addedsize + new_img.width
                ymin = pos[1] - addedsize
                ymax = pos[1] + addedsize + new_img.height
                repeat = False
                for p in range(len(xmins)):
                    # check if overlapping another sphero, bounding box style
                    if ymin >= ymins[p] and ymin <= ymaxs[p] and xmin >= xmins[p] and xmin <= xmaxs[p]:
                        repeat = True
                    if ymin >= ymins[p] and ymin <= ymaxs[p] and xmax >= xmins[p] and xmax <= xmaxs[p]:
                        repeat = True
                    if ymax >= ymins[p] and ymax <= ymaxs[p] and xmin >= xmins[p] and xmin <= xmaxs[p]:
                        repeat = True
                    if ymax >= ymins[p] and ymax <= ymaxs[p] and xmax >= xmins[p] and xmax <= xmaxs[p]:
                        repeat = True


            img = new_img
            bg = new_bg
            xmins.append(xmin)
            xmaxs.append(xmax)
            ymins.append(ymin)
            ymaxs.append(ymax)
            obj_class_str.append("sphero".encode('utf8'))
            obj_class.append(object_classes.get("sphero"))
            img_class_str.append((crop[:-6]).encode('utf8'))
            img_class = sphero_classes.get(img_class_str[-1])
        xmins = [x / bg.width for x in xmins]
        xmaxs = [x / bg.width for x in xmaxs]
        ymins = [x / bg.height for x in ymins]
        ymaxs = [x / bg.height for x in ymaxs]
        image_name = str(i) +  ".png"
        with io.BytesIO() as output:
            bg.save(output, format="PNG")
            io_image = output.getvalue()

        tf_example = tf.train.Example(features=tf.train.Features(feature={      
            'image/encoded': bytes_feature(io_image),
            'image/format': bytes_feature(b'png'),
            'image/filename': bytes_feature(image_name.encode('utf8')),
            'image/source_id': bytes_feature(image_name.encode('utf8')),
            'image/width': int64_feature(bg.width),
            'image/height': int64_feature(bg.height),
            'image/object/class/text': bytes_list_feature(obj_class_str),
            'image/object/class/label': int64_list_feature(obj_class),
            'image/object/bbox/xmin': float_list_feature(xmins),
            'image/object/bbox/xmax': float_list_feature(xmaxs),
            'image/object/bbox/ymin': float_list_feature(ymins),
            'image/object/bbox/ymax': float_list_feature(ymaxs),
            'image/object/subclass/text': bytes_list_feature(img_class_str),
            'image/object/subclass/label': int64_list_feature(img_class),
            'image/object/pose/orientation': int64_feature(rot),
        }))
        tf_examples.append(tf_example)
        if saveImages:
            bg.save(IMG_OUT_PATH+image_name)
        if i % 500:
            # prevents the RAM from getting full
            for example in tf_examples:
                tfrec_writer.write(example.SerializeToString())
            tf_examples = []

    for example in tf_examples:
        tfrec_writer.write(example.SerializeToString())
    tfrec_writer.close()

"""second stage: Identification CNN and Orientation CNN
=> Identification CNN: hohe vielfalt an farben, rotationen und helligkeiten
=> Orientation CNN: hohe vielfalt an rotationen mit unterschiedlichen farben und helligkeiten

- random rgb noise backgrounds mit random brightness in Größe 35x35
- die crops haben die größe 25x25
- jeder crop wird X mal mit jedem winkel erstellt
- zusätzlich erhalten die crops eine random scale, random helligkeit und random position im background
- die CSV datei enthält folgende spalten:
- img_class, img_class_str: ID der klasse und der name der klasse (siehe 'sphero_classes' dict)
"""

def secondstage(isTrainingData, saveImages=False, RotRepetitions=1, out_width=1600, withRandomMirroring=True, CROP_FOLDER="crops_8-24/"):
    FOLDER = "validation"
    if isTrainingData:
        FOLDER = "training"
    OUT_PATH = "output/"
    CROP_PATH = CROP_FOLDER + FOLDER + "/"
    timestamp = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    timestamp = "X"
    IMG_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/"  + FOLDER + "/"
    TFREC_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/" + FOLDER
    makedirs(IMG_OUT_PATH, exist_ok=True)
    
    CROPS = [f for f in listdir(CROP_PATH) if (isfile(join(CROP_PATH, f)) and f.endswith('.png'))]
    csv_rows = []
    tf_examples = []

    for crop in tqdm(CROPS):
        for rot in range(360):
            for k in range(RotRepetitions):
                img = Image.open(CROP_PATH + crop)
                bg = generate_gaussiannoiseimg(SHAPE = (35, 35, 3), brtn=0.2, out_width=out_width)
                
                bg, img, pos, brtn, rot, scl = generate_random_image(img, bg, edge_distance=0, rot=rot, out_width=out_width, withRandomMirroring=withRandomMirroring)

                #bg = bg.resize((128,128), resample=Image.LANCZOS)

                # save image and csv
                with io.BytesIO() as output:
                    bg.save(output, format="PNG")
                    io_image = output.getvalue()
                obj_class_str = "sphero"
                obj_class = object_classes.get(obj_class_str)
                img_class_str = crop[:-6]
                img_class = sphero_classes.get(img_class_str)
                image_name = img_class_str + crop[-6:-4] + "_r" + str(rot) + "_" + str(k) + ".png"
                if saveImages:
                    bg.save(IMG_OUT_PATH+image_name)
                tf_example = tf.train.Example(features=tf.train.Features(feature={      
                    'image/encoded': bytes_feature(io_image),
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
                    'image/object/pose/scale': float_feature(scl),
                    'image/object/pose/brightness': float_feature(brtn),
                    'image/object/pose/x': float_feature(pos[0]),
                    'image/object/pose/y': float_feature(pos[1]),

                }))
                tf_examples.append(tf_example)

    writetfrecord(TFREC_OUT_PATH+"_rot"+str(RotRepetitions)+"_13colors.record", tf_examples)

#from google.colab import drive
#drive.mount('/content/drive')

#import os
#os.chdir('/content/drive/My Drive/Colab Notebooks')
# Upload the crops folder to the folder above
# Should look like this:
#   crops_11_colors/
#   |------\training
#   |------\validation
#
# the *.record files from the compositor are created in '/content/drive/My Drive/Colab Notebooks/output'
# make sure you have enough space available in your google drive (at least ~2GB)
# I would not recommend to save the individual image files created by the compositor to google drive
# google drive might also take a while to display the 'output' folder in the google drive view

#firststage(isTrainingData=True, saveImages=True, out_width=1600) #1600,1200,800,600,400
#firststage(isTrainingData=False, saveImages=False, out_width=400)

secondstage(isTrainingData=True, saveImages=False, RotRepetitions=12, withRandomMirroring=False, CROP_FOLDER="crops_8-24/")
secondstage(isTrainingData=False, saveImages=False, RotRepetitions=12, withRandomMirroring=False, CROP_FOLDER="crops_8-24/")

#!ls
#delete the complete output folder
#!rm output -rf

# check to see what the difference in brightness, scaling looks like...
def test_brightness(isTrainingData, color=None):
    FOLDER = "validation"
    if isTrainingData:
        FOLDER = "training"
    OUT_PATH = "output/"
    CROP_PATH = "crops_8-24_10colors/" + FOLDER + "/"
    timestamp = "X"
    IMG_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/"  + FOLDER + "/"
    TFREC_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/" + FOLDER
    makedirs(IMG_OUT_PATH, exist_ok=True)
    
    CROPS = [f for f in listdir(CROP_PATH) if (isfile(join(CROP_PATH, f)) and f.startswith(color))] #f.endswith('.png'))]
    csv_rows = []
    tf_examples = []
    brtn_range = np.round(np.arange(0.9, 1.2, 0.1), 2)
    print(brtn_range)
    dst1 = Image.new('RGB', (25*len(brtn_range), 25*len(CROPS)))
    for k, crop in enumerate(CROPS):
      for i, brtn in enumerate(brtn_range):
        img = Image.open(CROP_PATH + crop)
        img = ImageEnhance.Brightness(img).enhance(brtn)
        #img = img.resize((int(brtn*img.width),int(brtn*img.height)), resample=Image.LANCZOS)
        dst1.paste(img, (25*i, 25*k))
      image_name = crop
    dst1.save(IMG_OUT_PATH+color+"_brtn.png")


# check to see what the difference in brightness, scaling looks like...
def test_scaling(isTrainingData, color=None):
    FOLDER = "validation"
    if isTrainingData:
        FOLDER = "training"
    OUT_PATH = "output/"
    CROP_PATH = "crops_8-24_10colors/" + FOLDER + "/"
    timestamp = "X"
    IMG_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/"  + FOLDER + "/"
    TFREC_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/" + FOLDER
    makedirs(IMG_OUT_PATH, exist_ok=True)
    
    CROPS = [f for f in listdir(CROP_PATH) if (isfile(join(CROP_PATH, f)) and f.startswith(color))] #f.endswith('.png'))]
    csv_rows = []
    tf_examples = []
    scale = np.round(np.arange(0.9, 1.2, 0.1), 2)
    print(scale)
    dst1 = Image.new('RGB', (int(25*len(scale)*max(scale)), int(25*len(CROPS)*max(scale))))
    for k, crop in enumerate(CROPS):
      for i, brtn in enumerate(scale):
        img = Image.open(CROP_PATH + crop)
        img = img.resize((int(brtn*img.width),int(brtn*img.height)), resample=Image.LANCZOS)
        dst1.paste(img, (int(25*max(scale)*i), int(25*max(scale)*k)))
      image_name = crop
    dst1.save(IMG_OUT_PATH+color+"_scale.png")

# check to see what the difference in brightness, scaling looks like...
def test_rotation(isTrainingData, color=None):
    FOLDER = "validation"
    if isTrainingData:
        FOLDER = "training"
    OUT_PATH = "output/"
    CROP_PATH = "crops_8-24_10colors/" + FOLDER + "/"
    timestamp = "X"
    IMG_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/"  + FOLDER + "/"
    TFREC_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/" + FOLDER
    makedirs(IMG_OUT_PATH, exist_ok=True)
    
    CROPS = [f for f in listdir(CROP_PATH) if (isfile(join(CROP_PATH, f)) and f.startswith(color))] #f.endswith('.png'))]
    csv_rows = []
    tf_examples = []
    rot_range = np.round(np.arange(0, 361, 45), 2)
    print(rot_range)
    dst1 = Image.new('RGB', (int(25*len(rot_range)), int(25*len(CROPS))))
    for k, crop in enumerate(CROPS):
      for i, brtn in enumerate(rot_range):
        img = Image.open(CROP_PATH + crop)
        img = img.rotate(brtn, resample=Image.BICUBIC, expand=False)
        dst1.paste(img, (int(25*i), int(25*k)))
      image_name = crop
    dst1.save(IMG_OUT_PATH+color+"_rot.png")

"""run_test = test_rotation

run_test(isTrainingData=True, color='yellow')
run_test(isTrainingData=True, color='blue')
run_test(isTrainingData=True, color='green')
run_test(isTrainingData=True, color='red')"""