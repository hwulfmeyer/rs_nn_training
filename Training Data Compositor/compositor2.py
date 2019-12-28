import csv
import datetime
from enum import Enum
from os import listdir, makedirs
from os.path import isfile, join

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from object_detection.utils import dataset_util, label_map_util
from PIL import Image, ImageEnhance
from tqdm import tqdm

np.random.seed(146324)

# SHAPE = (height, width, 3)
def creategaussiannoiseimg(SHAPE = (1200, 1600, 3)):
    noise = np.random.randint(0, 255, SHAPE)
    noise = noise.astype(dtype=np.uint8)
    img = Image.fromarray(noise, mode='RGB')
    img = ImageEnhance.Color(img).enhance(1.2)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = ImageEnhance.Brightness(img).enhance(0.5)
    return img

"""
random rgb noise backgrounds
"""
def firststage():
    img = creategaussiannoiseimg()

    img.save("gaussian_noise.png")

"""
second stage: Identification CNN and Orientation CNN
- uses all images in CROP_PATH to create the training data
- image name must be like so YX.png
    - where X=the number of the crop(one digit) and Y = the class in the 'classes' dictionary
- creates the crops with all 360 different degrees and 3 different brightnesses => 1080 images per crop
- output is cropped to FIXHEIGHT and FIXWIDTH

TODO:
    - create 5 different crops (e.g. each corner and the middle) per color for the training data set
    - create 2 different crops per color for the test data set
"""
def secondstage(isTrainingData):
    classes = {
        "bright_blue": 1,
        "bright_red": 2,
        "bright_green": 3,
        "bright_white": 4,
        "dark_blue": 5,
        "dark_green": 6,
        "dark_red": 7
    }
    folder = "test/"
    if isTrainingData:
        folder = "training/"


    OUT_PATH = "output/" + folder
    CROP_PATH = "crops/" + folder
    FIXHEIGHT = 30
    FIXWIDTH = 30



    ### Second stage compositor
    timestamp = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    timestamp = "test"
    csv_rows = []
    IMG_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/"
    makedirs(IMG_OUT_PATH, exist_ok=True)

    CROPS = [f for f in listdir(CROP_PATH) if isfile(join(CROP_PATH, f))]
    for crop in tqdm(CROPS):
        for rot in (0, 90, 180): # range(360)
            for k in (0.5,1.0,1.5):
                img = Image.open(CROP_PATH + crop)

                #rotation
                img = img.rotate(rot, resample=Image.BICUBIC, expand=False)

                """
                #cropping
                w, h = img.size
                left = (w - FIXWIDTH)/2
                top = (h - FIXHEIGHT)/2
                right = (w + FIXWIDTH)/2
                bottom = (h + FIXHEIGHT)/2
                img = img.crop((left, top, right, bottom))
                """
                # brightness transformation
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(k)

                bg = creategaussiannoiseimg(SHAPE = (30,30,3))
                bg.paste(img, (0,0), img)
                img = bg
                # save image and csv
                img_class_str = crop[:-5]
                img_class = classes.get(img_class_str)
                image_name = img_class_str + "_r" + str(rot) + "_" + str(k) + "_" + crop[-5] + ".png"
                img.save(IMG_OUT_PATH+image_name)
                csv_rows.append([image_name, img_class_str, img_class, rot])


    with open(IMG_OUT_PATH+"#groundtruth.csv", 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            spamwriter.writerow(row)


if __name__ == "__main__":
    #firststage()
    secondstage(True)
    #secondstage(False)
