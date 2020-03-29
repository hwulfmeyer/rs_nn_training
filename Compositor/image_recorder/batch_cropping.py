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
from PIL import Image, ImageEnhance
from tqdm import tqdm, trange
import math

"""
This script is useful if you take several pictures without moving the spheros in the pictures.
Simply add the coordinates of the spheros and it will crop them out.
 - variable CROP_LEFTTOP contains the coordinates with the top left point of a cut out
 - CROP_SIZE is the pixel size of the crop out which is added to the coordinates in CROP_LEFTTOP
 - the compositor only works with double-digit numbers at the end, which is why the BEGIN_INDEX is set to 10
"""
def crop_out():
    IMAGES_PATH = "images_13_links/"
    #IMAGES_PATH = "images_13_rechts/"
    OUT_PATH = IMAGES_PATH + "batch_crop_out/"
    makedirs(OUT_PATH, exist_ok=True)
    CROP_SIZE = 25

    IMAGES = [f for f in listdir(IMAGES_PATH) if (isfile(join(IMAGES_PATH, f)) and f.endswith('.png'))]
    CROP_LEFTTOP = [(320,254),(433,392),(523, 526),(664,702), (570,826)] #(790, 822) #images left
    #CROP_LEFTTOP = [(1113,194),(908,362),(993,621),(778,669), (1113,838)] #images

    BEGIN_INDEX = 10

    for img_name in IMAGES:
        for i, (LEFT, TOP) in enumerate(CROP_LEFTTOP):
            img = Image.open(IMAGES_PATH + img_name)
            crop = img.crop((LEFT, TOP, LEFT+CROP_SIZE, TOP+CROP_SIZE)) #((left, top, right, bottom))
            crop_name = img_name[11:-4]+str(i+BEGIN_INDEX)
            crop.save(OUT_PATH+crop_name+".png")

crop_out()
