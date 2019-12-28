import csv
import datetime
from os import listdir, makedirs
from os.path import isfile, join

import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm
import decimal

np.random.seed(146324)

sphereo_classes = {
    "bright_blue": 1,
    "bright_red": 2,
    "bright_green": 3,
    "bright_white": 4,
    "dark_blue": 5,
    "dark_green": 6,
    "dark_red": 7
}


"""
- SHAPE = (height, width, 3)
- the images have a random brightness
"""
def creategaussiannoiseimg(SHAPE = (1200, 1600, 3), brtn = None):
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


"""
- random rgb noise backgrounds mit random brightness in Größe 300x300
- auf die backgrounds werden die sphero crops superimposed
- die crops haben eine random scale, rotation, brightness und position im background
- die CSV datei enthält folgende spalten:
	[image_name, xmin, xmax, ymin, ymax, img_class_str, img_class, rot, scl, brtn]
	image_name: vollständige Name der Bilddatei
	xmin, xmax, ymin, ymax: sind die koordinaten der bounding box
	img_class_str: ist der name der klasse (siehe 'sphereo_classes' in compositor.py)
	img_class: ist die ID der klasse (siehe 'sphereo_classes' in compositor.py)
	rot: rotationswinkel
	scl: scalingfaktor
	brtn: brightnessfaktor
"""
def firststage(isTrainingData):
    TRAIN_SIZE = 1000
    TEST_SIZE = 300
    BGHEIGHT = 300
    BGWIDTH = 300
    FOLDER = "test/"
    SIZE = TEST_SIZE
    if isTrainingData:
        FOLDER = "training/"
        SIZE = TRAIN_SIZE
    OUT_PATH = "output/"
    CROP_PATH = "crops/" + FOLDER
    CROPS = [f for f in listdir(CROP_PATH) if isfile(join(CROP_PATH, f))]

    timestamp = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    timestamp = "X"
    IMG_OUT_PATH = OUT_PATH+"FirstStage_"+timestamp+"/"  + FOLDER
    makedirs(IMG_OUT_PATH, exist_ok=True)
    csv_rows = []

    for i in tqdm(range(SIZE)):
        bg = creategaussiannoiseimg(SHAPE = (BGHEIGHT, BGWIDTH, 3))
        crop = np.random.choice(CROPS)
        img = Image.open(CROP_PATH + crop)

        # random scale
        scl = round(np.random.uniform(0.9, 1.1), 2)
        img = img.resize((int(scl*img.width),int(scl*img.height)), resample=Image.LANCZOS)

        # random rotation
        rot = np.random.randint(360)
        img = img.rotate(rot, resample=Image.BICUBIC, expand=False)

        # random brightness
        brtn = round(np.random.uniform(0.5, 1.5), 2)
        img = ImageEnhance.Brightness(img).enhance(brtn)

        # random position
        # !position is the upper left corner of the crop in the picture!
        pos = (np.random.randint(0, bg.width-img.width), np.random.randint(0, bg.height-img.height))
        
        bg.paste(img, pos, img)

        # save image and csv
        img_class_str = crop[:-5]
        img_class = sphereo_classes.get(img_class_str)

        image_name = str(i) +  ".png"
        xmin = pos[0]
        xmax = xmin+img.width
        ymin = pos[1]
        ymax = ymin+img.height
        csv_rows.append([image_name, xmin, xmax, ymin, ymax, img_class_str, img_class, rot, scl, brtn])
        bg.save(IMG_OUT_PATH+image_name)

    with open(IMG_OUT_PATH+"groundtruth.csv", 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            spamwriter.writerow(row)



"""
second stage: Identification CNN and Orientation CNN
- uses all crops in CROP_PATH to create the training data
- crops need to have an alpha channel and a round cut out for the spheros
- image name must be like so YX.png
    - where X=the number of the crop(one digit) and Y = the class in the 'classes' dictionary

- random rgb noise backgrounds mit random brightness in Größe 35x35
- die crops haben die größe 30x30
- für jeden vorhandenen crop wird für alle 360 Grad ein bild erstellt
- zusätzlich für jeden der 360 winkel 3 verschiedene helligkeiten, random scale und random position im background
- die CSV datei enthält folgende spalten:
	[image_name, img_class_str, img_class, rot, scl, brtn, pos[0], pos[1]])
	image_name: vollständige Name der Bilddatei
	img_class_str: ist der name der klasse (siehe 'sphereo_classes' in compositor.py)
	img_class: ist die ID der klasse (siehe 'sphereo_classes' in compositor.py)
	rot: rotationswinkel
	scl: scalingfaktor
	brtn: brightnessfaktor
	pos[0] und pos[1]: position der crops im bg (ist der linke obere pixel vom crop im bg)

TODO:
    - create 5 different crops (e.g. each corner and the middle) per color for the training data set
    - create 2 different crops (different from the training) per color for the test data set
"""
def secondstage(isTrainingData):
    folder = "test/"
    if isTrainingData:
        folder = "training/"


    OUT_PATH = "output/"
    CROP_PATH = "crops/" + folder
    #FIXHEIGHT = 30
    #FIXWIDTH = 30

    timestamp = "{:%Y%m%d_%H%M%S}".format(datetime.datetime.now())
    timestamp = "X"
    csv_rows = []
    IMG_OUT_PATH = OUT_PATH+"SecondStage_"+timestamp+"/" + folder
    makedirs(IMG_OUT_PATH, exist_ok=True)

    CROPS = [f for f in listdir(CROP_PATH) if isfile(join(CROP_PATH, f))]
    for crop in tqdm(CROPS):
        for k in range(3):
            for rot in range(360):
                for brtn in (0.5,1.0,1.5):
                    img = Image.open(CROP_PATH + crop)
                    bg = creategaussiannoiseimg(SHAPE = (35,35,3))

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

                    # random scale
                    scl = round(np.random.uniform(0.9, 1.1), 2)
                    img = img.resize((int(scl*img.width),int(scl*img.height)), resample=Image.LANCZOS)

                    # brightness transformation
                    img = ImageEnhance.Brightness(img).enhance(brtn)

                    # random position
                    # !position is the upper left corner of the crop in the picture!
                    pos = (np.random.randint(0, bg.width-img.width), np.random.randint(0, bg.height-img.height))
                    bg.paste(img, pos, img)

                    # save image and csv
                    img_class_str = crop[:-5]
                    img_class = sphereo_classes.get(img_class_str)
                    image_name = img_class_str + crop[-5] + "_r" + str(rot) + "_s" + str(scl) + "_b" + str(brtn) + "_p" + str(pos[0]) + str(pos[1]) + str(k) + ".png"
                    bg.save(IMG_OUT_PATH+image_name)
                    csv_rows.append([image_name, img_class_str, img_class, rot, scl, brtn, pos[0], pos[1]])


    with open(IMG_OUT_PATH+"groundtruth.csv", 'w') as csvfile:
        spamwriter = csv.writer(csvfile, delimiter=',',
                                quotechar='"', quoting=csv.QUOTE_MINIMAL)
        for row in csv_rows:
            spamwriter.writerow(row)


if __name__ == "__main__":
    img = creategaussiannoiseimg(SHAPE = (500, 500, 3), brtn=1.0)
    img.save("gaussian_noise.png")

    #firststage(True)
    firststage(False)
    #secondstage(True)
    #secondstage(False)
