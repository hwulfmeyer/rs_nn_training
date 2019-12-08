import sys
#Fix problem with ros python path
try:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/site-packages')
except:
    print('Couldn\' remove ros cv2')

import os
import csv, pickle, json
from keras import backend as K
import cv2
import pascal_voc_writer
from tqdm import tqdm
import numpy as np
import math
import re
import tensorflow as tf
from object_detection.utils import dataset_util
from CNNRobotLocalisation.Utils.file_utils import *


def save_video_frames(file, out_dir, mod_frames = 1, num_frames = None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(file)
    if num_frames == None:
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in tqdm(range(num_frames)):
        if i%mod_frames!=0:
            cap.grab()
            continue
        ret, frame = cap.read()
        if ret == False:
            break
        cv2.imwrite(out_dir+"/frame{0:06d}.jpg".format(i), frame)
    cap.release()

def saveObjects(frame, contours, path, frameNum, min_area = float('-inf'), max_area = float('inf')):
    print("saveObjects is deprecated")
    os.makedirs(path,exist_ok=True)
    objCnt = 0
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        area = cv2.contourArea(cnt)
        if area > min_area and area < max_area:
            #cv2.rectangle(comb,(x,y),(x+w,y+h),(0,0,255),2)
            roi = frame[y:y+h, x:x+w]
            cv2.imwrite(path+"frame{0:06d}-object{1:02d}.png".format(frameNum,objCnt), roi)
            objCnt = objCnt+1

def extractObjects(frame, contours, rotation = 0,
                   min_len = float('-inf'), max_len = float('inf'),
                   arena_pmin = (128,250), arena_pmax = (1220,970),
                   led_mask=False):
    rois = []
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        #area = cv2.contourArea(cnt)
        if w > min_len and w < max_len and h > min_len and w < max_len and \
           x+w <= arena_pmax[0] and y+h <= arena_pmax[1] and \
           x >= arena_pmin[0] and y >= arena_pmin[1]:
            roi = frame[y:y+h, x:x+w]
            if rotation != 0:
                rows,cols,_ = roi.shape
                M = cv2.getRotationMatrix2D((cols/2,rows/2),rotation,1)
                roi = cv2.warpAffine(roi,M,(cols,rows))
            if led_mask: roi = ledMask(roi)
            rois.append(roi)
    return rois

def extractFrameNumber(string):
    frameNumber = re.search('frame(.*)[-.]', string).group(1)
    if frameNumber == '000000':
        frameNumber = 0
    else:
        frameNumber = int(frameNumber.lstrip('0'))
    return frameNumber

def writeObjects(objects, path, frame_num, seqCnt):
    objCnt = 0
    for roi in objects:
        objPath = path[:-1]+"_obj{0:01d}/".format(objCnt,seqCnt[objCnt])
        os.makedirs(objPath,exist_ok=True)
        cv2.imwrite(objPath+"frame{0:06d}-object{1:02d}.png".format(frame_num, objCnt), roi)
        objCnt = objCnt+1

def writeObject(obj, path, frame_num):
    os.makedirs(path,exist_ok=True)
    cv2.imwrite(path+"frame{0:06d}.png".format(frame_num), obj)

def get_meta_info(meta_file, matcher=None):
    meta_rows = readCSV(meta_file)
    keys = meta_rows[0]
    meta_dicts = []
    for r in meta_rows[1:]:
        if matcher == None or matcher in r[0]:
            meta_dict = {}
            for i in range(len(keys)):
                meta_dict[keys[i]] = r[i]
            meta_dicts.append(meta_dict)
    return meta_dicts

def color_for_frame(first_frame_color, i):
    margin = 2
    last_color = first_frame_color[0][1]
    for pair in first_frame_color:
        if i < pair[0]+margin:
            if i >= pair[0]-margin and i < pair[0]+margin:
                return "undefined"
            return last_color
        last_color = pair[1]
    return last_color

def hasCropChanged(current_crop, last_crop, threshold=10, kernel_size=4):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    if len(last_crop)==0: return True, current_crop
    diff = cv2.erode(subtractBackground(current_crop,
                                        last_crop,
                                        threshold), kernel)
    return cv2.countNonZero(diff)>0, diff

def ledMask(obj):
    obj = cv2.cvtColor(obj, cv2.COLOR_RGB2GRAY)
    ret, obj = cv2.threshold(obj, 200, 255, cv2.THRESH_BINARY)
    return obj

def centroid(contours):
    cx,cy = 0,0
    if len(contours) > 0:
        for cnt in contours:
            M = cv2.moments(cnt)
            cx += int(M['m10']/M['m00'])
            cy += int(M['m01']/M['m00'])
        return round(cx/len(contours)),round(cy/len(contours))
    return 0,0

def subtractBackground(frame, background, threshold):
    fgmask = cv2.absdiff(background,frame)
    fgmask_gray = cv2.cvtColor(fgmask, cv2.COLOR_RGB2GRAY)
    ret, mask = cv2.threshold(fgmask_gray, threshold, 255, cv2.THRESH_BINARY)
    return mask


def applyAlphaMask(frame, mask):
    _,alpha = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    b,g,r = cv2.split(frame)
    comb = cv2.merge([b,g,r, alpha],4)
    return comb

def custom_mae(y_true, y_pred):
    diff = K.abs(y_pred - y_true)
    return K.mean(K.minimum(diff, 360 - diff), axis=-1)
def custom_mse(y_true, y_pred):
    diff = K.abs(y_pred - y_true)
    return K.mean(K.square(K.minimum(diff, 360 - diff)), axis=-1)

def generateMaskMOG(fgbg, frame, threshold=50, ksize1 = 40, ksize2 = 20):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize1,ksize1))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(ksize2,ksize2))
    mask = fgbg.apply(frame)
    mask = cv2.dilate(mask,kernel1,iterations = 1)
    mask = cv2.erode(mask,kernel2,iterations = 1)
    ret, mask = cv2.threshold(mask, threshold, 255, cv2.THRESH_BINARY)
    return mask

def saveVOCBoundingBoxes(out_path, frame, contours, frameNum, robot_type,
                         identificiation, size_min, size_max, length_tol,
                         arena_pmin, arena_pmax,
                         padding=0, preparation_mode=False):
    os.makedirs(out_path+"/Automatic", exist_ok=True)
    #os.makedirs(out_path+"/Automatic_Debug", exist_ok=True)
    os.makedirs(out_path+"/Manual", exist_ok=True)

    objCnt = 0
    frameStr = "frame"+str(frameNum).zfill(6)
    writer = pascal_voc_writer.Writer(out_path+'/'+frameStr+".jpg", 1600, 1200)
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        x = x+padding; y = y+padding; w = w-2*padding; h = h-2*padding
        if w > size_min and w < size_max and h > size_min and h < size_max and \
           abs(h-w)/(w+h) < length_tol and \
           x+w <= arena_pmax[0] and y+h <= arena_pmax[1] and \
           x >= arena_pmin[0] and y >= arena_pmin[1]:
            if preparation_mode:
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                print(str(w) + "," + str(h))
            writer.addObject(robot_type+'/'+identificiation, x, y, x+w, y+h)
            roi = frame[y:y+h, x:x+w]
            objCnt = objCnt+1
    if objCnt == 1:
        cv2.imwrite(out_path+"/Automatic/"+frameStr+".jpg", frame)
        #cv2.imwrite(out_path+"/Automatic_Debug/"+frameStr+".jpg", roi)
        if not preparation_mode:
            writer.save(out_path+"/Automatic/"+frameStr+".xml")
    else:
        cv2.imwrite(out_path+"/Manual/"+frameStr+".jpg", frame)
        if not preparation_mode:
            writer.save(out_path+"/Manual/"+frameStr+".xml")


############### Experiments #########################

# Orientation experiments
# %autoreload
# j = 0
# for f in video_files:
#     cap = cv2.VideoCapture(f)
#     mask = cv2.imread(f.replace('.avi','_mask.png'),cv2.IMREAD_GRAYSCALE)
#     ret, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
#     im2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#     for i in range(num_frames):
#         ret, frame = cap.read()
#         comb = applyAlphaMask(frame,mask)
#         path = MAN_MASK_OUT+'/'
#         saveObjects(comb, contours, path, j, 0, led_mask=True)
#         j += 1
#         break
#     cap.release()

# def alignObjectCropMinRect(frame, contour, debug=True):
#     thetas = []
#     #im2, contours, hierarchy = cv2.findContours(b,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#     rect = cv2.minAreaRect(contour)
#     box = cv2.boxPoints(rect)
#     box = np.int0(box)
#     for j in range(4):
#         thetas.append(math.atan2(box[(j+1)%4][0]-box[j][0],box[(j+1)%4][1]-box[j][1]) * 180 / np.pi)
#     if debug:
#         cv2.drawContours(frame,[box],0,(0,255,255,255),2)
#         #cv2.circle(frame,(500,500),380,(0,255,255,255))
#         print(thetas)
#     return frame

# def alignObjectCropFittingLine(frame, contour, debug=True):
#     rows,cols = frame.shape[:2]
#     [vx,vy,x,y] = cv2.fitLine(contour, cv2.DIST_L2,0,0.01,0.01)
#     lefty = int((-x*vy/vx) + y)
#     righty = int(((cols-x)*vy/vx)+y)
#     cv2.line(frame,(cols-1,righty),(0,lefty),(0,255,0,255),2)
#     return frame
