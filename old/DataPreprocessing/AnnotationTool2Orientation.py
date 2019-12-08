import sys
#Fix problem with ros python path
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
sys.path.remove('/home/lukas/Nextcloud/workspace/robotto_ws/devel/lib/python2.7/dist-packages')

import numpy as np
import os, math
import re
import pascal_voc_writer
from tqdm import tqdm
import cv2
import csv

sys.path.insert(0, '/home/lukas/Nextcloud/Studium/Bachelorarbeit/CNNRobotLocalisation/Utils')
from file_utils import *
# from CNNRobotLocalisation.Utils.file_utils import *

path = "/home/lhoyer/cnn_robot_localization/output/180420_sphero/180417_bg_al_sphero_rolling.avi"
size = 50

files = get_recursive_file_list(path,file_extensions=[".xml"])

def save(f,orientation, defined):
    with open(f) as file:
        content = file.read()
    content = re.sub("<pose>.*</pose>", "<pose>"+str(orientation)+"</pose>",content)
    content = re.sub("<pose_defined>.*</pose_defined>", "<pose_defined>"+str(defined)+"</pose_defined>",content)
    with open(f,'w') as file:
        content = file.write(content)

def load(f):
    data = parseXML(f)['annotation']
    obj = data['object'][0]
    pose = obj['pose']
    if pose == 'Unspecified': pose = 0
    else: pose = int(pose)
    cx = int((int(obj['bndbox']['xmin']) + int(obj['bndbox']['xmax']))/2)
    cy = int((int(obj['bndbox']['ymin']) + int(obj['bndbox']['ymax']))/2)
    return cx,cy,pose,obj

print('YOU HAVE TO RUN THIS CODE WITH PYTHON2')
i = 0
while(1):
    frame = cv2.imread(files[i].replace('.xml','.jpg'))
    cx,cy,theta,obj_dict = load(files[i])
    pose_defined = int(obj_dict['pose_defined'])
    if pose_defined==0:
        cv2.putText(frame,'pose undefined',
            (100,100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0,0,255),
            2)
    else:
        xOff = math.sin(theta*np.pi/180)*size
        yOff = math.cos(theta*np.pi/180)*size
        cv2.line(frame,
                    (int(cx+xOff/5),int(cy+yOff/5)),
                    (int(cx-xOff),int(cy-yOff)),
                    (0,0,255),1)
    cv2.putText(frame,files[i].rsplit('/',1)[1],
            (100,1000),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255,0,0),
            2)

    cv2.imshow('outframe', frame)

    key = cv2.waitKey(-1)
    if key & 0xFF == ord('j'):
        theta = (theta - 90) % 360
    if key & 0xFF == ord('l'):
        theta = (theta + 180) % 360
    if key & 0xFF == ord('k'):
        theta = (theta - 4) % 360
    if key & 0xFF == ord('i'):
        theta = (theta + 4) % 360
    if key & 0xFF == ord('u'):
        pose_defined = (pose_defined+1)%2
    save(files[i],theta,pose_defined)
    if key & 0xFF == ord('a') and i > 0:
        i -= 1
    if key & 0xFF == ord('d') and i < len(files)-1:
        i += 1
    if key & 0xFF == ord('w') and i < len(files)-1:
        i += 1
        # Use pose of last frame
        save(files[i],theta,pose_defined)
    if key & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
