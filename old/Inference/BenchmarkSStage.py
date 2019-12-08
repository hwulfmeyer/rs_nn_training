import numpy as np
import time
import socket
from PIL import Image
from datetime import datetime
from tqdm import tqdm
from threading import Thread
from multiprocessing import Process
from matplotlib import pyplot as plt
from keras.applications.mobilenet import relu6
from keras.models import load_model
from object_detection.utils import label_map_util

from CNNRobotLocalisation.SecondStage.second_stage_utils import *
from CNNRobotLocalisation.Utils.file_utils import *
#from inference_utils import *


CPU = True
DIR = '/home/lhoyer/cnn_robot_localization/benchmark'

if CPU:
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

cols, stdevs = [], []
for alpha in [25,50,75,100]:
    cols.append(['Alpha {}'.format(alpha/100)])
    netname = '{}/alpha_{}_copter_{}.h5'.format(DIR,alpha,'cat')
    second_stage_model = load_model(netname,
                       custom_objects={
                       'relu6': relu6,
                       'angle_mse': angle_mse,
                       'angle_mae': angle_mae,
                       'angle_bin_error': angle_bin_error})

    times = []
    for num_crops in range(1,21):
        for i in range(100+10):
            image_np = np.random.rand(num_crops,128, 128, 3) * 255
            start = time.time()
            second_stage_model.predict(image_np)
            end = time.time()
            # Skip first two inferences which are significantly slower
            if (i >= 10): times.append(1000*(end - start))
        print('{}: Average exec time for {} crops in ms: {} +- {}%'
            .format(netname.rsplit('/',1)[1],
                num_crops,
                np.mean(times),
                np.std(times)/np.mean(times)*100)
        )
        cols[-1].append(np.mean(times))
        stdevs.append(np.std(times)/np.mean(times)*100)
#     plt.plot(range(1,10),cols[-1], label=str(alpha))
# plt.ylabel('Validation accuracy')
# plt.xlabel('Image size')
# plt.legend(loc='lower right')
# plt.title('TITLE')
# plt.show()

print('Average relative standard deviation: {}%'.format(np.mean(stdevs)))
print('Maximum relative standard deviation: {}%'.format(np.max(stdevs)))

rows = zip(*cols)
writeCSV('{}/benchmark-result-{}-{}.csv'.format(DIR,socket.gethostname(),'cpu' if CPU else 'gpu'), rows)
