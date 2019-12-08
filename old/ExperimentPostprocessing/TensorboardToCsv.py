import os, csv
from tqdm import tqdm
from tensorboard_utils import *

LOG_DIR = '/home/lhoyer/cnn_robot_localization/training'
OUT_DIR = '/home/lhoyer/cnn_robot_localization/training_processed'

recfiles = []
for root, subdirs, files in os.walk(LOG_DIR):
    for file in files:
        recfiles.append(root+'/'+file)
recfiles.sort()

for file in recfiles:
    if 'tfevents' not in file:
        continue

    tensorboard_dict = readTfEvents(file)

    out_file = file.replace(LOG_DIR, OUT_DIR) + '.csv'
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    print(out_file)

    with open(out_file, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=';',
                            quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(tensorboard_dict.keys())
        writer.writerows(zip_longest(*tensorboard_dict.values(), fillvalue=''))
