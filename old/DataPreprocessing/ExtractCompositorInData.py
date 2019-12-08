import shutil
import tarfile
from tqdm import tqdm
from CNNRobotLocalisation.Utils.file_utils import *

OBJECT_TAR_PATH = "/home/lhoyer/cnn_robot_localization/data/CompositorInputData/RobotCrops/TrainData1"
BACKGROUND_TAR_PATH = "/home/lhoyer/cnn_robot_localization/data/CompositorInputData/Backgrounds"
OBJECT_PATH = '/home/lhoyer/cnn_robot_localization/data/tmp/RobotCrops'
BACKGROUND_PATH = '/home/lhoyer/cnn_robot_localization/data/tmp/Backgrounds'

# Untar to tmp dir
assert 'tmp' in BACKGROUND_PATH
assert 'tmp' in OBJECT_PATH
shutil.rmtree(BACKGROUND_PATH,ignore_errors=True)
shutil.rmtree(OBJECT_PATH,ignore_errors=True)
for f in tqdm(get_recursive_file_list(OBJECT_TAR_PATH)):
    out_f = f.replace(OBJECT_TAR_PATH,OBJECT_PATH).rsplit('/',1)[0]
    #print(out_f)
    with tarfile.open(f) as tar:
        tar.extractall(out_f)
for f in tqdm(get_recursive_file_list(BACKGROUND_TAR_PATH,file_excludes=["COCO.tar"])):
    out_f = f.replace(BACKGROUND_TAR_PATH,BACKGROUND_PATH).rsplit('/',1)[0]
    #print(out_f)
    with tarfile.open(f) as tar:
        tar.extractall(out_f)
