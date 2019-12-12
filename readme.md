##Second Stage

python3 verwenden

####benötige Tools:
- tqdm
- keras
- lxml
- sklearn (pip3 install scipy, pip3 install sklearn)

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#manual-protobuf-compiler-installation-and-usage
(sonst fehlt string_int_label_map_pb2 in object_detection)


###fehlende Ordner/ Dateien:

TRAIN_DIR = "/home/lhoyer/cnn_robot_localization/data/**sstage_dataset2**/"
EVAL_DIR = "/home/lhoyer/cnn_robot_localization/data/Validation/**secondstage110balance4**/"

TRAIN_RECORD = '/**media/data**/LocalizationDataNew/Remote/data/sstage_dataset2/**no_crop_var_crop_youbot.record**'
OUT_PATH = '**/media/data**/LocalizationDataNew/Output/Inference/'
MODEL = '**/media/data**/LocalizationDataNew/Remote/training/second_stage/**sstage_sphero_test_sphero_ori/2018-06-07-20-03/model-final.h5**'

FILE = '/media/data/LocalizationDataNew/Remote/data/Validation/**secondstage110balance3/eval_bg_al_sphero.record**'
OUT_PATH = '/media/data/LocalizationDataNew/Output/**TFRecordInspectorSStage/**'

from object_detection.protos import **string_int_label_map_pb2**

