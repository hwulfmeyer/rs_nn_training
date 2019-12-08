import os
import stat
from CNNRobotLocalisation.DataPreprocessing.experiment_definitions import *

################################################################################

REPITIONS = 5
MODEL = "ssd_mobilenet2"
WORK_DIR = os.path.dirname(os.path.realpath(__file__))+'/'
CONFIG_PATH = WORK_DIR+MODEL+"/"
DATA_PATH = "/home/lhoyer/cnn_robot_localization/data/"
TF_RECORD_PATH = "/home/lhoyer/cnn_robot_localization/data/dataset2/"
TRAINING_PATH = "/home/lhoyer/cnn_robot_localization/training/first_stage_2/"+MODEL+'/'
BASE_PIPELINE_FILE = MODEL+'_base_pipeline.config'
LABEL_MAP_FILE = '/home/lhoyer/cnn_robot_localization/CNNRobotLocalisation/LabelMaps/robot_label_map.pbtxt'

os.makedirs(CONFIG_PATH,exist_ok=True)

################################################################################

def compile_eval_records(exp_conf, size_str):
    str = '['
    for rec in exp_conf['eval']:
        str += '"{0}/Validation/1400PerTypeDistr3/eval_{1}{2}.record",'.format(
            DATA_PATH,
            rec,
            size_str)
    str = str[:-1]+']'
    return str

with open(WORK_DIR+BASE_PIPELINE_FILE) as f:
    base_config = ''.join(f.readlines())
with open(WORK_DIR+'pattern_train.txt') as f:
    pattern_train = ''.join(f.readlines())
with open(WORK_DIR+'pattern_eval.txt') as f:
    pattern_eval = ''.join(f.readlines())

exps = create_all_experiments()

for exp_i, exp in enumerate(exps):
    for e in exp:
        width = e['first_stage_resolution'][0]
        height = e['first_stage_resolution'][1]
        #Smaller images seem to make sense only for GPU
        #But smaller images are faster transfered in tensorboard
        #And they reduce the RAM usage drastically
        size_str = '_'+str(width)+'x'+str(height)
        #size_str = ''

        exp_str = 'exp'+str(exp_i)
        format_dict = {
            'WIDTH': width,
            'HEIGHT': height,
            'CHECKPOINT': '"'+TRAINING_PATH + 'base_model/model.ckpt"',
            'TRAIN_RECORD': '"'+TF_RECORD_PATH + exp_str + '/' + e['name'] + size_str + '.record"',
            #'EVAL_RECORD': '"'+TF_RECORD_PATH + exp_str + '/eval/' + e['name'] + size_str + '.record"',
            'EVAL_RECORD': compile_eval_records(e,size_str),
            'CHECKPOINT': '"'+TRAINING_PATH + 'base_model/model.ckpt"',
            'LABEL_MAP': '"'+LABEL_MAP_FILE+'"',
            'PIPELINE': CONFIG_PATH+exp_str+'_'+e['name']+'.config'
        }

        with open(format_dict['PIPELINE'], 'w') as f:
            f.write(base_config.format(**format_dict))
        for i in range(REPITIONS):
            rep_num = '_{:02d}'.format(i)
            format_dict['TRAIN_DIR'] = \
                '"'+TRAINING_PATH+exp_str+'/{}/train{}/"'.format(e['name'],rep_num)
            format_dict['EVAL_DIR'] = \
                '"'+TRAINING_PATH+exp_str+'/{}/eval{}/"'.format(e['name'],rep_num)
            train_file = CONFIG_PATH+'train_'+exp_str+'_'+e['name'] + rep_num
            with open(train_file, 'w') as f:
                f.write(pattern_train.format(**format_dict))
            st = os.stat(train_file)
            os.chmod(train_file, st.st_mode | stat.S_IEXEC)
            eval_file = CONFIG_PATH+'eval_'+exp_str+'_'+e['name'] + rep_num
            with open(eval_file, 'w') as f:
                f.write(pattern_eval.format(**format_dict))
            st = os.stat(eval_file)
            os.chmod(eval_file, st.st_mode | stat.S_IEXEC)
