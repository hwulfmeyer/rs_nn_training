from CNNRobotLocalisation.Utils.file_utils import *

OBJECT_PATH = '/home/lhoyer/cnn_robot_localization/data/tmp/RobotCrops'

metaFileList = get_recursive_file_list(OBJECT_PATH, file_matchers=['meta.json'])
for metaFile in metaFileList:
    meta = load_json(metaFile)
    if 'youbot' in metaFile and 'mp2' not in metaFile:
        meta['marker_position'] = 'bottom'
        save_json(metaFile, meta)
    else:
        meta['marker_position'] = 'center'
        save_json(metaFile, meta)
    print('{} add marker position: {}'.format(metaFile, meta['marker_position']))
