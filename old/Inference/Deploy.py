import os

DIR='/home/lhoyer/cnn_robot_localization/training/first_stage_2/ssd_mobilenet2/exp0'
SUFFIX='_00'
CKPT=27818

os.system("python3 /home/lhoyer/tensorflow/models/research/object_detection/export_inference_graph.py \
    --input_type image_tensor \
    --pipeline_config_path {DIR}/train_00/pipeline.config \
    --trained_checkpoint_prefix {DIR}/train{SUF}/model.ckpt-{CKPT} \
    --output_directory {DIR}/deploy{SUF}".format(DIR=DIR,CKPT=CKPT,SUF=SUFFIX))
