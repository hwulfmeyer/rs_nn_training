#!/bin/bash

rsync --include='*.tfevents*' --include='*/' --exclude='*' --max-size=100m -amvz lukas@ws:/home/lhoyer/cnn_robot_localization/training/second_stage_2 /home/lhoyer/cnn_robot_localization/training_results_tmp
#rsync --include='*.tfevents*' --include='*/' --exclude='*' --max-size=100m -amvz lhoyer@chicken:/home/lhoyer/cnn_robot_localization/training/first_stage_2 /home/lhoyer/cnn_robot_localization/training_results_tmp
#rsync --include='*.tfevents*' --include='*/' --exclude='*' --max-size=100m -amvz lhoyer@ocaml:/home/lhoyer/cnn_robot_localization/training/second_stage_2 /home/lhoyer/cnn_robot_localization/training_results_tmp

#rsync --include='*.tfevents*' --include='*/' --exclude='*' --max-size=100m -amvz /run/media/lukas/MyPassport/ocaml/training/second_stage_2 /home/lhoyer/cnn_robot_localization/training_results_tmp
