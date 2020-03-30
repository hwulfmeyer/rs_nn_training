# Second Stage



## Train Local

### Installation

The second stage runs in python3 with tensorflow 1.15. You also need the tensorflow object detection API, which installation is described in the documentation of the first stage.

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md#manual-protobuf-compiler-installation-and-usage

Additional packages: keras, tqdm, lxml, sklearn, pandas, numpy, seaborn, matplotlib, pillow

### File structure

| file                 | description                                                  |
| -------------------- | ------------------------------------------------------------ |
| Secondstage          | training pipeline, mobilenet configuration                   |
| second_stage_utils   | contains custom metrics, load crops, define tensorboard callbacks |
| label_map            | assigns a label (colour) to each ID                          |
| PredictionInspection | iterates through test data and saves each image with predicted and groundtrouth label (debugging) |
| exp_model	       | contains an experimental model with a custom network		| 



## Train On Google Colaboratory (recommended)

- upload SecondStage_colab to your drive
- open SecondStage.ipynb in Google Colaboratory
- set paths for the training and validation data set
- choose method 
  - Identification Net: classification
  - Rotation Net: binning or regression (binning worked better for us)
- Runtime -> Run all



**editing files** (second_stage_utils, labelmap,..) **:**

- copy file into new code block

- write `%%writefile <example_file.py>` in first line

  -> overwrites existing file when executing (makes editing easier than reuploading every time)

use: Runtime -> Factory reset runtime to delete old file saved in memory



## Configuration

- alpha

  The alpha value sets the width of the [MobileNet](https://github.com/keras-team/keras-applications/blob/master/keras_applications/mobilenet.py). Increasing alpha from 0.5 to 0.75 significantly   improved the classification on our test data set.

- batch size

  The batch size is mainly chosen based on the available hardware. A higher batch size trains faster, but needs more memory. We achieved better results with a smaller batch size. Probably because small batch sizes can result in a more robust result due to a regularizing effect.

- dropout

  Because our training data were really similar (356400 images generated out of 110 crops), we thought a higher dropout could have a beneficial impact on preventing overfitting. But for our data the effect was rather negative.

- epochs and early stopping

  To always save the weights at the best epoch, we used early stopping. It saves the validation loss value and the weights (or any other value, eg F1 average) and compares it to the values of the next 7 epochs (adjustable). When the loss doesn't further decrease, the training process stops. 

- learning rate

  We sticked to Lukas learning rate of 0.001, because neither a higher nor lower rate improved the predictions.

- noise

  Our main problem when classifying colours was, that our data set was too small to represent all possible pixel combinations. Adding a noise layer after the input and before the activation layer, seemed to help improving the performance on unknown data.
  
  

## Evaluation

### Tensorboard

The easiest way to evaluate the training results is reviewing the created event file with tensorboard.

`tensorboard --logdir=./folder_with_event_file/`

For sharing the board with your team mates, use:

`tensorboard dev upload --logdir=./folder_with_event_file/`



### PreditionInspection

Reviewing the wrong label for each file, can help with debugging. Especially when one original crop is always or never marked with a wrong colour or the rotation of a crop with poorly visible back led varies deviates by +/- 180°.

**usage**:

- set paths

- TRAIN_OR_VAL= <'val' | 'train'>

  MODE = <'classification' | 'regression'>

run `python3 PredictionInspection.py`



### Test in Arena

The output is a keras h5 file, but the nn_tracking_node needs a tensorflow pb graph.

`python3 convert_h5_to_pb.py --model final-model.h5 -n 3 --outdir=.`

Check the input name in the ascii graph and modify the meta file.



