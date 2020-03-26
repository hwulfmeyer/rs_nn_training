# FirstStage
This folder contains the first stage of the detection framework, which detects any number of desired objects in an input image and outputs a list of bounding boxes around them. The following explanation is a walk-through of how to build a custom object detection framework either on your local machine, or on Google Colaboratory.

## Training a model on your local machine
The following steps are a short Guideline on how to install the Tensorflow Object Detection API on your local machine. Since the underlying code is prone to changes, refer to the [https://github.com/tensorflow/models] Repository.

### Download and install the Tensorflow Object Detection API
1. Install tensorflow
2. Clone [https://github.com/tensorflow/models] into your tensorflow installation (e.g. C:\Python37\Lib\site-packages\tensorflow\<here>)
3. Fully follow these [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
4. cd into the models/research/ folder
5. `python setup.py build`
6. `python setup.py build_ext`
7. `python setup.py install`

### Preparing the model
To train an existing model based on various scientific publication, go to [Tensorflow Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and download any of the provided model .tar.gz files. These contain a `pipeline.config` file, which specifies with which parameters the training framework will train the model. For a mode detailed explanation of what these parameters do, refer to the [models](models/) directory, or read through the protobuf definitions at [https://github.com/floydhub/object-detection-template/tree/master/object_detection/protos].

### Building an image dataset
Generate a TFRecord file for your training and test data. This can be done through the provided [Compositor](../Training Data Compositor). Once generated, amend the paths to your label_map.pbtxt, training.record and validation.record in their respective parameters inside the `pipeline.config`. The label map is simply the file that assigns a label to every output of the model. Lastly the `fine_tune_checkpoint` parameter should point to the checkpoint of the last training process you wish to continue from. When building a clean model, this should point to the `model.ckpt` in the .tar.gz file.

When generating your dataset, keep in mind that the Object Detector generates bounding boxes whose coordinates are relative to the whole image, no matter its' size in pixels (the x, y of the bounding box will be a float value between 0 and 1). Therefore you should train your model with images, in which the proportions of object to the entire image are as realistic as possible.

### Train the model
python /content/models/research/object_detection/model_main.py \
    --pipeline_config_path=PATH_TO_YOUR_PIPELINE_CONFIG \
    --model_dir=PATH_TO_YOUR_MODEL_CHECKPOINT_DIRECTORY \
    --alsologtostderr \
    --num_train_steps=4000 \
    --num_eval_steps=1000

### Export the model as a frozen inference graph
python /content/models/research/object_detection/export_inference_graph.py \
    --input_type=image_tensor \
    --pipeline_config_path=PATH_TO_YOUR_PIPELINE_CONFIG \
    --output_directory=PATH_TO_WHERE_YOUR_GRAPH_WILL_BE_EXPORTED \
    --trained_checkpoint_prefix=PATH_TO_THE_CHECKPOINT

### Popular bugs
[Unable to install cocoapi](https://github.com/cocodataset/cocoapi/issues/295)
[No module named ...](https://github.com/tensorflow/models/issues/1842)
["protoc" not found](https://stackoverflow.com/questions/52929161/cannot-find-protoc-command?rq=1)

## Training your model on Google Colaboratory
To build your object detection model on Google Colaboratory, simply upload the provided `first_stage_colab.ipynb` file and run it.

## Evaluating a trained model
To evaluate your trained models, refer to the [evaluation](evaluation/) folder.
