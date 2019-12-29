# How to retrain the first stage model

## Download and install the Tensorflow Object Detection API
1. Install tensorflow
2. Clone [https://github.com/tensorflow/models] into your tensorflow installation (e.g. C:\Python37\Lib\site-packages\tensorflow\<here>)
3. Fully follow these [instructions](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md)
4. cd into the models/research/ folder
5. `python setup.py build`
6. `python setup.py build_ext`
7. `python setup.py install`

## Add your own images
* Image files can be added into the data/images
* Generate respective labels
* Create a tfrecords dataset from images and labels (for train and test data) using xml_to_csv.py and generate_tfrecord.py

## Download the model
* Run the setup script

## Amend the Object Detection Training Pipeline
* Edit the data/model_default.config files
* Important: Amend all paths to the respective locations in your system!

## Train the model
* Amend the paths in the "train" file
* Execute the commands inside the "train" file through an elevated command prompt inside your tensorflow/models/research/ folder

### Popular bugs
[Unable to install cocoapi](https://github.com/cocodataset/cocoapi/issues/295)
[No module named ...](https://github.com/tensorflow/models/issues/1842)
["protoc" not found](https://stackoverflow.com/questions/52929161/cannot-find-protoc-command?rq=1)