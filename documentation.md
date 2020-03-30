# WS 19/20 - Rolling Swarm Camera Tracking
The camera tracking module is responsible for detecting various robots in a camera image and labeling them according to their colour. For this project the focus was lying entirely on the Sphero platform, however some of the results may also be applicable to different robots.

The goals for this project were:
* improving the detection rate at different positions inside the arena
* previously certain positions, especially near edges, led to erroneous detections
* increasing the amount of distinguishable colours
* reducing the rotation deviation
* cleaning up and improving the training pipeline for the Two-Stage Detector architecture
* updating deprecated parts of the code in conjunction with current framework versions

Most code found in this project was optimizied to be run on google colab. The files are jupyter notebooks and it should be able to run them on a pc with minor tweaks. Use the jupyter notebook files (ipynb) before using the .py files.

## Compositor
The compositor creates data sets for the training and validation of the first and second stage of the framework. The data is serialized and stored in a file using the TFRecord data format. [Compositor directory](./Compositor/)

### Data Aquisition
The data aquisition process is a rather mundane and time consuming task, which is unfortunately not really fully automateable.  
Based on the previous work we followed the same principle: 
1. Spheros are placed in different spots in the arena and static pictures are taken with the camera. [See Image Recorder directory](./Compositor/image_recorder)
2. The spheros in these pictures are cropped out using a round cutout of size 25x25 and saved with a transparent background.
3. Crops of the Spheros are all zeroed to a predefined angle. (back LED at the bottom of the picture)

Gimp was used for Step 2 and 3.
Since the second stage has the goal of predicting the angle of the spheros a round crop out (and subsequently a transparent background) is used to prevent the neural network of being able to learn the edges of a hypothetical square crop. Because the compositor rotates the crops to create more data great care has to be taken such that the spheros in the crops are roughly centered. Centered here means that the middle of a crop should be between the two color leds of the sphero.

The above process has to be repeated for every possible color that is going to be used. The following colors were tested:
| Color         | RGB Values    |
|---------------|---------------|
|red            |255,0,0        |
|orange         |255,85,0       |
|yellow         |255,170,0      |
|lime_green     |255,255,0      |
|magenta        |255,0,185      |
|purple         |255,0,255      |
|green          |0,255,0        |
|light_green    |0,255,85       |
|blue_green     |0,255,170      |
|light_blue     |0,255,255      |
|blue           |0,0,255        |
|dark_red       |150,0,0        |
|dark_green     |0,128,0        |
|dark_blue      |0,0,150        |
|white          |185,128,50     |

The rgb values [0,255,255] and [255,255,255] look visually identical.

Initially we used 7 crops per color and later scaled this up to 17 crops per color.

### Data Augmentation
The crops are split up into training and validation and then used to augment more data.
The augmentation utilizes changes in brightness, scaling, rotation, and translation. Additionally, also horizontal mirroring is used. Relatively speaking the process is almost identical to the first and second stage. Nevertheless, there are different use cases for each stage that need to be considered.

The training data for the first stage contains pictures that have the same aspect ratio resolution as the camera pictures (1600x1200). Usually the full size is not taken but instead a scaled down resolution (400x300). A number of crops are then superimposed with a random position, brightness, rotation, and scaling on these pictures and also accurately scaled to represent the resolution, i.e. a background size of 400x300 means the spheros have 1/4 of their original size. For each picture a list of bounding box values are supplied to the TFRecord file.

For the second stage for each crop every 360° rotation is repeated a set amount of times. Concurrently the crops are set to a random scale and brightness for every example. Next, the crops are superimposed on a 35x35 background with a random position.  
A 35x35 sized background is used to make it possible to choose a random position for the crops. This is done because it can not be expected that the first stage supplies the second stage with very accurate bounding boxes of the spheros, which means the training data should include examples where spheros are not exactly in the middle. Theoretically this should not constitute a problem since Convolutional Neural Networks (CNN) are translation invariant. However, additionaly to making sure that every weight of every filter in the CNN is trained equally (we only have 17 crops per color) the inaccuracy of the first stage is problematic for the rotation prediction in the second stage. If we do not include a random position the rotation prediction could simply learn the location of the back LED in the crops to predict the angle. Using a random position forces the network to learn the location of the back LED in relation to the color LEDs.  
Since the spheros were manually zeroed to a set angle and manually cropped and centered it includes some human error. This error can be reduced by including random horizontal mirroring in the compositor, which is applied before every other augmentation method. For each example the ID and Rotation is saved in the TFRecord.

The above mentioned backgrounds contain very minor random noise.
We used +-10% brightness and scaling. The brightness differences could be made larger but then it should be ensured that they do not look like another color in the set, e.g. green and dark green or blue, light blue and dark blue.


### Future Work
A neural network is only as good as the data that you feed it to. This is why the current data aquisition process presents a serious shortcoming which should be adressed in future revisions of this framework. Even 17 crops per color does not seem to accurately approximate the possible dynamic and characteristic of the colored leds of the spheros in real life scenarios. One problem is that these are only 17 positions from infinitely possible positions inside the arena. Another problem is that the chosen positions of the spheros are biased because we placed them. This likely results in the data not having a high representing accuracy of the actual data that has to be predicted later on. At best this will only be an issue for the Second Stage Identification Prediction.

One possible way of preventing this issue is through semi-automatic data aquisition. The crops are still used to train the first stage. The bounding box values created by the first stage on live pictures are then used to create crops from these live pictures, for example by letting spheros of one specific color drive around in the arena. However, this still has to be repeated for every possible color. This only represents a way of automating the process for the training of the second stage identification prediction but not for the rotation prediction.

The rotation identification already worked very well even with a few amount of data. However, when the first stage supplies the second stage with inaccurate bounding boxes such that the spheros are not roughly centered the identification error got worse. The accuracy could be further improved by increasing the random positioning of the crops in the backgrounds by making the crops smaller and also allowing the script to cut a few pixels off from the crops to place them further to the edges of the backgrounds.

## First Stage
This part of the architecture detects each robot contained within a full camera image. The output of the network is a list of bounding boxes with soft-max confidences towards each possible robot type. The first stage does however not difference between the colours of the detected Spheros, this is part of the Second Stage. The whole architecture is based on the [Tensorflow Object Detection API](https://github.com/tensorflow/models/tree/master/research/object_detection). The following sections will mainly focus on theoretical descriptions of our actions and evaluations. The practical How-to's are discussed in the [First Stage directory](./FirstStage)

### The training data
To train our object detector, we used automatically generated images of Sphero crops on a black background. Refer to the Compositor for more information on how to generate image datasets.
Each Sphero crop was scaled to 25x25 Pixels in reference to the full 1600x1200 image. When using a lower resolution for the camera image, the Sphero crops were scaled proportionally. The ground truth bounding box which is also saved within the TFRecord file was centered around the Sphero with a size of 35x35 Pixels so that small deviations in the later detection will not "cut-off" parts of the robot.

When choosing the resolution of the full camera images, no improvements were measurable when going above 400x300. The computation time however increased noticeably. Therefore for the training we mostly used images of 400x300 Pixels in size as the input for our First Stage.

### The detection model
We mainly used a Single Shot Detector approach based on the MobileNet v2 Architecture for Object Detection. This proved to be a problem with small objects such as Spheros, since SSD uses a drastically downscaled feature map on the entire image to detect objects while saving computational power. For large objects that cover multiple "Pixels" within this feature map, the model is then able to make out accurate bounding boxes containing these. Since Spheros only cover 25x25 of the entire 1600x1200 Pixel area, this feature map either has have a very high resolution, or a single Sphero will merely be detected by one Pixel in this feature map, resulting in poor accuracy. For more information on the Single Shot Detector architecture, refer to the [official paper](https://arxiv.org/pdf/1512.02325.pdf).

One solution the is more invariant to object size is Faster RCNN. This architecture however at its' fastest still takes about twice as much time for a detection as an SSD, while being much more computationally expensive.

Further descriptions about which parameters were used in conjunction with those models can be found in the [models directory](./FirstStage/models).

### Evaluation
#### No-detection-rate
For evalutation purposes we drove a single Sphero through the arena on a circular path for 5 times while logging the detections on every camera image. We then evaluated how often no Sphero was detected in an input image. These are the results:

| Farbe      | %     |
|------------|-------|
| purple     | 4.73  |
| blue       | 4.96  |
| green      | 6.27  |
| magenta    | 6.34  |
| dark_blue  | 8.29  |
| red        | 13.38 |
| dark_green | 22.70 |
| lime_green | 6.79  |
| light_blue | 8.04  |
| yellow     | 11.05 |

For the last three lines, the Spheros were not able to navigate the path on their own, indicating that the Second Stage most likely had problems accurately identifying these colours. It is however apparent that dark colours are harder to continuously detect than light ones. When looking at `dark_green` though, even though the Sphero was only detected in about 78% of all input images, those were still enough to allow for a continuous navigation without any "dead ends".

#### Processing time
After fine tuning the parameters of the SSD MobileNet v2 we were able to reach a total processing time of 1600x1200 images of 16.14ms. This is an improvement of 3ms in comparison to the baseline MobileNet v2 (19.14ms).

## Second stage
[SecondStage directory](./SecondStage/)  
The second stage determines the colour and rotation in two separate convolutional neural networks. Both networks use a pre-trained MobileNet with a custom output appended at the end. The custom output at the end always contains every output, that is for binning (binning of the angles), regression and the identification network. When training for a specific output the losses of the other outputs are automatically set to zero so that only the desired output is trained. This can be changed but we refrained from doing so because if changed it requires more complicated changes in the ROS framework used, because the networks now only contain 1 output instead of 3.  
There are two parameters with which the MobileNet can be tweaked. The 'alpha' value and 'dropout'. Dropout controls the amount of dropout in the network.
Alpha controls the width of the network. This is known as the width multiplier in the MobileNet paper.
* If 'alpha' < 1.0, proportionally decreases the number of filters in each layer.
* If 'alpha' > 1.0, proportionally increases the number of filters in each layer.
* If 'alpha' = 1, default number of filters from the paper are used at each layer.

In some instances it could be observed that a too high 'alpha' value tends to overfit on the data. For these instances lower 'alpha' values and a carefully set amount of 'dropout' reached good results. This is heavily dependend on the data used and should always be figured out by experiments including mixtures of several 'alpha' and 'dropout' values. The default 'alpha' value we began with was '0.5'.
'Adam' was always used as the optimizer.  
The learning rate can be lowered to get a more detailed learning curve and may also prevent overfitting if set to lower values, default value we used is '1e-3'.

Additionally to the MobileNet it is possible to use a custom network. We tested this custom nework and achieved good validation results but decided to use the MobileNet instead, because we were unsure which other changes a custom network with a few layers would introduce in the test results and the MobileNet worked very well (never change a running system?). Another reason is that the custom network should be heavily optimized (number of layers, number of neurons, dropout, ...) and we were limited in computation power and time to do so.
The custom network could potentially produce better results for the spheros.


### Training Data

We used 17 crops of each colour to generate our data sets. The crops were divided into 11 crops for training and 6 to test the result. The compositor gives the opportunity to configure the amount of variations (number of times the 360° rotations are repeated) of each image used for the data sets. We used 9 variations. Less than 9 seemed to underfit on the test data. Way more than 9 tends to overfit on the training set and has a higher risk of overfitting on artificially generated images. With a different amount of colours and crops per colour, the number of variations probably needs to be adapted.

### Identification Net

The critical part for achieving a reliable identification is choosing the right colours. Spheros accept any RGB colour with values between 0 and 255, but aren't able to visualize all of them. To select the most different colours, we filtered by using high value differences and trusting on our eyes to detect the best ones. Maybe better results could be achieved with analysing the RGB values of the images the camera records, but there where no signs that the camera makes colours more distinguishable. Rather, the low resolution lead to very similar images of colours, we could easily tell apart.

Another problem is, that the colour representation varies widely depending on the position in the arena, daylight and symbols on the spheros that could cover the LEDs. With our sample of crops, it wasn't possible to generate data sets that include all the different shades. Increasing or decreasing the brightness in the generated images and adding noise while training, made the network more robust. A higher amount of training data should improve the performance tremendously. 

### Rotation Net

The rotation  of a sphero is determined dependent on the position of the back led in relation to the two other LEDs. We already achieved good training results with the default configuration. Using mirrored data further increased the performance (see compositor). 

But the rotation is never a reliable value. In motion, different camera angles and due to the marks, LEDs are often covered, which leads to a wrong prediction. Because we used binning, there will always be an output between 0 and 360 degree. Classification with an additional 'not predictable' class, could solve that problem, but maybe doesn't work as good. Otherwise, the output is just usable as a reference value when standing still.


## Problems
This section is dedicated to various problems/anomalies encountered during the project.

### Sphero
The colours displayed by the Spheros' RGB LEDs are not "uniform". The three LEDs are visibly distinguishable from each other.

The brightness of all LEDs are not equal. The blue LED for example heavily outshines the other two LEDs, making most mix-colours appear mostly blue. The red LED seems to be the darkest.

Because of the imprints on the clear hull of the Spheros, often some of the LEDs are covered, leading to erroneous detections. This is especially problematic if the blue LED used for rotational identification is covered up. Another source of this problem is the tilting of the internal circuit board of the Sphero during movements.
