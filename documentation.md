# WS 19/20 - Rolling Swarm Camera Tracking
The camera tracking module is responsible for detecting various robots in a camera image and labeling them according to their colour. For this project the focus was lying entirely on the Sphero platform, however some of the results may also be applicable to different robots.

The goals for this project were:
* improving the detection rate at different positions inside the arena
 * previously certain positions, especially near edges, led to erroneous detections
* improving the amount of colours that can be detected independently
* cleaning up and improving the training pipeline for the Two-Stage Detector architecture
* updating deprecated parts of the code in conjunction with current framework versions

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

## Compositor

## Problems
This section is dedicated to various problems/anomalies encountered during the project.

### Sphero
The colours displayed by the Spheros' RGB LEDs are not "uniform". The three LEDs are visibly distinguishable from each other.

The brightness of all LEDs are not equal. The blue LED for example heavily outshines the other two LEDs, making most mix-colours appear mostly blue. The red LED seems to be the darkest.

Because of the imprints on the clear hull of the Spheros, often some of the LEDs are covered, leading to erroneous detections. This is especially problematic if the blue LED used for rotational identification is covered up. Another source of this problem is the tilting of the internal circuit board of the Sphero during movements.
