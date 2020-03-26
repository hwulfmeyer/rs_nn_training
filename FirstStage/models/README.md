# Working Tensorflow Object Detection models
This folder contains two working model files that can be used for Sphero detection. The chosen parameters for their respective pipeline files are discussed in the following sections.

## SSD MobileNet v2
This model is based on the ssd_mobilenet_v2 architecture.

### anchor_generator
Here we should only have aspect ratios that a bounding box around our tracked objects could actually take (e.g. Spheros are always round, so their bounding boxes are always going to have an aspect ratio of 1:1). The float value n of the respective aspect ratio is interpreted as n:1 (e.g. `aspect_ratios: 2.0` would mean 2:1)

If your images are always the exact same minimum and maximum scale, it can work to set the `min_scale` and `max_scale` to exactly those sizes (relative to the whole image). E.g. if your objects take from 5% to 50% of the whole image, set the min_scale to 0.05 and max_scale to 0.5. When doing this, make sure `ssd_random_crop` is not set within `data_augmentation_options`, otherwise the scales will not apply to those cropped images anymore.

### fixed_shape_resizer
Put the desired size that your images should be resized to.

### loss
When only very few object classes should be detected (e.g. either nothing or Sphero), it helps changing the `loss_type` to `BOTH` instead of `CLASSIFICATION`. This will take into account the localization loss of the bounding box as well. You can then also adjust the weight applied to either of the loss types.

### dropout
When realizing that the model generalizes poorly and produces a low loss on the training, but a high one on the evaluation set, think about using dropout. For the mobilenet model, this did however not produce any noticeable improvement.

## Faster RCNN Inception v2
This model was trained with its' default pipeline configuration. Since the accuracy of the predictions was already very good, there has not been much testing on trying to improve the performance.
