# Evaluation

## Evaluating the bounding boxes of your model
Move the `frozen_inference_graph.pb` and `label_map.pbtxt` file of your model into this folder, and put your sample images into the `images` directory. Then run `python run_inference.py`.

## Evaluating the detection time of your model
Follow the same steps as before, but instead run `python detection_time_evaluation.py`
