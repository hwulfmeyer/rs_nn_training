import rospy
import threading
import numpy as np
import time
import atexit
from tracking_msgs.msg import TaggedPose2D

downscaling_factor = 0.05
new_calls = 0
current_x = 0
current_y = 0

try:
    heatmap = np.load("heatmap.npy")
except:
    heatmap = np.zeros((int(1600 * downscaling_factor), int(1200 * downscaling_factor)))

def callback(data):
    global current_x
    global current_y
    global new_calls
	
    current_x = int(data.x) # replace with value from topic
    current_y = int(data.y) # replace with value from topic
    new_calls += 1

def save_state():
    global new_calls
    global current_x
    global current_y
    global heatmap

    while(True):
        heatmap[int(current_x * downscaling_factor), int(current_y * downscaling_factor)] = new_calls
        np.save("heatmap", heatmap)
        print("Saved " + str(new_calls) + " values at " + str(current_x) + ", " + str(current_y))
        new_calls = 0
        time.sleep(0.5)

def save_heatmap():
    global heatmap
    np.save("heatmap", heatmap)

atexit.register(save_heatmap)

heatmap_thread = threading.Thread(target = save_state)
heatmap_thread.start()

print("Initializing subscriber")
rospy.init_node("subscriber")
subscriber = rospy.Subscriber("/nn_track0/eval/sphero_bright_blue", TaggedPose2D, callback)
rospy.spin()
