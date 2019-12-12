import rospy
import threading
import numpy as np
import time
import atexit
from std_msgs.msg import String

downscaling_factor = 0.25
heatmap = np.zeros(int(1600 * downscaling_factor), int(900 * downscaling_factor))
new_calls = 0
current_x = 0
current_y = 0

def callback(data):
    global current_x
    global current_y
    global new_calls

    print(data.data)
    current_x = 0 # replace with value from topic
    current_y = 0 # replace with value from topic
    new_calls += 1

def save_state():
    global new_calls
    global current_x
    global current_y
    global heatmap

    while(true):
        heatmap[int(current_x * downscaling_factor), int(current_y * downscaling_factor)] = new_calls
        print("Saved " + new_calls + " values at " + current_x + ", " + current_y)
        new_calls = 0
        time.sleep(1)

def save_heatmap():
    global heatmap
    np.save("heatmap", heatmap)

atexit.register(save_heatmap)

heatmap_thread = threading.Thread(target = save_state())
heatmap_thread.start()

subscriber = rospy.Subscriber("name", String, callback)
rospy.spin()
