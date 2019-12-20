import rospy
import cv2
import time
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


last_time = time.time()

def image_callback(data):
    global last_time
    try:
        image = bridge.imgmsg_to_cv2(data, "passthrough")
    except CvBridgeError as e:
        print(e)
    path = "images/{}.jpg".format(int(time.time()))
    if  time.time() - last_time > 3:
        last_time = time.time()
    	cv2.imwrite(path, image)


bridge = CvBridge()
image_sub = rospy.Subscriber("camera/0", Image, image_callback)
rospy.init_node('image_handler', anonymous=True)
rospy.spin()
