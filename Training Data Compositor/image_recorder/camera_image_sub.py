import rospy
import cv2
import time
import sys
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


last_time = time.time()
img_suffix = "8"
if len(sys.argv) >= 2:
	img_suffix = sys.argv[1]
	

def image_callback(data):
	global last_time
	try:
		image = bridge.imgmsg_to_cv2(data, "bgr8")
	except CvBridgeError as e:
		print(e)
	path = "images/{}_{}.png".format(int(time.time()), img_suffix)
	if  time.time() - last_time > 1:
		last_time = time.time()
		cv2.imwrite(path, image)


bridge = CvBridge()
image_sub = rospy.Subscriber("camera/0", Image, image_callback)
rospy.init_node('image_handler', anonymous=True)
rospy.spin()
