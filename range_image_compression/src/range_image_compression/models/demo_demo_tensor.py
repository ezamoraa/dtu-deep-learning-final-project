import rospy

from pointcloud_to_rangeimage.msg import RangeImage as RangeImage_msg
from pointcloud_to_rangeimage.msg import RangeImageEncoded as RangeImageEncoded_msg

import numpy as np


class MsgEncoder:
    def __init__(self):
        self.pub = rospy.Publisher('msg_encoded', RangeImage_msg, queue_size=10)
        self.sub = rospy.Subscriber("/pointcloud_to_rangeimage_node/msg_out", RangeImage_msg, self.callback)

    def callback(self, msg):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.send_time)
        # Forward the received message to the output
        self.pub.publish(msg)

class MsgDecoder:
    def __init__(self):
        self.pub = rospy.Publisher("msg_decoded", RangeImage_msg, queue_size=10)
        self.sub = rospy.Subscriber("/msg_encoded", RangeImage_msg, self.callback)

    def callback(self, msg):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.send_time)
        # Forward the received message to the output
        self.pub.publish(msg)