import rospy
import torch
import cv2
import numpy as np

from cv_bridge import CvBridge

from pointcloud_to_rangeimage.msg import RangeImage as RangeImage_msg
from pointcloud_to_rangeimage.msg import RangeImageEncoded as RangeImageEncoded_msg

from range_image_compression.models.additive_lstm import LidarCompressionNetwork
from range_image_compression.utils import log_execution_time

# NOTE: The size of the LIDAR range image is 32x1812 -> padded to 32x1824
# (height, width)
RANGE_IMG_SIZE = (32, 1812)
RANGE_IMG_WIDTH_PAD = 12
RANGE_IMG_SIZE_PADDED = (RANGE_IMG_SIZE[0], RANGE_IMG_SIZE[1]+RANGE_IMG_WIDTH_PAD)

CODE_RATIO = LidarCompressionNetwork.CODE_IMG_DIM_RATIO
CODE_SIZE = (RANGE_IMG_SIZE_PADDED[0] // CODE_RATIO,
             RANGE_IMG_SIZE_PADDED[1] // CODE_RATIO)

class MsgDecoder:
    """
    Subscribe to topic /msg_encoded published by the encoder.
    Decompress the images and pack them in message type RangeImage.
    Publish message to the topic /msg_decoded.
    """
    def __init__(self, demo=False):
        self.pub = rospy.Publisher("msg_decoded", RangeImage_msg, queue_size=10)
        self.sub = rospy.Subscriber("/msg_encoded", RangeImageEncoded_msg, self.callback)
        self.bridge = CvBridge()

        bottleneck = rospy.get_param("/rnn_compression/bottleneck")
        num_iters = rospy.get_param("/rnn_compression/num_iters")
        weights_path = rospy.get_param("/rnn_compression/weights_path")

        self.num_iters = num_iters

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.decoder = LidarCompressionNetwork(
            bottleneck=bottleneck,
            num_iters=num_iters,
            image_size=RANGE_IMG_SIZE_PADDED,
            device=self.device,
            mode=LidarCompressionNetwork.MODE_INFERENCE_DECODE,
            demo=demo,
        )

        # Load model weights
        state_dict = torch.load(weights_path)
        if weights_path.endswith('.tar'):
            # Assume that checkpoint state is read from files ending with .tar
            model_state_dict = state_dict['model_state_dict']
        else:
            # Assume that the final model state is read from a file ending with other extension (.pt,.pth)
            model_state_dict = state_dict
        self.decoder.load_state_dict(model_state_dict)
        self.decoder.to(self.device)

        # Initialize decoder with zero input
        zero_codes = []
        for i in range(num_iters):
            zero_codes.append(torch.zeros((1, bottleneck, CODE_SIZE[0], CODE_SIZE[1]), device=self.device))
        self.decoder(zero_codes)
    @log_execution_time(out_path="/catkin_ws/decode_time.csv")
    def callback(self, msg):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.send_time)

        # Unpack compressed binary bitstreams
        shape = (msg.shape[0], msg.shape[1], msg.shape[2], msg.shape[3], msg.shape[4])
        code = np.unpackbits(np.fromstring(msg.code, np.uint8))
        code = code.astype(np.float32)*2-1
        encoded_vec = code.reshape(shape)
        codes = []
        for i in range(encoded_vec.shape[0]):
            code_tensor = torch.from_numpy(encoded_vec[i]).to(self.device)
            codes.append(code_tensor)

        # Run range image decoder model
        out = self.decoder(codes)
        decoded_vec = out["outputs"].cpu().detach().numpy()

        # Decode the images
        range_image_decoded = (decoded_vec[0, 0, :, :RANGE_IMG_SIZE[1]]*65535).astype(np.uint16)
        azimuth_map_array = np.fromstring(msg.AzimuthMap, np.uint8)
        azimuth_map_decoded = cv2.imdecode(azimuth_map_array, cv2.IMREAD_UNCHANGED)
        intensity_map_array = np.fromstring(msg.IntensityMap, np.uint8)
        intensity_map_decoded = cv2.imdecode(intensity_map_array, cv2.IMREAD_UNCHANGED)

        # Convert OpenCV image to ROS image.
        try:
            range_image = self.bridge.cv2_to_imgmsg(range_image_decoded, encoding="mono16")
        except CvBridgeError as e:
            print(e)
        try:
            intensity_map = self.bridge.cv2_to_imgmsg(intensity_map_decoded, encoding="mono8")
        except CvBridgeError as e:
            print(e)
        try:
            azimuth_map = self.bridge.cv2_to_imgmsg(azimuth_map_decoded, encoding="mono16")
        except CvBridgeError as e:
            print(e)

        # Pack images in ROS message.
        msg_decoded = RangeImage_msg()
        msg_decoded.header = msg.header
        msg_decoded.send_time = msg.send_time
        msg_decoded.RangeImage = range_image
        msg_decoded.IntensityMap = intensity_map
        msg_decoded.AzimuthMap = azimuth_map
        msg_decoded.NansRow = msg.NansRow
        msg_decoded.NansCol = msg.NansCol

        self.pub.publish(msg_decoded)

def main():
    method = rospy.get_param("/compression_method")
    if method in ["additive_lstm", "additive_lstm_demo"]:
        decoder = MsgDecoder(demo=method.endswith("_demo"))
    else:
        raise NotImplementedError

    rospy.init_node('compression_decoder', anonymous=True)
    rospy.spin()


if __name__ == '__main__':
    main()
