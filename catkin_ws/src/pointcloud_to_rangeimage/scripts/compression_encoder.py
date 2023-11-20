import rospy
import torch
import cv2
import numpy as np

from cv_bridge import CvBridge

from pointcloud_to_rangeimage.msg import RangeImage as RangeImage_msg
from pointcloud_to_rangeimage.msg import RangeImageEncoded as RangeImageEncoded_msg

from range_image_compression.models.additive_lstm import LidarCompressionNetwork

# NOTE: The size of the LIDAR range image is 32x1812 -> padded to 32x1824
# height, width
RANGE_IMG_SIZE = (32, 1812)
RANGE_IMG_WIDTH_PAD = 12
RANGE_IMG_SIZE_PADDED = (RANGE_IMG_SIZE[0], RANGE_IMG_SIZE[1]+RANGE_IMG_WIDTH_PAD)

class MsgEncoder:
    """
    Subscribe to topic /pointcloud_to_rangeimage_node/msg_out,
    compress range image using RNN image compression model,
    azimuth image using JPEG2000 and intensity image using PNG compression.
    Publish message type RangeImageEncoded to topic /msg_encoded.
    """
    def __init__(self, demo=False):
        self.pub = rospy.Publisher('msg_encoded', RangeImageEncoded_msg, queue_size=10)
        self.sub = rospy.Subscriber("/pointcloud_to_rangeimage_node/msg_out", RangeImage_msg, self.callback)
        self.bridge = CvBridge()

        bottleneck = rospy.get_param("/rnn_compression/bottleneck")
        num_iters = rospy.get_param("/rnn_compression/num_iters")
        weights_path = rospy.get_param("/rnn_compression/weights_path")

        self.num_iters = num_iters

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.encoder = LidarCompressionNetwork(
            bottleneck=bottleneck,
            num_iters=num_iters,
            image_size=RANGE_IMG_SIZE_PADDED,
            device=self.device,
            mode=LidarCompressionNetwork.MODE_INFERENCE_ENCODE,
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
        self.encoder.load_state_dict(model_state_dict)
        self.encoder.to(self.device)

        # Initialize encoder with zero input
        input = torch.zeros((1, 1, RANGE_IMG_SIZE_PADDED[0], RANGE_IMG_SIZE_PADDED[1]), device=self.device)
        self.encoder(input)

    def parse_img(self, range_image):
        """
        Preprocessing of image array for the RNN image compression model.
        Normalize image array and pad it to be divisible by 32. Details can be found in the thesis.
        """
        range_image = np.reshape(range_image, (1, 1, RANGE_IMG_SIZE[0], RANGE_IMG_SIZE[1])) / 65535
        image_vec = np.concatenate((range_image, range_image[:, :, :, :RANGE_IMG_WIDTH_PAD]), axis=3)
        return image_vec

    def callback(self, msg):
        rospy.loginfo(rospy.get_caller_id() + "I heard %s", msg.send_time)
        try:
            range_image = self.bridge.imgmsg_to_cv2(msg.RangeImage, desired_encoding="mono16")
        except CvBridgeError as e:
            print(e)
        try:
            intensity_map = self.bridge.imgmsg_to_cv2(msg.IntensityMap, desired_encoding="mono8")
        except CvBridgeError as e:
            print(e)
        try:
            azimuth_map = self.bridge.imgmsg_to_cv2(msg.AzimuthMap, desired_encoding="mono16")
        except CvBridgeError as e:
            print(e)

        # Pack binary bitstreams into numpy int8 arrays.
        image_vec = self.parse_img(range_image)
        image_tensor = torch.from_numpy(image_vec).to(dtype=torch.float32).to(self.device)

        # Run range image encoder model
        out = self.encoder(image_tensor)
        codes = [code.cpu().detach().numpy() for code in out["codes"]]

        # Prepare encoded range image message
        codes = (np.stack(codes).astype(np.int8) + 1) // 2
        shape = codes.shape
        codes = np.packbits(codes.reshape(-1))

        msg_encoded = RangeImageEncoded_msg()
        msg_encoded.header = msg.header
        msg_encoded.send_time = msg.send_time
        msg_encoded.code = codes.tostring()
        msg_encoded.shape = [shape[0], shape[1], shape[2], shape[3], shape[4]]
        msg_encoded.NansRow = msg.NansRow
        msg_encoded.NansCol = msg.NansCol

        # Compress azimuth image and intensity image with JPEG 2000 and PNG.
        params_jp2 = [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, 100]
        params_png = [cv2.IMWRITE_PNG_COMPRESSION, 10]
        _, azimuth_map_encoded = cv2.imencode('.jp2', azimuth_map, params_jp2)
        msg_encoded.AzimuthMap = azimuth_map_encoded.tostring()
        _, intensity_map_encoded = cv2.imencode('.png', intensity_map, params_png)
        msg_encoded.IntensityMap = intensity_map_encoded.tostring()

        self.pub.publish(msg_encoded)

def main():
    method = rospy.get_param("/compression_method")
    if method in ["additive_lstm", "additive_lstm_demo"]:
        encoder = MsgEncoder(demo=method.endswith("_demo"))
    else:
        raise NotImplementedError

    rospy.init_node('compression_encoder', anonymous=True)
    rospy.spin()


if __name__ == '__main__':
    main()
