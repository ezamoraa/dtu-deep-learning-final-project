#import tensorflow as tf
import rospy
from models import demo_demo_tensor

# physical_devices = tf.config.list_physical_devices('GPU')
# if len(physical_devices) > 0:
#     tf.config.experimental.set_memory_growth(physical_devices[0], True)

# if rospy.get_param("/rnn_compression/xla"):
#     tf.config.optimizer.set_jit("autoclustering")

# if rospy.get_param("/rnn_compression/mixed_precision"):
#     tf.keras.mixed_precision.set_global_policy("mixed_float16")


def main():
    method = rospy.get_param("/compression_method")
    if method == "demo":
        decoder = demo_demo_tensor.MsgDecoder()
    else:
        raise NotImplementedError

    rospy.init_node('compression_decoder', anonymous=True)
    rospy.spin()


if __name__ == '__main__':
    main()
