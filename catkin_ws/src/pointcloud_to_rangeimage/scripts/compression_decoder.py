import tensorflow as tf
import rospy
from architectures import additive_lstm, additive_lstm_slim, oneshot_lstm, additive_gru, image_compression

physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

if rospy.get_param("/rnn_compression/xla"):
    tf.config.optimizer.set_jit("autoclustering")

if rospy.get_param("/rnn_compression/mixed_precision"):
    tf.keras.mixed_precision.set_global_policy("mixed_float16")


def main():
    method = rospy.get_param("/compression_method")
    if method == "image_compression":
        decoder = image_compression.MsgDecoder()
    elif method == "additive_lstm":
        decoder = additive_lstm.MsgDecoder()
    elif method == "additive_lstm_slim":
        decoder = additive_lstm_slim.MsgDecoder()
    elif method == "oneshot_lstm":
        decoder = oneshot_lstm.MsgDecoder()
    elif method == "additive_gru":
        decoder = additive_gru.MsgDecoder()
    else:
        raise NotImplementedError

    rospy.init_node('compression_decoder', anonymous=True)
    rospy.spin()


if __name__ == '__main__':
    main()
