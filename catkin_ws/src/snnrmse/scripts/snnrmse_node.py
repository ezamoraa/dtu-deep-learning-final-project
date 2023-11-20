import rospy

import numpy as np

import message_filters

from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2

from sklearn.neighbors import KDTree


class SNNRMSE:
    """
    Implements the SNNRMSE metric to compute a distance between two point clouds.
    """
    def callback(self, pc1, pc2, symmetric=True):
        pc1_xyzi = np.asarray(point_cloud2.read_points_list(
            pc1, field_names=("x", "y", "z", "intensity"), skip_nans=True), dtype=np.float32)
        pc2_xyzi = np.asarray(point_cloud2.read_points_list(
            pc2, field_names=("x", "y", "z", "intensity"), skip_nans=True), dtype=np.float32)

        tree1 = KDTree(pc1_xyzi[:, :3], leaf_size=2, metric='euclidean')
        sqdist1, nn1 = tree1.query(pc2_xyzi[:, :3], k=1)
        mse1 = np.sum(np.power(sqdist1, 2)) / sqdist1.shape[0]
        msei1 = np.sum(np.power(pc2_xyzi[:, 3] - pc1_xyzi[nn1[:, 0], 3], 2)) / pc2_xyzi.shape[0]

        if symmetric:
            tree2 = KDTree(pc2_xyzi[:,:3], leaf_size=2, metric='euclidean')
            sqdist2, nn2 = tree2.query(pc1_xyzi[:, :3], k=1)
            mse2 = np.sum(np.power(sqdist2, 2)) / sqdist2.shape[0]
            msei2 = np.sum(np.power(pc1_xyzi[:, 3] - pc2_xyzi[nn2[:, 0], 3], 2))/pc1_xyzi.shape[0]
            snnrmse = np.sqrt(0.5*mse1 + 0.5*mse2)
            snnrmsei = np.sqrt(0.5*msei1 + 0.5*msei2)
        else:
            snnrmse = np.sqrt(mse1)
            snnrmsei = np.sqrt(msei1)

        pointlost = pc1_xyzi.shape[0] - pc2_xyzi.shape[0]

        self.sum_snnrmse = self.sum_snnrmse + snnrmse
        self.sum_snnrmsei = self.sum_snnrmsei + snnrmsei
        self.sum_pointlost = self.sum_pointlost + pointlost
        self.sum_points = self.sum_points + pc1_xyzi.shape[0]

        metrics = {
            "frame": pc1.header.seq+1,
            "n_points_original": pc1_xyzi.shape[0],
            "width": pc1.width,
            "height": pc2.height,
            "current_point_lost": pointlost,
            "current_geometry_snnrmse": snnrmse,
            "current_intensity_snnrmse": snnrmsei,
            "avg_geometry_snnrmse": self.sum_snnrmse / (pc1.header.seq+1),
            "avg_intensity_snnrmse": self.sum_snnrmsei/ (pc1.header.seq+1),
            "avg_point_lost": self.sum_pointlost / (pc1.header.seq+1),
            "average_point_number": self.sum_points / (pc1.header.seq+1)
        } 

        print("==============EVALUATION==============")
        print("Frame:", metrics["frame"])
        print("Number of points in original cloud:", metrics["n_points_original"])
        print("Shape:", metrics["width"], metrics["height"])
        print("Current Point Lost:", metrics["current_point_lost"])
        print("Current Geometry SNNRMSE:", metrics["current_geometry_snnrmse"])
        print("Current Intensity SNNRMSE:", metrics["current_intensity_snnrmse"])
        print("Average Geometry SNNRMSE:", metrics["avg_geometry_snnrmse"])
        print("Average Intensity SNNRMSE:", metrics["avg_intensity_snnrmse"])
        print("Average Point Lost:", metrics["avg_point_lost"])
        print("Average Point Number:", metrics["average_point_number"])
        print("======================================")
        
        save_path = '/catkin_ws/snnrmse_metrics.txt' 
        with open(save_path, 'a') as file:
            if metrics["frame"] == 1:
                file.write(",".join(metrics.keys()))
            file.write("\n")
            file.write(",".join([str(metric) for metric in metrics.values()]))
            
            
    def __init__(self):
        # initialize ROS node
        rospy.init_node('snnrmse_node', anonymous=False)

        original_cloud = message_filters.Subscriber('/points2', PointCloud2)
        decompressed_cloud = message_filters.Subscriber('/decompressed', PointCloud2)

        self.sum_snnrmse = 0
        self.sum_snnrmsei = 0
        self.sum_pointlost = 0
        self.sum_points = 0

        ts = message_filters.TimeSynchronizer([original_cloud, decompressed_cloud], 10)
        ts.registerCallback(self.callback)

        rospy.spin()


if __name__ == '__main__':
    snnrmse_node = SNNRMSE()
