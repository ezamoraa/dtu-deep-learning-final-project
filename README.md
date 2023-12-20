# LIDAR 3D POINT CLOUD DEEP COMPRESSION

Reference implementation of the publication __3D Point Cloud Compression with Recurrent Neural Network
and Image Compression Methods__ published on the 33rd IEEE Intelligent Vehicles Symposium (IV 2022)
in Aachen, Germany.

This repository implements a RNN-based compression framework for range images implemented in Tensorflow 2.9.0.
Furthermore, it implements preprocessing and inference nodes for ROS Noetic in order to perform the compression
task on a `/points2` sensor data stream.

> **LIDAR 3D POINT CLOUD DEEP COMPRESSION**
> Esteban Zamora, Lucas M. Sandby, Rolando Esquivel-Sancho, Steven Tran
> 
> _**Abstract**_ — 
> LiDARs are vital for autonomous vehicles, and supplying 3D data is crucial for SLAM and object detection. However, the substantial data output poses storage and transmission challenges, and thus compression is crucial to reduce latency and optimize storage. The following paper will describe and evaluate a self-supervised deep compression approach \cite{germans_rnn} to compress and reconstruct 3D LiDAR scans based on convolutional recurrent neural networks. The architecture is adjusted and implemented in PyTorch while inference is being run in ROS (Robot Operating System) to provide a fully functional implementation for robotics. Evaluating the trade-off between inference speed and reconstruction quality in contrast to latent space (encoded size) and network size we find a direct correlation between reducing error and the size of the latent space as well as the error being dependent on the network size, but at the cost of inference time.




## Range Image Compression


![](assets/training_reconstruction.png)


## Model Training

The training process is described in `range_image_compression/README.md`



## Inference & Evaluation
The inference is implemented in the robotics application Robot Operating System (ROS). The model is implemented as multiple ROS nodes that perform the point cloud pre-processing, encoding, and decoding stages, communicating by passing messages through topics. In order to run the inference and evaluation you will need to execute the following steps.


### 1. Pull Docker Image
You can pull this dockerfile from [Docker Hub](https://hub.docker.com/r/rolandoesq/ros-noetic-pytorch)
with the command:
```bash
docker pull rolandoesq/ros-noetic-pytorch
```

### 2. Download ROS Bag
Perform the same procedure with the bag file. Copy the bag file into `catkin_ws/rosbags`. The filename should be
`evaluation_frames.bag`. This bag file will be used in this [launch file](catkin_ws/src/pointcloud_to_rangeimage/launch/compression.launch).

### 3. Start Docker Container
Start the docker container using the script
```bash
# /docker >$
./docker_eval.sh
```
Instructions in `/docker/README.md`
