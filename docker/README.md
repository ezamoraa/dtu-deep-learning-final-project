# ROS Point Cloud Compression

This repository contains code related to point cloud compression using ROS.


## Docker Image

Use the Docker image `rolandoesq/ros-noetic-pytorch`. Pull the image using the following command:

```bash
docker pull rolandoesq/ros-noetic-pytorch
```


## Setup

Before running the point cloud compression node, make sure to follow these setup instructions:

0. First make sure that the git submodules are initialized for the project repo (needed for velodyne driver):

    ```bash
    git submodule init
    git submodule update
    ```

1. Update ROS dependencies:

    ```bash
    rosdep update
    ```

2. Install ROS dependencies for the source packages:

    ```bash
    rosdep install --from-paths ./src --ignore-packages-from-source -y
    ```

3. Clean the Catkin workspace:

    ```bash
    catkin clean -y
    ```

4. Build the Catkin workspace:

    ```bash
    catkin build
    ```

5. Source the workspace setup file:

    ```bash
    source devel/setup.bash
    ```

## Point Cloud Compression

After setting up the workspace, follow these steps to run the point cloud compression from catkin_ws:

1. Install the range_image_compression Python package using pip:

    ```bash
    pip3 install -e ../range_image_compression/
    ```

2. Launch the point cloud compression node:

    ```bash
    roslaunch pointcloud_to_rangeimage compression.launch
    ```

This will start the point cloud compression node with the specified launch file.
