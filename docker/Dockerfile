FROM ros:noetic-ros-base-focal

# Update and install prerequisites
RUN apt update && apt install -y curl gnupg2 lsb-release

# Add ROS repository
RUN sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'

# Add ROS key
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -

# Update again to include ROS packages
RUN apt update

# install ros packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-perception=1.5.0-1* \
    rviz \
    ros-noetic-rviz \
    python3-pip \ 
    python3-rosdep \
    python3-rosinstall \
    python3-rosinstall-generator \
    python3-wstool \
    python3-catkin-tools \
    build-essential \
    vim 

RUN python3 -m pip install torch torchvision torchaudio pyyaml numpy 

# Source ROS setup.bash
RUN echo "source /opt/ros/noetic/setup.bash" >> ~/.bashrc

# Set environment variables
ENV MESA_LOADER_DRIVER_OVERRIDE=i965
ENV QT_X11_NO_MITSHM 1
ENV CATKIN_WS /ws/catkin_ws
ENV ROS_DISTRO noetic

# Create a catkin workspace and build it
RUN mkdir -p $CATKIN_WS/src && \
    cd $CATKIN_WS && \
    /bin/bash -c '. /opt/ros/noetic/setup.bash; catkin_make'

# Source the workspace setup.bash
RUN echo "source $CATKIN_WS/devel/setup.bash" >> ~/.bashrc



# Clean up
#RUN apt-get clean && rm -rf /var/lib/apt/lists/

# Start a shell by default
CMD ["/bin/bash"]



