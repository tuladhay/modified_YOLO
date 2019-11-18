# Modified_YOLO for relative pose estimation

This is a modified version of YOLO2. It has few more outputs in the final convolutional layer such that it can detect pose x, y, z of a single object. I was using this to get bounding boxes around CargoDoor, and estimate the pose of the door from the camera. The data collection and application implementation was in ROS Indigo.

This was one of the projects during my summer 2018 internship. The objective this project was to be able to detect cargo doors in the image using YOLO, and also be able to predict the relative distance and orientation from the camera frame.
