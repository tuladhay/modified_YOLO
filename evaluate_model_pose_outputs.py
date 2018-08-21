#! /usr/bin/env python

import os
import cv2
import matplotlib.pyplot as plt
from frontend import YOLO
import json
import pandas as pd


''' This file will read the video file created from the images (see img2video.py), it will run it through the
modified network and get the pose_x. It will then calculate the error abs(actual-predicted), and the plot it '''

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # "0" use cpu

# CONFIGURATION
image_path = '/home/ubuntu/tensorflow_data/output.mp4'
config_path = 'config_cargo_door.json'
weights_path = '/home/ubuntu/tensorflow_data/YOLO/CargoDoor/full_yolo_cargo_door_with_pose_v6.h5'

with open(config_path) as config_buffer:
    config = json.load(config_buffer)

###############################
#   Make the model
###############################
yolo = YOLO(architecture=config['model']['architecture'],
            input_size=config['model']['input_size'],
            labels=config['model']['labels'],
            max_box_per_image=config['model']['max_box_per_image'],
            anchors=config['model']['anchors'])

###############################
#   Load trained weights
###############################
print(weights_path)
yolo.load_weights(weights_path)

###############################
#   Load true pose data
###############################
csv_filename = '/home/ubuntu/catkin_ws/src/Northstar/airplane_loader/scripts/image_pose_data.csv'
true_pose_x = pd.read_csv(csv_filename, usecols=['pose_x'])
true_pose_x = true_pose_x.values

###############################
#   Run modified yolo
###############################
counter = 0

if image_path[-4:] == '.mp4':
    video_reader = cv2.VideoCapture(image_path)

    nb_frames = int(video_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
    print("The number of frames are: " + str(nb_frames + 1))
    frame_h = int(video_reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))

    error = []    # store per image rpy loss in an list
    pred_pose_list = []
    true_pose_list = []

    for i in range(nb_frames):
        print("\npredicting for frame: " + str(i))
        _, image = video_reader.read()
        _, rpy = yolo.predict(image)
        #print(rpy)
        # print("true_pose_x: " + str(true_pose_x[i]) + "\tpredicted pose_x: " + str(rpy[0]))
        if any(rpy):
            loss = abs(true_pose_x[i] - rpy[0])
            error.append(loss)
            pred_pose_list.append(rpy[0])
            true_pose_list.append(true_pose_x[i])


    samples = [s for s in range(len(true_pose_list))]
    plt.plot(samples,  true_pose_list, color='skyblue')
    plt.plot(samples, pred_pose_list, color='red')
    plt.xlabel('frames with detected CargoDoors')
    plt.title('True(blue) vs Predicted(red) values for pose_x using modified YOLO')
    plt.ylabel('pose_x in radians')
    plt.show()
    video_reader.release()
