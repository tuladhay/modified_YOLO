#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np

from utils import draw_boxes
from frontend import YOLO
import json
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #"0" use cpu

#Simple function to predict bounding boxes using YOLO

def _main_():

    image_path = '/home/ubuntu/tensorflow_data/YOLO/CargoDoor/train_images/0.jpeg' #'/home/ubuntu/tensorflow_data/output.mp4'
    config_path  = 'config_cargo_door.json'
    weights_path = '/home/ubuntu/tensorflow_data/YOLO/CargoDoor/full_yolo_cargo_door_with_pose_v5.h5'

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(architecture        = config['model']['architecture'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################    

    print(weights_path)
    yolo.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.startWindowThread() #to make sure we can close it later on


    if image_path[-4:] == '.mp4':
        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))

        for i in range(nb_frames):
            _, image = video_reader.read()
            
            boxes, rpy = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            print len(boxes), 'box(es) found'

            #display frame
            image_np = np.uint8(image)
            cv2.imshow("Detection", image_np)
            k = cv2.waitKey(3)# & 0xEFFFFF
            # if k == 27:
            #     print("You Pressed Escape")
            #     break

        video_reader.release()
    else:
        image = cv2.imread(image_path)


        #Predict and time it
        t0 = time.time()
        boxes, rpy = yolo.predict(image)    # gets the predicted bounding boxes and the rpy distances
        t1 = time.time()
        total = t1-t0

        #overlay boxes
        image = draw_boxes(image, boxes, config['model']['labels'])

        #feedback
        print len(boxes), 'box(es) found'
        print 'Prediciton took %f seconds'%(total)

        print(rpy)
        
        #display frame
        cv2.imshow("Detection", image)
        k = cv2.waitKey(0) & 0xEFFFFF
        if k == 27:
            print("You Pressed Escape")

    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)
        
if __name__ == '__main__':
    print("RUNNING MODIFIED YOLO WITH BOUNDING BOXES + X,Y,Z DISTANCES FROM CAMERA FRAME TO THE DOOR FRAME")
    _main_()
