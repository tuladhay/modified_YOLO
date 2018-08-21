#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import time

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0" #"0" use cpu

#Simple function to predict bounding boxes using YOLO

def _main_():
 
    #where are weights and video are
    weights_path = '/media/ubuntu/hdd/tensorflow_data/YOLO/PeopleData/yolo_people_final.h5'
    image_path   = '/media/ubuntu/hdd/tensorflow_data/YOLO/PeopleData/train_images/people_1776.jpeg'

    #Configuration
    architecture = "Full Yolo"
    input_size = 416
    anchors = [0.58,0.97, 1.09,1.84, 1.79,5.82, 2.69,9.29, 5.14,10.71]
    max_box_per_image = 10
    labels = ["person"]

    ###############################
    #   Make the model 
    ###############################

    model = YOLO(architecture        = architecture,
                input_size          = input_size, 
                labels              = labels, 
                max_box_per_image   = max_box_per_image,
                anchors             = anchors)

    ###############################
    #   Load trained weights
    ###############################    

    print("loading weights from")
    print(weights_path)
    model.load_weights(weights_path)

    ###############################
    #   Predict bounding boxes 
    ###############################

    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.startWindowThread() #to make sure we can close it later on

    #load image
    image = cv2.imread(image_path)

    #Predict and time it
    t0 = time.time()
    boxes = model.predict(image)
    t1 = time.time()
    total = t1-t0

    #overlay boxes
    image = draw_boxes(image, boxes, labels)

    #feedback
    print len(boxes), 'box(es) found'
    print 'Prediciton took %f seconds'%(total)      
    
    #display frame
    cv2.imshow("Detection", image)
    k = cv2.waitKey(0) & 0xEFFFFF
    if k == 27:
        print("You Pressed Escape")

    cv2.destroyAllWindows()
    for i in range (1,5):
        cv2.waitKey(1)
        
if __name__ == '__main__':
    _main_()
