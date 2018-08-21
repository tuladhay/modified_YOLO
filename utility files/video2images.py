#! /usr/bin/env python

import argparse
import os
import cv2
import numpy as np
import sys

#Simple function toconvert video to image frames


def _main_():
 
    #define paths e.g.
    # /media/ubuntu/hdd/tensorflow_data/YOLO/CowData/train_images/cow-2.jpg
    video_path = '/home/ubuntu/python_ws/davis_ai/airplane_video/output_append.avi'
    image_folder   = '/home/ubuntu/python_ws/davis_ai/airplane_images_append/'
    image_root = 'cargo_' #whats this?

    index_offset = 0 #541 whats this?

    #desired image size
    WIDTH = 640
    HEIGHT = 480

    #open up video
    video_reader = cv2.VideoCapture(video_path)

    #determine number of frames
    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))

    count = 0

    print('Starting Frame Conversion')
    for _ in range(0, nb_frames, 4):
        #grab current frame
        _, image = video_reader.read()

        image=cv2.resize(image, (WIDTH, HEIGHT)) 
        
        #make filename
        filetemp = image_folder + image_root + str(count) + '.jpeg'

        #write file
        cv2.imwrite(filetemp, image)

        #percent complete
        #percent = (float(i) / float(nb_frames)) * 100.0
        #print "%.0f %% Complete"%percent
        count += 1
        
    
    #release
    video_reader.release()

    #feedback
    print 'Processed %i frames'%(nb_frames)
        
        
if __name__ == '__main__':
    _main_()
