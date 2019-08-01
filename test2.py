import numpy as np
import os
import sys
sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import glob
from tracker import re3Tracker
from data_generator import TrackerDataset

DATA_PATH = "/home/bcr/Summer19/VisualTracker/ILSVR2015/Data/VID/train/ILSVRC2015_VID_train_0001/ILSVRC2015_train_00033007"
DATA_PATH = "/home/bcr/Summer19/VisualTracker/ILSVR2015/Data/VID/train/ILSVRC2015_VID_train_0001/ILSVRC2015_train_00321002"
#DATA_PATH = "/home/bcr/Summer19/VisualTracker/ILSVR2015/Data/VID/val/ILSVRC2015_val_00000004/ILSVRC2015_val_00000003"

def main():
    tracker =re3Tracker()
    print("Tracker Initialized!!")
    image_paths = sorted(glob.glob(DATA_PATH + '/*.JPEG'))
    #initial_bbox = [309, 58, 604,469]
    # Annotation for first training video frame
    #initial_bbox = [148, 120, 448,261]
    # ILSVRC2015_train_00321002
    initial_bbox = [192, 176, 269, 215]
    image = cv2.imread(image_paths[0])
    imageRGB = image[:,:,::-1]
    bbox = tracker.track(imageRGB,initial_bbox)
    print(bbox)
    for ii,image_path in enumerate(image_paths):
        image = cv2.imread(image_path)
        imageRGB = image[:, :, ::-1]
        bbox = tracker.track(imageRGB)
        #print(bbox)
        cv2.rectangle(image,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                (0,255,0), 2)
        cv2.imshow('Image',image)
        cv2.waitKey(2)
if __name__ == "__main__":
    main()