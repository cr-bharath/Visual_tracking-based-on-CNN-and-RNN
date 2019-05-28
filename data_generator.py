import torch
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
import re

from torch.utils.data.dataset import Dataset

DEBUG = False
Unrolling_factor = 2


class TrackerDataset(Dataset):
    def __init__(self, train_data_path, train_annot_path, list_id, folder_start_pos, dim):
        self.train_data_path = train_data_path
        self.train_annot_path = train_annot_path
        self.list_id = list_id
        self.folder_start_pos = folder_start_pos
        self.dim = dim
        self.folder = [dI for dI in os.listdir(self.train_data_path) if
                       os.path.isdir(os.path.join(self.train_data_path, dI))]
        self.transforms = transforms.ToTensor()

    def __len__(self):
        return len(self.list_id)

    def __getitem__(self, item):
        # Returns one sample of data and its label
        # One sample for tracking means image at time t, t+1, t+2...t+(Unrolling_factor-1)
        # However, the label is for the last image, t+(Unrolling_factor-1)

        # Find folder index and file index of the given ID
        folder_index = 0
        file_index = 0
        x = np.empty((Unrolling_factor, self.dim[2], self.dim[0], self.dim[1]))# Since x is a tensor and #channels
                                                                               # appear first
        temp = x.shape
        y = np.empty((1, 4))

        for i in range(len(self.folder_start_pos)):
            if item < (self.folder_start_pos[i]):
                folder_index = i
                if DEBUG:
                    print("Given list id is %d" % item)
                    print("Corresponding folder index %d" % folder_index)
                break
        if folder_index == 0:
            file_index = item
        else:
            file_index = item - self.folder_start_pos[folder_index - 1]
        # Finding X
        #TODO: Crop properly for X and apply appropritae Y
        for train_len in range(Unrolling_factor):
            # After finding in which folder the file lives, do imread of that file
            folder_name = self.folder[folder_index]
            file_name = "{:06d}".format(file_index + train_len)
            img_path = self.train_data_path + folder_name + "/" + file_name + ".JPEG"
            img = cv2.imread(img_path)
            ws = img.shape[1] / self.dim[1]
            hs = img.shape[0] / self.dim[0]
            w = int(img.shape[1] / ws)
            h = int(img.shape[0] / hs)
            dim = (w, h)
            img = cv2.resize(img, dim)
            img = img / 255.0
            temp = img.shape
            img = self.transforms(img)
            temp = img.shape
            x[train_len, :, :, :] = img
        # Taking the label of last image
        bbox = self.get_label(folder_name, file_name)
        bbox[0] /= hs
        bbox[2] /= hs
        bbox[1] /= ws
        bbox[3] /= ws
        y = np.int32(bbox)
        return x, y

    def get_label(self, folder_name, file_name):
        # base_path = os.getcwd()
        xml_file = self.train_annot_path+folder_name+"/"+file_name+".xml"
        file = open(xml_file)
        contents = file.read()
        a = re.search('<xmax>([\d]+)<\/xmax>',contents)
        if(a == None):
            xmax = 20
        else:
            xmax = int(a.group(1))

        a = re.search(r'<xmin>([\d]+)<\/xmin>', contents)
        if (a == None):
            xmin = 0
        else:
            xmin = int(a.group(1))

        a = re.search(r'<ymax>([\d]+)<\/ymax>', contents)
        if (a == None):
            ymax = 20
        else:
            ymax = int(a.group(1))

        a = re.search(r'<ymin>([\d]+)<\/ymin>',contents)
        if (a == None):
            ymin = 0
        else:
            ymin = int(a.group(1))

        return [xmin, ymin, xmax, ymax]


    #TODO: To be removed. Just to check the working of __getitem__
    def print(self):
        self.__getitem__(1)