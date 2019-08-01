import torch
import torchvision.transforms as transforms
import os
import cv2
import numpy as np
import re
import im_util
from torch.utils.data.dataset import Dataset
# from matplotlib import pyplot as plt

from constants import CROP_SIZE
from constants import CROP_PAD
from constants import IMAGENET_MEAN_BGR
from constants import IMAGENET_STD_DEV_BGR

def data_preparation(image):
        # Data format is uint8
        image = image.astype(np.float32)
        if image.max() > 1.0:
            image /= 255.0
        image = (image - IMAGENET_MEAN_BGR)/IMAGENET_STD_DEV_BGR
        # To make pixel intensity between [0,1] rather than [-1,1]
        #image = np.clip(image, -1, 1)
        # image = (image + 1.0)/2.0
        # print("Image Mean = %.6f, Image Std Dev = %.6f" %(image.mean(), image.std()))
        # print("Image Min = %.6f, Image Max = %.6f"%(image.min(), image.max()))

        # To make channels first , for pytorch
        image = np.moveaxis(image, -1, 0)
        return image

class TrackerDataset(Dataset):
    def __init__(self, train_data_path, train_annot_path, list_id, folder_start_pos, dim, unrolls, debug):
        self.data_path = train_data_path
        self.annot_path = train_annot_path
        self.list_id = list_id
        self.folder_start_pos = folder_start_pos
        self.dim = dim
        self.unrolling_factor = unrolls
        self.debug = debug
        self.folder = [dI for dI in os.listdir(self.data_path) if
                       os.path.isdir(os.path.join(self.data_path, dI))]
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
        # x = np.empty((unrolling_factor, 2, self.dim[2], self.dim[0], self.dim[1]))# Since x is a tensor and #channels
        #                                                                        # appear first
        # temp = x.shape
        # y = np.empty((Unrolling_factor, 1, 4))
        tImage = np.zeros((self.unrolling_factor, 2, 3, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        xywhLabels = np.zeros((self.unrolling_factor, 4), dtype=np.uint8)

        for i in range(len(self.folder_start_pos)):
            if item < (self.folder_start_pos[i]):
                folder_index = i
                if self.debug:
                    print("Given list id is %d" % item)
                    print("Corresponding folder index %d" % folder_index)
                break
        if folder_index == 0:
            file_index = item
        else:
            file_index = item - self.folder_start_pos[folder_index - 1]
        # Beginning of one sequence . Sequence length = Unrolling factor
        folder_name = self.folder[folder_index]
        images, labels = self.getData(folder_name, file_index)
        height = images[1].shape[0]
        initbox = labels[0]
        bboxPrev = initbox

        for dd in range(self.unrolling_factor):
            bboxOn = labels[dd]
            if dd==0:
                noisyBox = bboxOn.copy()
            else:
                noisyBox = im_util.fix_bbox_intersection(bboxPrev, bboxOn, images[0].shape[1], images[0].shape[0])

            image_0, output_box0 = im_util.get_crop_input(
                images[max(dd - 1, 0)], bboxPrev, CROP_PAD, CROP_SIZE)

            tImage[dd, 0, ...] = data_preparation(image_0)

            image_1, output_box1 = im_util.get_crop_input(
                images[dd], noisyBox, CROP_PAD, CROP_SIZE)
            tImage[dd, 1, ...] = data_preparation(image_1)

            # if self.debug:
                # plt.subplot(121)
                # plt.imshow(image_0, cmap=plt.get_cmap('gray'))
                # plt.subplot(122)
                # plt.imshow(image_1, cmap=plt.get_cmap('gray'))
                # plt.show()
            # bboxPrev = bboxOn


            # Finding Labels
            shiftedBBox = im_util.to_crop_coordinate_system(bboxOn, noisyBox, CROP_PAD, 1)
            if self.debug:
                bbox_t = im_util.to_crop_coordinate_system(bboxPrev, noisyBox, CROP_PAD, 1)
                bbox_t = np.round(bbox_t*CROP_SIZE).astype(int)
                img = cv2.rectangle(image_0, (bbox_t[0], bbox_t[1]), (bbox_t[2], bbox_t[3]), (0, 255, 0), 2)
                bbox_t_1 = np.round(shiftedBBox.copy()*CROP_SIZE).astype(int)
                img = cv2.rectangle(img, (bbox_t_1[0], bbox_t_1[1]), (bbox_t_1[2], bbox_t_1[3]), (0, 0, 255), 2)
                cv2.imshow('Image at t', img)
                cv2.waitKey(5000)
                cv2.destroyAllWindows()
            shiftedBBoxXYWH = im_util.xyxy_to_xywh(shiftedBBox)
            xywhLabels[dd, :] = shiftedBBoxXYWH
            bboxPrev = bboxOn

        tImage = tImage.reshape([self.unrolling_factor* 2] + list(tImage.shape[2:]))
        xyxyLabels = im_util.xywh_to_xyxy(xywhLabels.T).T * 10
        xyxyLabels = xyxyLabels.astype(np.float32)

        return tImage, xyxyLabels

    def getData(self, folder_name, file_index):
        print(folder_name, file_index)
        images = [None]*self.unrolling_factor
        labels = [None]*self.unrolling_factor
        for dd in range(self.unrolling_factor):
            image_name = "{:06d}".format(file_index + dd)
            img_path = self.data_path + folder_name + "/" + image_name + ".JPEG"
            img = cv2.imread(img_path)
            images[dd] = img
            label = self.get_label(folder_name, image_name)
            labels[dd] = label
        return images, labels



    def get_patch_and_label(self, folder_name, image_name, bbox):
        img_path = self.data_path + folder_name + "/" + image_name + ".JPEG"
        img = cv2.imread(img_path)
        patch, crop_box = im_util.image_crop(img, bbox)
        label = im_util.find_crop_label(bbox, crop_box)
        return patch, label

    def get_label(self, folder_name, file_name):
        # base_path = os.getcwd()
        xml_file = self.annot_path+folder_name+"/"+file_name+".xml"
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
