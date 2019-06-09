import numpy as np
import cv2

from constants import CROP_SIZE
from constants import BBOX_SCALE
from constants import DEBUG


def scale_bbox(bbox):
    bbox = bbox.astype(float)
    x_mid = (bbox[0] + bbox[2])/2.0
    y_mid = (bbox[1] + bbox[3])/2.0
    width = int(bbox[2] - bbox[0])
    height = int(bbox[3] - bbox[1])

    new_width = width * BBOX_SCALE
    new_height = height * BBOX_SCALE
    new_x1 = x_mid - (new_width/2.0)
    new_y1 = y_mid - (new_height/2.0)
    new_x2 = x_mid + (new_width/2.0)
    new_y2 = y_mid + (new_height/2.0)
    new_bbox = np.array([new_x1, new_y1, new_x2, new_y2]).astype(int)
    return new_bbox



def image_crop(image, bbox):
    img_shape = np.array(image.shape)
    bbox = scale_bbox(bbox)
    crop_location = bbox.copy()
    boundedW = bbox[2] - bbox[0]
    boundedH = bbox[3] - bbox[1]

    # After clipping
    bbox_clip = np.clip(bbox, 0, img_shape[[1, 0, 1, 0]])
    crop_image = image[bbox_clip[1]: bbox_clip[3], bbox_clip[0]:bbox_clip[2], :]
    # TODO: Remove not
    if DEBUG:
        cv2.imshow('Original Image', image)
        cv2.waitKey(5000)
        cv2.imshow('CroppedImage', crop_image)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()
    boundedW_clip = bbox_clip[2] - bbox_clip[0]
    boundedH_clip = bbox_clip[3] - bbox_clip[1]

    left_pad = np.maximum(-bbox[0], 0)
    top_pad = np.maximum(-bbox[1], 0)
    right_pad = np.maximum(bbox[2] - img_shape[1], 0)
    bottom_pad = np.maximum(bbox[3] - img_shape[0], 0)
    patch_img = crop_image.copy()
    patch_img_shape = patch_img.shape
    patch_img_new = np.pad(patch_img,
                           ((top_pad, bottom_pad), (left_pad, right_pad), (0, 0)),
                           'constant', constant_values=((0, 0), (0, 0), (0, 0)))

    # patch = np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
    # patch_img_width = int(CROP_SIZE * boundedW_clip / boundedW)
    # patch_img_height = int(CROP_SIZE * boundedH_clip / boundedH)
    #
    # resized_img = cv2.resize(crop_image, (patch_img_height, patch_img_width))

    # patch[patch_y1:patch_y2, patch_x1:patch_x2] = resized_img
    if DEBUG:
        cv2.imshow('Patch with padding', patch_img_new)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    patch = cv2.resize(patch_img_new, (CROP_SIZE, CROP_SIZE))
    if DEBUG:
        cv2.imshow('Patch resized with padding', patch)
        cv2.waitKey(5000)
        cv2.destroyAllWindows()

    return patch, crop_location


def find_crop_label(bbox, crop_location):
    crop_width = crop_location[2] - crop_location[0]
    crop_height = crop_location[3] - crop_location[1]
    output_box = (bbox - crop_location[[0, 1, 0, 1]]).astype(float)
    # Making labels scale invariant
    output_box[0] /= crop_width
    output_box[1] /= crop_height
    output_box[2] /= crop_width
    output_box[3] /= crop_height
    return output_box



if __name__ == '__main__':

    path = 'C:/Users/Janani/Desktop/Computer Vision/Project/final/data3/ILSVRC2015/Data/train/ILSVRC2015_train_00033007/'
    img = cv2.imread(path + '000000.JPEG')
    xmin = 50
    ymin = 120
    xmax = 448
    ymax = 261
    bbox = np.array([xmin, ymin, xmax, ymax])
    patch = image_crop(img, bbox)



