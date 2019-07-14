import numpy as np
import cv2
import numbers
import torch

from constants import CROP_SIZE
from constants import BBOX_SCALE
from constants import CROP_PAD
from constants import AREA_CUTOFF

LIMIT = 99999999

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

# Copied directly from re3 the below functions
# BBoxes are [x1, y1, x2, y2]
def clip_bbox(bboxes, minClip, maxXClip, maxYClip):
    bboxesOut = bboxes
    addedAxis = False
    if len(bboxesOut.shape) == 1:
        addedAxis = True
        bboxesOut = bboxesOut[:,np.newaxis]
    bboxesOut[[0,2],...] = np.clip(bboxesOut[[0,2],...], minClip, maxXClip)
    bboxesOut[[1,3],...] = np.clip(bboxesOut[[1,3],...], minClip, maxYClip)
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    return bboxesOut

# @bboxes {np.array} 4xn array of boxes to be scaled
# @scalars{number or arraylike} scalars for width and height of boxes
# @in_place{bool} If false, creates new bboxes.
def scale_bbox(bboxes, scalars,
        clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT,
        round=False, in_place=False):
    addedAxis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes, dtype=np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    if isinstance(scalars, numbers.Number):
        scalars = np.full((2, bboxes.shape[1]), scalars, dtype=np.float32)
    if not isinstance(scalars, np.ndarray):
        scalars = np.array(scalars, dtype=np.float32)
    if len(scalars.shape) == 1:
        scalars = np.tile(scalars[:,np.newaxis], (1,bboxes.shape[1])).astype(np.float32)

    bboxes = bboxes.astype(np.float32)

    width = bboxes[2,...] - bboxes[0,...]
    height = bboxes[3,...] - bboxes[1,...]
    xMid = (bboxes[0,...] + bboxes[2,...]) / 2.0
    yMid = (bboxes[1,...] + bboxes[3,...]) / 2.0
    if not in_place:
        bboxesOut = bboxes.copy()
    else:
        bboxesOut = bboxes

    bboxesOut[0,...] = xMid - width * scalars[0,...] / 2.0
    bboxesOut[1,...] = yMid - height * scalars[1,...] / 2.0
    bboxesOut[2,...] = xMid + width * scalars[0,...] / 2.0
    bboxesOut[3,...] = yMid + height * scalars[1,...] / 2.0

    if clipMin != -LIMIT or clipWidth != LIMIT or clipHeight != LIMIT:
        bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut).astype(np.int32)
    return bboxesOut

def intersection(rect1, rect2):
    return (max(0, min(rect1[2], rect2[2]) - max(rect1[0], rect2[0])) *
        max(0, min(rect1[3], rect2[3]) - max(rect1[1], rect2[1])))

def fix_bbox_intersection(bbox, gtBox, imageWidth, imageHeight):
    if type(bbox) == list:
        bbox = np.array(bbox)
    if type(gtBox) == list:
        gtBox = np.array(gtBox)

    gtBoxArea = float((gtBox[3] - gtBox[1]) * (gtBox[2] - gtBox[0]))
    bboxLarge = scale_bbox(bbox, CROP_PAD)
    while intersection(bboxLarge, gtBox) / gtBoxArea < AREA_CUTOFF:
        bbox = bbox * .9 + gtBox * .1
        bboxLarge = scale_bbox(bbox, CROP_PAD)
    return bbox

# @inputImage{ndarray HxWx3} Full input image.
# @bbox{ndarray or list 4x1} bbox to be cropped in x1,y1,x2,y2 format.
# @padScale{number} scalar representing amount of padding around image.
#   padScale=1 will be exactly the bbox, padScale=2 will be 2x the input image.
# @outputSize{number} Size in pixels of output crop. Crop will be square and
#   warped.
# @return{tuple(patch, outputBox)} the output patch and bounding box
#   representing its coordinates.
def get_crop_input(inputImage, bbox, padScale, outputSize):
    bbox = np.array(bbox)
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    imShape = np.array(inputImage.shape)
    drawbbox = np.round(bbox).astype(int)
    # img = cv2.rectangle(inputImage, (drawbbox[0], drawbbox[1]), (drawbbox[2], drawbbox[3]), (0, 255, 0), 3)
    # cv2.imshow('Image',img)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    if len(imShape) < 3:
        inputImage = inputImage[:,:,np.newaxis]
    xC = float(bbox[0] + bbox[2]) / 2
    yC = float(bbox[1] + bbox[3]) / 2
    boxOn = np.zeros(4)
    boxOn[0] = float(xC - padScale * width / 2)
    boxOn[1] = float(yC - padScale * height / 2)
    boxOn[2] = float(xC + padScale * width / 2)
    boxOn[3] = float(yC + padScale * height / 2)
    outputBox = boxOn.copy()
    boxOn = np.round(boxOn).astype(int)
    boxOnWH = np.array([boxOn[2] - boxOn[0], boxOn[3] - boxOn[1]])
    imagePatch = inputImage[max(boxOn[1], 0):min(boxOn[3], imShape[0]),
            max(boxOn[0], 0):min(boxOn[2], imShape[1]), :]
    # cv2.imshow('ImagePatch',imagePatch)
    # cv2.waitKey(5000)
    # cv2.imshow('Original Image',inputImage)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    boundedBox = np.clip(boxOn, 0, imShape[[1,0,1,0]])
    boundedBoxWH = np.array([boundedBox[2] - boundedBox[0], boundedBox[3] - boundedBox[1]])

    if imagePatch.shape[0] == 0 or imagePatch.shape[1] == 0:
        patch = np.zeros((int(outputSize), int(outputSize), 3))
    else:
        patch = cv2.resize(imagePatch, (
            max(1, int(np.round(outputSize * boundedBoxWH[0] / boxOnWH[0]))),
            max(1, int(np.round(outputSize * boundedBoxWH[1] / boxOnWH[1])))))
        if len(patch.shape) < 3:
            patch = patch[:,:,np.newaxis]
        patchShape = np.array(patch.shape)

        pad = np.zeros(4, dtype=int)
        pad[:2] = np.maximum(0, -boxOn[:2] * outputSize / boxOnWH)
        pad[2:] = outputSize - (pad[:2] + patchShape[[1,0]])

        if np.any(pad != 0):
            if len(pad[pad < 0]) > 0:
                patch = np.zeros((int(outputSize), int(outputSize), 3))
            else:
                patch = np.lib.pad(
                        patch,
                        ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)),
                        'constant', constant_values=0)
    # print(outputBox)
    # print(inputImage.shape)
    # cv2.imshow('Patch',patch)
    # cv2.waitKey(5000)
    # cv2.destroyAllWindows()
    # patch = torch.from_numpy(patch)
    return patch, outputBox

# [x1 y1, x2, y2] to [xMid, yMid, width, height]
def xyxy_to_xywh(bboxes, clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT,
        round=False):
    addedAxis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    bboxesOut = np.zeros(bboxes.shape)
    x1 = bboxes[0,...]
    y1 = bboxes[1,...]
    x2 = bboxes[2,...]
    y2 = bboxes[3,...]
    bboxesOut[0,...] = (x1 + x2) / 2.0
    bboxesOut[1,...] = (y1 + y2) / 2.0
    bboxesOut[2,...] = x2 - x1
    bboxesOut[3,...] = y2 - y1
    if clipMin != -LIMIT or clipWidth != LIMIT or clipHeight != LIMIT:
        bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if bboxesOut.shape[0] > 4:
        bboxesOut[4:,...] = bboxes[4:,...]
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut).astype(int)
    # bboxesOut = torch.from_numpy(bboxesOut)
    return bboxesOut

# [xMid, yMid, width, height] to [x1 y1, x2, y2]
def xywh_to_xyxy(bboxes, clipMin=-LIMIT, clipWidth=LIMIT, clipHeight=LIMIT,
        round=False):
    addedAxis = False
    if isinstance(bboxes, list):
        bboxes = np.array(bboxes).astype(np.float32)
    if len(bboxes.shape) == 1:
        addedAxis = True
        bboxes = bboxes[:,np.newaxis]
    bboxesOut = np.zeros(bboxes.shape)
    xMid = bboxes[0,...]
    yMid = bboxes[1,...]
    width = bboxes[2,...]
    height = bboxes[3,...]
    bboxesOut[0,...] = xMid - width / 2.0
    bboxesOut[1,...] = yMid - height / 2.0
    bboxesOut[2,...] = xMid + width / 2.0
    bboxesOut[3,...] = yMid + height / 2.0
    if clipMin != -LIMIT or clipWidth != LIMIT or clipHeight != LIMIT:
        bboxesOut = clip_bbox(bboxesOut, clipMin, clipWidth, clipHeight)
    if bboxesOut.shape[0] > 4:
        bboxesOut[4:,...] = bboxes[4:,...]
    if addedAxis:
        bboxesOut = bboxesOut[:,0]
    if round:
        bboxesOut = np.round(bboxesOut).astype(int)
    # bboxesOut = torch.from_numpy(bboxesOut)
    return bboxesOut


# Converts from the full image coordinate system to range 0:crop_padding. Useful for getting the coordinates
#   of a bounding box from image coordinates to the location within the cropped image.
# @bbox_to_change xyxy bbox whose coordinates will be converted to the new reference frame
# @crop_location xyxy box of the new origin and max points (without padding)
# @crop_padding the amount to pad the crop_location box (1 would be keep it the same, 2 would be doubled)
# @crop_size the maximum size of the coordinate frame of bbox_to_change.
def to_crop_coordinate_system(bbox_to_change, crop_location, crop_padding, crop_size):
    if isinstance(bbox_to_change, list):
        bbox_to_change = np.array(bbox_to_change)
    if isinstance(crop_location, list):
        crop_location = np.array(crop_location)
    bbox_to_change = bbox_to_change.astype(np.float32)
    crop_location = crop_location.astype(np.float32)

    crop_location = scale_bbox(crop_location, crop_padding)
    crop_location_xywh = xyxy_to_xywh(crop_location)
    bbox_to_change -= crop_location[[0,1,0,1]]
    bbox_to_change *= crop_size / crop_location_xywh[[2,3,2,3]]
    return bbox_to_change

# if __name__ == '__main__':
#
#     path = 'C:/Users/Janani/Desktop/Computer Vision/Project/final/data3/ILSVRC2015/Data/train/ILSVRC2015_train_00033007/'
#     img = cv2.imread(path + '000000.JPEG')
#     xmin = 50
#     ymin = 120
#     xmax = 448
#     ymax = 261
#     bbox = np.array([xmin, ymin, xmax, ymax])
#


