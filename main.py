import os
import argparse
import numpy as np

from data_generator import TrackerDataset
from torch.utils.data import DataLoader
from constants import CROP_SIZE

def prepare_for_dataset(path, unrolling_factor):
    folder_start_pos = []
    total_images = 0
    # Recursively iterate through imagenet data folders to assign IDs for the eligible images of the  video sequence.
    # Last few images are decided as eligible or not eligible by Unrolling factor
    for _, dirnames, filenames in os.walk(path):
        if len(filenames) != 0:
            n_files = len(filenames) # Total number of images present in the video
            # Not all are eligible for list_ids
            total_images += (n_files - (unrolling_factor-1))
            folder_start_pos.append(total_images)

    list_id = np.arange(0, total_images)
    return list_id, folder_start_pos


def main(FLAGS):
    unrolling_factor = FLAGS.num_unrolls
    batch_size = FLAGS.batch_size
    debug = FLAGS.debug
    print("Unrolling factor %d" % unrolling_factor)
    print("Batch size = %d"% batch_size)
    print(" Debug =  %r" % debug)

    # Since my data for train and val are located two folders back
    os.chdir("../../")
    data_path = os.getcwd()
    if debug:
        print("Data folder is present in %s" % data_path)
    # Training data path
    train_data_path = data_path + '/data/train/Data/'
    train_annot_path = data_path + '/data/train/Annotations/'
    list_id, folder_start_pos = prepare_for_dataset(train_data_path, unrolling_factor)
    train_dataset = TrackerDataset(train_data_path, train_annot_path, list_id,
                                   folder_start_pos, (CROP_SIZE, CROP_SIZE, 3),
                                   unrolling_factor, debug)
    # img,labels = train_dataset.__getitem__(1)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    count = 0
    for images, labels in train_loader:
        images = images.view(-1, 3, CROP_SIZE, CROP_SIZE)
        print(images.shape)
        print(labels.shape)
        break
    # print(count)
    if debug:
        print("Folder start positions")
        print(folder_start_pos)
        print("Total images chosen for list_id %d" % list_id[-1])

    path = data_path + '/data/val/Data/'
    list_id, folder_start_pos = prepare_for_dataset(path, unrolling_factor)
    if debug:
        print("Folder start positions")
        print(folder_start_pos)
        print("Total images chosen for list_id %d" % list_id[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Input debug, unrolling factor')
    parser.add_argument('-n', '--num_unrolls', action='store', default=2, type=int)
    parser.add_argument('-b', '--batch_size', action='store', default=32, type=int)
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    FLAGS = parser.parse_args()
    main(FLAGS)
