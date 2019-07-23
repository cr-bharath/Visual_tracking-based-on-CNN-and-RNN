import argparse
import numpy as np
import os
import tracker
from data_generator import TrackerDataset
from constants import CROP_SIZE
from constants import CROP_PAD
from constants import IMAGENET_MEAN_BGR
from constants import IMAGENET_STD_DEV_BGR

DATA_PATH = "../../"

class Test(TrackerDataset):
    def __init__(self, data_path, annot_path, list_id, folder_start_pos):
        self.list_id = list_id
        self.folder_start_pos = folder_start_pos
        self.data_path = data_path
        self.annot_path = annot_path
        self.folder = [dI for dI in os.listdir(self.data_path) if
                       os.path.isdir(os.path.join(self.data_path, dI))]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tracker_ = re3Tracker(self,device)
    def run_test(self):
        folder_idx = 0
        file_idx = 0
        initial_frame = True
        for item in range(len(self.list_id)):
            if item < self.folder_start_pos[folder_idx]:
                # Image belongs to same folder

                folder_name = self.folder[folder_idx]

                if folder_idx == 0:
                    file_index = item
                else:
                    file_index = item - self.folder_start_pos[folder_idx - 1]

                image, label = self.getData(folder_name, file_index)

                if(initial_frame):
                    bbox = self.tracker.track(image,label)
                    initial_frame = False
                else:
                    bbox = self.tracker.track(image,None)
                cv2.rectangle(image,(int(bbox[0]), int(bbox[1])),(int(bbox[2]), int(bbox[3])),
                	(0,255,0),2)
                cv2.imshow('Test Image',image)
                cv2.waitKey(1)

            else:
                # Start using a new folder of images
                folder_idx += 1
                item -= 1 # Repeat with same iterator
            # TODO: Remove
            break

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
    data_path = FLAGS.data_path
    unrolling_factor = 1 # For testing
    # val_data_path = "C:/Users/Janani/Desktop/Computer Vision/Project/final/data/val/Data/"
    DATA_PATH = os.getcwd()
    val_data_path = "../.." + "/data/val/Data/"
    val_annot_path = "../.." + "/data/val/Annotations/"
    print("Val path %s"%val_data_path)
    list_id, folder_start_pos = prepare_for_dataset(val_data_path, unrolling_factor)
    obj = Test(val_data_path, val_annot_path, list_id, folder_start_pos)
    obj.run_test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p','--data_path', type=str, default="data/", help='path were Val data is stored')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    FLAGS = parser.parse_args()
    main(FLAGS)