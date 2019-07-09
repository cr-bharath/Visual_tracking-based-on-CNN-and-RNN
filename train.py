import torch
import torch.nn as nn
import numpy as np
import os
from model import ConvNet, RNN
import constants
import argparse

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Update your Datapath here
DATA_PATH = "../../"

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
    # Define Models
    CNN_model = ConvNet(FLAGS.batch_size,FLAGS.unroll).to(device)
    RNN_model = RNN(CNN_OUTPUT_SIZE,FLAGS.unroll).to(device)
    
   if debug:
        print("Data folder is present in %s" % DATA_PATH)
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
    criterion = nn.MSELoss()
    #Only skip connection parameters need to be learned
    skip_conv_params = list(CNN_model.skip_conv3.parameters()) + list(CNN_model.skip_conv2.parameters()) + list(cnn_model.skip_conv1.parameters())
    prelu_params = list(CNN_model.prelu1.parameters()) + list(CNN_model.prelu1lu2.parameters()) + list(cnn_model.prelu3.parameters())
    #Network parameters that needs to be learnt by training
    params = list(RNN_model.parameters()) + prelu_params + skip_conv_params
    optimizer = torch.optim.Adam(params,lr=FLAGS.learning_rate)

    total_step = len(train_loader)
    #Start training
    for epoch in range(FLAGS.num_epochs):
        for i,(images,gt_labels) in enumerate(train_loader):

            images = images.to(device)
            #Forward, backward and optimize
            CNN_features = CNN_model(images)
            pred_labels = RNN_model(CNN_features)
            loss = criterion(pred_labels,gt_labels)
            CNN_model.zero_grad()
            RNN_model.zero_grad()
            loss.backward()
            optmizer.step()

            # Print log info
            if i % 50 == 0:
                print('Epoch [{}/{}], Step [{},{}], Loss {:4f}'.format(epoch,FLAGS.num_epochs,i,total_step,loss.item()))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batch_size', type=int, default=64)
    parser.add_argument('-n','--unroll', type=int, default=2)
    parser.add_argument('-l','--learning_rate', type=float, default=0.00001)
 #   parser.add_argument('-p','--data_path', type=str, default="data/", help='path were data is stored')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-e', '--num_epochs', type=int, default=10)
    FLAGS = parser.parse_args()
    main(FLAGS)