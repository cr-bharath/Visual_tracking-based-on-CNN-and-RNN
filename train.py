import torch
import torch.nn as nn
import numpy as np
import os
from model import CNN, RNN
import constants
from constants import CNN_OUTPUT_SIZE
from data_generator import TrackerDataset
from torch.utils.data import DataLoader
from constants import CROP_SIZE
import argparse

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Update your Datapath here
DATA_PATH = "../"
#LOG file
flog = open("LOG.txt","a+")
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
    data_path = DATA_PATH
    debug = FLAGS.debug
    unrolling_factor = FLAGS.unroll
    batch_size = FLAGS.batch_size

    # Initialize Models
    CNN_model = CNN(FLAGS.batch_size,FLAGS.unroll).to(device)
    RNN_model = RNN(CNN_OUTPUT_SIZE,FLAGS.unroll, FLAGS.batch_size, False).to(device) # Stateless LSTM for training

    criterion = nn.MSELoss()
    # Only skip connection parameters need to be learned
    skip_conv_params = list(CNN_model.skip_conv3.parameters()) + list(CNN_model.skip_conv2.parameters()) + list(
        CNN_model.skip_conv1.parameters())
    prelu_params = list(CNN_model.prelu1.parameters()) + list(CNN_model.prelu2.parameters()) + list(
        CNN_model.prelu3.parameters())
    # Network parameters that needs to be learnt by training
    params = list(RNN_model.parameters()) + prelu_params + skip_conv_params

    # Initialize optimizer , added weight decay
    optimizer = torch.optim.Adam(params, lr=FLAGS.learning_rate,weight_decay=0.0005)

    # Load Checkpoint if present
    epoch = 0
    # TODO: Manually copy the checkpoint to this path and has to be renamed as below
    checkpoint_name = './final_checkpoint/re3_final_checkpoint.pth'
    if debug:
        print("Checkpoint name is %s" % checkpoint_name)
    if os.path.isfile(checkpoint_name):
        checkpoint = torch.load(checkpoint_name)
        CNN_model.load_state_dict(checkpoint['cnn_model_state_dict'])
        RNN_model.load_state_dict(checkpoint['rnn_model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        print("Checkpoint loaded")
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

    total_step = len(train_loader)
    #Start training
    ckpt_path = 'trained_checkpoints'
    # TODO: The checkpoints are saved in one folder and loaded from another folder.
    # TODO : By this way there is freedom to see model`s performance across different checkpoints.
    for epoch in range(epoch, FLAGS.num_epochs):
        for minibatch ,(images,gt_labels) in enumerate(train_loader):
            try:
                images = images.view(-1, 3, CROP_SIZE, CROP_SIZE).float()
                images = images.to(device)
                gt_labels = gt_labels.view(-1,4).float()
                gt_labels = gt_labels.to(device)
            #Forward, backward and optimize
                CNN_features = CNN_model(images)
                pred_labels = RNN_model(CNN_features)
                loss = criterion(pred_labels,gt_labels)
                CNN_model.zero_grad()
                RNN_model.zero_grad()
                loss.backward()
                optimizer.step()

            # TODO: Remove the below lines. Written to check if one minibatch is proper
            #print("Done one mini batch")
            #break

            # Print log info
            if minibatch % 50 == 0:
                # TODO: Revert change
                print('Epoch [{}/{}], Step [{},{}], Loss {:4f}'.format(epoch,FLAGS.num_epochs,minibatch,total_step,loss.item()))

        if (epoch+1) % 10 == 0:
                if minibatch % 20 == 0:
                    # TODO: Revert change
                    print('Epoch [{}/{}], Step [{},{}], Loss {:4f}'.format(epoch,FLAGS.num_epochs,minibatch,total_step,loss.item()))
                    flog.write('Epoch [{}/{}], Step [{},{}], Loss {:4f}\n'.format(epoch,FLAGS.num_epochs,minibatch,total_step,loss.item()))
            except:
                print(images.size())
        #if (epoch) % 5 == 0:
        if (epoch) % 10 == 0:
            # Save the model checkpoint
            torch.save({
                        'epoch': epoch,
                        'cnn_model_state_dict': CNN_model.state_dict(),
                        'rnn_model_state_dict': RNN_model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        }, ckpt_path+'/checkpoint_'+str(epoch)+'.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-b','--batch_size', type=int, default=64)
    parser.add_argument('-n','--unroll', type=int, default=4)
    parser.add_argument('-l','--learning_rate', type=float, default=0.00001)
    parser.add_argument('-p','--data_path', type=str, default="data/", help='path were data is stored')
    parser.add_argument('-d', '--debug', action='store_true', default=False)
    parser.add_argument('-e', '--num_epochs', type=int, default=51)
    FLAGS = parser.parse_args()
    main(FLAGS)
