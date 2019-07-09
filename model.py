import os
import numpy as np
import torch.nn as nn
import torch
import torchvision.models as models

import constants

class CNN(nn.Module):
    def __init__(self,batch_size,unroll):
        super(ConvNet,self).__init__()
        self.batch_size = batch_size
        self.unroll = unroll
        alexnet = models.alexnet(pretrained=True)

        # Delete last FC layers
        modules = list(alexnet.children())[0]

        # Splitting model at skip connection points
        self.conv1 = nn.Sequential(*modules[:3])
        self.conv2 = nn.Sequential(*modules[3:6])
        self.conv3 = nn.Sequential(*modules[6:-1])
        self.last_maxpool = modules[-1]

        # Prelu activations for skip connections
        self.prelu1 = nn.PReLU(16)
        self.prelu2 = nn.PReLU(32)
        self.prelu3 = nn.PReLU(64)

        # Convolution layers for skip connections
        self.skip_conv1 = nn.Conv2d(64,16,1)
        self.skip_conv2 = nn.Conv2d(192,32,1)
        self.skip_conv3 = nn.Conv2d(256,64,1)

    def forward(self,images):
        # Extract image features using Conv net
        with torch.no_grad():
            conv1_output = self.conv1(images)
            conv2_output = self.conv2(conv1_output)
            conv3_output = self.conv3(conv2_output)
            alexnet_features = self.last_maxpool(conv3_output)
        # Convolution for skip connections with prelu activations
        skip_conv1_output = self.prelu1(self.skip_conv1(conv1_output))
        skip_conv2_output = self.prelu2(self.skip_conv2(conv2_output))
        skip_conv3_output = self.prelu3(self.skip_conv3(conv3_output))

        # Flatten skip connections
        skip_conv1_output = skip_conv1_output.reshape(skip_conv1_output.size(0),-1)
        skip_conv2_output = skip_conv2_output.reshape(skip_conv2_output.size(0),-1)
        skip_conv3_output = skip_conv3_output.reshape(skip_conv3_output.size(0),-1)
        alexnet_features = alexnet_features.reshape(alexnet_features.size(0),-1)

        # Concatentate all skip connections
        convnet_output = torch.cat((skip_conv1_output,skip_conv2_output,skip_conv3_output,alexnet_features),1)

        # (batch_size x num_unroll x 2) x feature_size --> (batch_size x num_unroll) x (2 x feature_size)
        convnet_output = convnet_output.reshape(int(convnet_output.size(0)/2),-1)
        return convnet_output


class RNN(nn.Module):
    def __init__(self,feature_size,num_unroll, batch_size, use_state):
        super(RNN,self).__init__()
        self.unroll = num_unroll
        self.batch_size = batch_size
        self.use_state = use_state
        self.num_layers = 1
        self.num_directions = 1
        #TODO: Remove this hardcoding od cuda
        self.h1 = torch.zeros(self.num_layers*self.num_directions, self.batch_size, LSTM_SIZE).cuda()
        self.c1 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, LSTM_SIZE).cuda()
        self.h2 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, LSTM_SIZE).cuda()
        self.c2 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, LSTM_SIZE).cuda()
        self.fc1 = nn.Linear(feature_size,1024)
        self.lstm1 = nn.LSTM(1024,LSTM_SIZE,1,batch_first=True)
        self.lstm2 = nn.LSTM(1024+LSTM_SIZE,LSTM_SIZE,1,batch_first=True)
        self.fc_last = nn.Linear(LSTM_SIZE*self.unroll,4)

    def forward(self,features):
        fc1_output = self.fc1(features)
        fc1_output_reshape = fc1_output.reshape(int(fc1_output.size(0)/self.unroll),self.unroll,-1)
        lstm1_output,state1 = self.lstm1(fc1_output_reshape, (self.h1, self.c1))
        # Concatenate lstm1 output and fc layer output
        lstm2_input = torch.cat((fc1_output_reshape,lstm1_output),2)
        lstm2_output,state2 = self.lstm2(lstm2_input, (self.h2, self.c2))
        # Flatten LSTM output
        fc_last_input = lstm2_output.reshape(lstm2_output.size(0),-1)
        fc_output = self.fc_last(fc_last_input)
        if(self.use_state):
            self.h1 = state1[0]
            self.c1 = state1[1]
            self.h2 = state2[0]
            self.c2 = state2[1]
        else:
            self.h1 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, LSTM_SIZE).cuda()
            self.c1 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, LSTM_SIZE).cuda()
            self.h2 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, LSTM_SIZE).cuda()
            self.c2 = torch.zeros(self.num_layers * self.num_directions, self.batch_size, LSTM_SIZE).cuda()

        return fc_output
