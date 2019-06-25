import os
import numpy as np
import torch.nn as nn
import torch
import torchvision.models as models


class ConvNet(nn.Module):
    def __init__(self,batch_size,unroll):
        super(ConvNet,self).__init__()
        self.batch_size = batch_size
        self.unroll = unroll
        alexnet = models.alexnet(pretrained=True)
        # Delete last FC layers
        modules = list(alexnet.children())[0]
        modules = list(modules.children())[:-1]
        # Extracting skip layers
        self.conv1 = nn.Sequential(*modules[:2])
        self.conv2 = nn.Sequential(*modules[2:5])
        self.conv3 = nn.Sequential(*modules[5:])
    def forward(self,images):
        # Extract image features using Conv net
        with torch.no_grad():
            features1 = self.conv1(images)
            features2 = self.conv2(features1)
            features3 = self.conv3(features2)
        # Flatten all skip layers
        features1 = features1.reshape(features1.size(0),-1)
        features2 = features2.reshape(features2.size(0),-1)
        features3 = features3.reshape(features3.size(0),-1)
        features = torch.cat((features1,features2,features3),1)
        features = features.reshape(features.size(0)/2,-1)
        return features


class RNN(nn.Module):
    def __init__(self,feature_size,num_unroll):
        super(RNN,self).__init__()
        self.unroll = num_unroll
        self.fc1 = nn.Linear(feature_size,1024)
        self.lstm1 = nn.LSTM(1024,512,1,batch_first=True)
        self.lstm2 = nn.LSTM(1536,512,1,batch_first=True)
        self.fc_last = nn.Linear(512*self.unroll,4)
    def forward(self,features):
        fc1_output = self.fc1(features)
        fc1_output_reshape = fc1_output.reshape(fc1_output.size(0)/self.unroll,self.unroll,-1)
        lstm1_output,state = self.lstm1(fc1_output_reshape)
        lstm2_input = torch.cat((fc1_output,lstm1_output),2)
        lstm2_output,state = self.lstm2(lstm2_input)
        fc_last_input = lstm2_output.reshape(lstm2_output.size(0),-1)
        fc_output = self.fc2(fc_last_input)
        return fc_output
