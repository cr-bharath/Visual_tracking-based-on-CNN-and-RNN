from model import CNN,RNN
import os
import sys
from data_generator import data_preparation
from constants import CROP_PAD,CROP_SIZE
import torch 
import cv2
import im_util
import numpy as np

class re3Tracker():
	def __init__(self,device,checkpoint_name ='./final_checkpoint/re3_final_checkpoint.pth') :
		self.device = device
		self.CNN = CNN(1,1).to(self.device)
		self.RNN = RNN(1,1,True).to(self.device)
		if os.path.isfile(checkpoint_name):
        	checkpoint = torch.load(checkpoint_name)
        	self.CNN.load_state_dict(checkpoint['cnn_model_state_dict'])
        	self.RNN.load_state_dict(checkpoint['rnn_model_state_dict'])
		else:
			print("Invalid/No Checkpoint. Aborting...!!")
			sys.exit()
		self.forward_count = -1
		self.previous_frame = None
		self.cropped_input = np.zeros((2, 3, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
        
	def track(self,image,starting_box):
		if type(image) == str:
			image = cv2.imread(image)
		if starting_box is not None:
			prev_image = image
			past_box = starting_box
			self.forward_count = 0
		else:
			prev_image, past_box = self.previous_frame

		image_0, output_box0 = im_util.get_crop_input(previous_image, bbox_prev, CROP_PAD, CROP_SIZE)
        self.cropped_input[0, 0, ...] = data_preparation(image_0)

        image_1, = im_util.get_crop_input(image, bbox_prev, CROP_PAD, CROP_SIZE)
        self.cropped_input[0, 1, ...] = data_preparation(image_1)

        cropped_input_tensor = torch.from_numpy(self.cropped_input)
        features = self.CNN(cropped_input_tensor.to(self.device))
        predicted_bbox = self.RNN(features)
        
        # Save initial LSTM states
        if self.forward_count == 0:
        	self.RNN.lstm_state_init()

        output_bbox = im_util.from_crop_coordinate_system(predicted_bbox.squeeze() / 10.0, output_box0,1,1)

        # Reset LSTM states to initial state once #MAX_TRACK_LENGTH frames are processed and perform one forward pass
        if self.forward_count > 0 and self.forward_count % MAX_TRACK_LENGTH == 0:
        	cropped_input = im_util.get_crop_input(image,output_bbox,CROP_PAD,CROP_SIZE)
        	input_image = np.tile(cropped_input[np.newaxis,...],(2,1,1,1))
        	self.RNN.reset()
        	features = self.CNN(input_image)
        	prediction = self.RNN(features)
        if starting_box is not None:
        	output_bbox = starting_box

        self.previous_frame = (image, output_bbox)
        return output_bbox


