from model import CNN,RNN
import os
import sys
from data_generator import data_preparation
from constants import CROP_PAD,CROP_SIZE,CNN_OUTPUT_SIZE,MAX_TRACK_LENGTH
import torch
import torch.nn as nn
import im_util
import numpy as np
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class re3Tracker():
	def __init__(self,loss_flag=False,checkpoint_name ='./final_checkpoint/re3_final_checkpoint.pth') :
		
		self.device = device
		self.CNN = CNN(1,1).to(self.device)
		self.RNN = RNN(CNN_OUTPUT_SIZE,1,1,True).to(self.device)
		if os.path.isfile(checkpoint_name):
			checkpoint = torch.load(checkpoint_name,map_location='cpu')
			self.CNN.load_state_dict(checkpoint['cnn_model_state_dict'])
			self.RNN.load_state_dict(checkpoint['rnn_model_state_dict'])

		else:
			print("Invalid/No Checkpoint. Aborting...!!")
			sys.exit()
		self.CNN = self.CNN.to(device)
		self.RNN = self.RNN.to(device)
		self.forward_count = -1
		self.previous_frame = None
		self.cropped_input = np.zeros((2, 3, CROP_SIZE, CROP_SIZE), dtype=np.uint8)
		self.calculate_loss = loss_flag
		self.criterion = nn.MSELoss()
		self.MSE_loss = 0
        
	def track(self,image,starting_box=None,gt_labels=None):
		if starting_box is not None:
			prev_image = image
			past_box = starting_box
			self.forward_count = 0
		else:
			prev_image, past_box = self.previous_frame

		image_0, output_box0 = im_util.get_crop_input(prev_image, past_box, CROP_PAD, CROP_SIZE)
		self.cropped_input[0, ...] = data_preparation(image_0)

		image_1,_ = im_util.get_crop_input(image, past_box, CROP_PAD, CROP_SIZE)
		self.cropped_input[1, ...] = data_preparation(image_1)

		cropped_input_tensor = torch.from_numpy((self.cropped_input))
		cropped_input_tensor = cropped_input_tensor.view(-1,3,CROP_SIZE,CROP_SIZE)
		with torch.no_grad():
			features = self.CNN(cropped_input_tensor.to(self.device))
			predicted_bbox = self.RNN(features)
		# Loss Calculation
		if starting_box is None and self.calculate_loss == True:
                        gt_labels = torch.from_numpy(gt_labels).float()
                        gt_labels = gt_labels.to(self.device)
                        loss = self.criterion(predicted_bbox,gt_labels)
                        # Running averagae loss
                        self.MSE_loss = (self.MSE_loss*self.forward_count + loss)/(self.forward_count +1)
                        print(self.MSE_loss)
		predicted_bbox_array = predicted_bbox.cpu().numpy()
		#print(predicted_bbox_array.squeeze())
                
                # Save initial LSTM states
		predicted_bbox_array = predicted_bbox.numpy()
		print(predicted_bbox_array.squeeze())
        
        # Save initial LSTM states
		if self.forward_count == 0:
			self.RNN.lstm_state_init()

		output_bbox = im_util.from_crop_coordinate_system(predicted_bbox_array.squeeze() / 10.0, output_box0,1,1)

        # Reset LSTM states to initial state once #MAX_TRACK_LENGTH frames are processed and perform one forward pass
		if self.forward_count > 0 and self.forward_count % MAX_TRACK_LENGTH == 0:
			cropped_input,_ = im_util.get_crop_input(image,output_bbox,CROP_PAD,CROP_SIZE)
			cropped_input = data_preparation(cropped_input)
			input_image = np.tile(cropped_input[np.newaxis,...],(2,1,1,1))
			input_tensor = torch.from_numpy(np.float32(input_image)).to(self.device)
			#input_tensor = input_tensor.view(-1,3,CROP_SIZE,CROP_SIZE)
			self.RNN.reset()
			features = self.CNN(input_tensor)
			prediction = self.RNN(features)
		if starting_box is not None:
			output_bbox = starting_box
                
		self.forward_count += 1
		self.previous_frame = (image, output_bbox)
		return output_bbox


