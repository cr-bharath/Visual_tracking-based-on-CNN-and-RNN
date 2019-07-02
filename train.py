import torch
import torch.nn as nn
import numpy as np
import os
from model import ConvNet, RNN
import constants
import argparse

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def main(args):
    # Define Models
    CNN = ConvNet(args.batch_size,args.unroll).to(device)
    RNN = RNN(CNN_OUTPUT_SIZE,args.unroll).to(device)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--unroll', type=int, default=2)
    parser.add_argument('--learning_rate', type=float, default=0.00001)
    parser.add_argument('--data_path', type=str, default="data/", help='path were data is stored')
    args = parser.parse_args()
    main(args)