# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 15:08:53 2022

@author: Xiang LI
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as io_wav
import os
import os.path
import datetime
import pickle
import cv2
import random

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from numpy import genfromtxt

# Set random seed for reproducibility
manualSeed = 775
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 68

# Number of channels in the training images. For color images this is 3
nc = 1

# Size of z latent vector (i.e. size of generator input)
nz = 1024

# Size of feature maps in generator
ngf = 68

# Size of feature maps in discriminator
ndf = 68

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

ngpu = 1

device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")


# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# from LipReading with slight modifications
# https://github.com/hassanhub/LipReading/blob/master/codes/data_integration.py
################## VIDEO INPUT ##################
def load_video_3D(path, framesPerSec):
    
    cap = cv2.VideoCapture(path)
    frameCount = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT ))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH ))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # make sure that all the videos are the same FPS
    if (np.abs(fps - framesPerSec) > 0.01):
        print('fps:', fps, '(' + path + ')')
        raise

    buf = np.empty((frameHeight, frameWidth, frameCount), np.dtype('float32'))
    fc = 0
    ret = True
    
    while (fc < frameCount  and ret):
        ret, frame = cap.read()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = frame.astype('float32')
        # min-max scaling to [0-1]
        frame = frame-np.amin(frame)
        # make sure not to divide by zero
        if np.amax(frame) != 0:
            frame = frame/np.amax(frame)
        buf[:,:,fc]=frame
        fc += 1
    cap.release()

    return buf

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            torch.nn.LSTM(input_size =nz, hidden_size =n_width* n_height, num_layers = 2),
        )
    def forward(self, input):
        return self.main(input)
    
#dataloader: X_train: soundnet y_train: MRI
def train_epoch(net, optimizer, dataloader):
    for i, data in enumerate(dataloader, 0):
        net.cpu()
        net.zero_grad()
        X = data[0].cpu()
        y = data[1].cpu()
        output = net(X)[0]
        err = criterion(output, y)
        # Calculate gradients for D in backward pass
        err.backward()
        optimizer.step()

def train_single(net, optimizer, x_train, y_train):
        net.cpu()
        net.zero_grad()
        X = x_train.cpu()
        y = y_train.cpu()
        output = net(X)[0]
        err = criterion(output, y)
        # Calculate gradients for D in backward pass
        err.backward()
        optimizer.step()
        return err.item()

# convert an array of values into a dataset matrix
# code with modifications from
# https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/
def create_dataset_img_inverse(data_in_X, data_in_Y, look_back=1):
    (dim1_X, dim2_X) = data_in_X.shape
    (dim1_Y, dim2_Y, dim3_Y, dim4_Y) = data_in_Y.shape
    data_out_X = np.empty((dim1_X - look_back - 1, look_back, dim2_X))
    data_out_Y = np.empty((dim1_Y - look_back - 1, dim2_Y, dim3_Y, dim4_Y))
    
    for i in range(dim1_X - look_back - 1):
        for j in range(look_back):
            data_out_X[i, j] = data_in_X[i + j]
        data_out_Y[i] = data_in_Y[i + j]
    return data_out_X, data_out_Y
criterion = nn.MSELoss()
for speaker in ['f2']:
    # TODO: modify this according to your data path
    dir_mri = './data/'+speaker+ '/avi/'
    dir_sound = './data/'+speaker + '/soundnet/raw/'

    # context window of LSTM
    n_sequence = 10
    
    # properties of MRI videos
    framesPerSec = 23.18
    n_width = 68
    n_height = 68
    
    # USC-TIMIT contains 92 files (460 sentences) for each speaker
    # train-valid-test split (random) :
    # - 4 files for valid
    # - 2 files for test
    # - the remaining (86 files) for training
    files_mri = dict()
    mri = dict()
    sound = dict()
    files_mri['all'] = []
    if os.path.isdir(dir_mri):
        for file in sorted(os.listdir(dir_mri)):
            if ".avi" in file:
                files_mri['all'] += [file]
    
    # randomize file order
    random.seed(17)
    random.shuffle(files_mri['all'])
    
    files_mri['valid'] = files_mri['all'][0:4]
    files_mri['test'] = files_mri['all'][4:6]
    files_mri['train'] = files_mri['all'][6:]
    
    print('valid files', files_mri['valid'])
    print('test files', files_mri['test'])   # ['usctimit_mri_f1_146_150.avi', 'usctimit_mri_f1_441_445.avi']
    
    files_sound = dict()
    files_sound ['all'] = []
    if os.path.isdir(dir_sound):
        for file in sorted(os.listdir(dir_sound )):
            if ".npy" in file:
                files_sound ['all'] += [file]
            
    files_sound ['valid'] = files_sound ['all'][0:4]
    files_sound ['test'] = files_sound ['all'][4:6]
    files_sound ['train'] = files_sound ['all'][6:]

    max_mri_frames = 400
    select = 25
    sound_frame = 16
    for train_valid in ['train', 'valid']:
        n_files = len(files_mri[train_valid])
        n_file = 0
        n_max_mri_frames = n_files * 1000
        mri[train_valid] = np.zeros((n_files, int(max_mri_frames/select),n_width, n_height))
        #sound[train_valid] = np.zeros((n_files, nz,sound_frame))
        mri_size = 0
        sound_size = 0

        for file in files_mri[train_valid]:
            try:
                print('starting', train_valid, file)
                mri_data = load_video_3D(dir_mri + file, framesPerSec)
            except ValueError as e:
                print("wrong data, check manually!", e)
            
            else:
                print('minmax:', np.min(mri_data), np.max(mri_data))
                
                
                mgc_mri_len = mri_data.shape[2]
                
                mri_data = mri_data[:, :, 0:mgc_mri_len]
                temp_mri = mri_data[:,:,0:min(mgc_mri_len,int(max_mri_frames))]
                temp_mri = temp_mri[:,:,range(0,min(mgc_mri_len,int(max_mri_frames)),select)]
                for i in range(temp_mri.shape[2]):
                    mri[train_valid][n_file][i,:,:] = temp_mri[:,:,i].squeeze()
                n_file += 1
        n_file = 0
        '''
        for file in files_sound[train_valid]:
            temp_sound = np.load(dir_sound+file,allow_pickle=True).item()['conv7'].squeeze()
            sound[train_valid][n_file][:,0:min(temp_sound.shape[1],sound_frame)] = temp_sound[:,0:min(temp_sound.shape[1],sound_frame)]
            n_file += 1
        '''


        #mri[train_valid] = mri[train_valid].reshape(-1, n_width*n_height)
        
        #x_train = torch.nn.utils.rnn.pack_padded_sequence(torch.tensor(sound['train'].reshape(sound_frame,sound['train'].shape[0],nz)),sound_frame*np.ones((sound['train'].shape[0],)), batch_first=False)
        #y_train = torch.nn.utils.rnn.pack_padded_sequence(torch.tensor(mri['train'].reshape(sound_frame,mri['train'].shape[0],n_width, n_height)),sound_frame*np.ones((mri['train'].shape[0],)), batch_first=False)
        
        y_train = torch.tensor(mri['train'].reshape(sound_frame,mri['train'].shape[0],n_width* n_height))
        for train_valid in ['train', 'valid']:
            n_files = len(files_mri[train_valid])
            sound[train_valid] = np.zeros((n_files, nz,sound_frame))
            sound_size = 0
            n_file = 0
        for file in files_sound[train_valid]:
            temp_sound = np.load(dir_sound+file,allow_pickle=True).item()['conv7'].squeeze()
            sound[train_valid][n_file][:,0:min(temp_sound.shape[1],sound_frame)] = temp_sound[:,0:min(temp_sound.shape[1],sound_frame)]
            n_file += 1     
        net = Network()
        
        x_train = torch.tensor(sound['train'].reshape(sound_frame,sound['train'].shape[0],nz))
        optimizerLSTM = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))
        dataset_train = torch.utils.data.TensorDataset(x_train.float(), y_train.float())
        err_list = []
    for epoch in range(20):
        err = train_single(net, optimizerLSTM, x_train.float(), y_train.float())
        print(err)
        err_list.append(err)