# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 13:06:15 2022

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


# Set random seed for reproducibility
manualSeed = 775
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Root directory for dataset
#dataroot = "data/celeba"

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
nz = 100

# Size of feature maps in generator
ngf = 68

# Size of feature maps in discriminator
ndf = 68

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

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

# Generator Code

class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 68 x 68
        )

    def forward(self, input):
        return self.main(input)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

for speaker in ['f1', 'f2', 'm1', 'm2']:
    # TODO: modify this according to your data path
    dir_mri = '/home/csapot/deep_learning_mri/usctimit_mri/' + speaker + '/'
    
    # Parameters of vocoder
    samplingFrequency = 20000
    frameLength = 1024 # 
    frameShift = 863 # 43.14 ms at 20000 Hz sampling, correspondong to 23.18 fps (MRI video)
    order = 24
    alpha = 0.42
    stage = 3
    n_mgc = order + 1

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
    mgc = dict()
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
    
    for train_valid in ['train', 'valid']:
        n_files = len(files_mri[train_valid])
        n_file = 0
        n_max_mri_frames = n_files * 1000
        mri[train_valid] = np.empty((n_max_mri_frames, n_width, n_height))
        mgc[train_valid] = np.empty((n_max_mri_frames, n_mgc))
        mri_size = 0
        mgc_size = 0

        for file in files_mri[train_valid]:
            try:
                print('starting', train_valid, file)
                mri_data = load_video_3D(dir_mri + file, framesPerSec)
                (mgc_lsp_coeff, lf0) = get_mgc_lsp_coeff(dir_mri + file[:-4])
            except ValueError as e:
                print("wrong data, check manually!", e)
            
            else:
                print('minmax:', np.min(mri_data), np.max(mri_data))
                n_file += 1
                
                mgc_mri_len = np.min((mri_data.shape[2], len(mgc_lsp_coeff)))
                
                mri_data = mri_data[:, :, 0:mgc_mri_len]
                mgc_lsp_coeff = mgc_lsp_coeff[0:mgc_mri_len]
                
                if mri_size + mgc_mri_len > n_max_mri_frames:
                    raise
                
                for i in range(mgc_mri_len):
                    mri[train_valid][mri_size + i] = mri_data[:, :, i] # original, 68x68
                    mgc[train_valid][mgc_size + i] = mgc_lsp_coeff[i]
                
                mri_size += mgc_mri_len
                mgc_size += mgc_mri_len
                
                print('n_frames_all: ', mri_size, 'mgc_size: ', mgc_size)
                    
        mri[train_valid] = mri[train_valid][0 : mri_size].reshape(-1, n_width*n_height)
        mgc[train_valid] = mgc[train_valid][0 : mgc_size]


    
    # target: min max scaler to [0,1] range
    # already scaled in load_video
    
    # input: normalization to zero mean, unit variance
    # feature by feature
    mgc_scalers = []
    for i in range(n_mgc):
        mgc_scaler = StandardScaler(with_mean=True, with_std=True)
        mgc_scalers.append(mgc_scaler)
        mgc['train'][:, i] = mgc_scalers[i].fit_transform(mgc['train'][:, i].reshape(-1, 1)).ravel()
        mgc['valid'][:, i] = mgc_scalers[i].transform(mgc['valid'][:, i].reshape(-1, 1)).ravel()
        
    