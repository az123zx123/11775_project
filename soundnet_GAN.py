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

from numpy import genfromtxt

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
nz = 1024

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

class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
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

def train_epoch(netD, netG, optimizerD, optimizerG, dataloader):
    real_label = 1.
    fake_label = 0.
    G_losses = []
    D_losses = []
    iters = 0

    #dataloader: X_train: soundnet y_train: MRI
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        ## Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu = data[1].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Forward pass real batch through D
        # Calculate loss on all-real batch
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x = output.mean().item()

        ## Train with all-fake batch
        # Generate batch of latent vectors
        sound_cpu = data[0].to(device)
        # Generate fake image batch with G
        generate = netG(sound_cpu)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output = netD(generate.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake = criterion(output, label)
        # Calculate the gradients for this batch, accumulated (summed) with previous gradients
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # Compute error of D as sum over the fake and the real batches
        errD = errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        sound_cpu = data[0].to(device)
        output = netD(generate).view(-1)
        # Calculate G's loss based on this output
        errG = criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2 = output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                 % (epoch, num_epochs, i, len(dataloader),
                    errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())
        iters += 1
    return G_losses, D_losses



device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

for speaker in ['f2']:
    # TODO: modify this according to your data path
    dir_mri = './data/'+speaker+ '/avi/'
    dir_sound = './data/'+speaker + '/soundnet/avg_pooling/y_scns/'


    
    # Parameters of vocoder
    samplingFrequency = 20000
    frameLength = 1024 # 
    frameShift = 863 # 43.14 ms at 20000 Hz sampling, correspondong to 23.18 fps (MRI video)
    order = 24
    alpha = 0.42
    stage = 3

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
            if ".csv" in file:
                files_sound ['all'] += [file]
            
    files_sound ['valid'] = files_sound ['all'][0:4]
    files_sound ['test'] = files_sound ['all'][4:6]
    files_sound ['train'] = files_sound ['all'][6:]

    
    max_mri_frames = 500
    select = 25
    for train_valid in ['train', 'valid']:
        n_files = len(files_mri[train_valid])
        n_file = 0
        n_max_mri_frames = n_files * 1000
        mri[train_valid] = np.empty((n_files, n_width, n_height))
        sound[train_valid] = np.empty((n_files, nz))
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
                mri[train_valid][n_file] = np.mean(mri_data,axis=2) # original, 68x68
                n_file += 1
        n_file = 0
        for file in files_sound[train_valid]:
            sound[train_valid][n_file] = genfromtxt(dir_sound+file)
            n_file += 1


        mri[train_valid] = mri[train_valid].reshape(-1, n_width*n_height)


    
    # target: min max scaler to [0,1] range
    # already scaled in load_video
    
    # input: normalization to zero mean, unit variance
    # feature by feature
    
    y_train = torch.tensor(mri['train']).float().reshape((-1,1,n_width,n_height))
    X_train = torch.tensor(sound['train']).float().reshape((-1,nz,1,1))
    dataset_train = torch.utils.data.TensorDataset(X_train, y_train)



    dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)    
    
    # Create the generator
    netG = Generator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netG = nn.DataParallel(netG, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.02.
    netG.apply(weights_init)

    # Print the model
    print(netG)
    
    # Create the Discriminator
    netD = Discriminator(ngpu).to(device)

    # Handle multi-gpu if desired
    if (device.type == 'cuda') and (ngpu > 1):
        netD = nn.DataParallel(netD, list(range(ngpu)))

    # Apply the weights_init function to randomly initialize all weights
    #  to mean=0, stdev=0.2.
    netD.apply(weights_init)

    # Print the model
    print(netD)
    
    # Initialize BCELoss function
    criterion = nn.BCELoss()

    # Create batch of latent vectors that we will use to visualize
    #  the progression of the generator
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)

    # Setup Adam optimizers for both G and D
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))
    
    
    print("Starting Training Loop...")
    # For each epoch
    for epoch in range(num_epochs):
        train_epoch(netD, netG, optimizerD, optimizerG, dataloader_train)
