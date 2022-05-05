# -*- coding: utf-8 -*-
"""
Created on Sun Apr 24 15:08:53 2022

@author: Xiang LI
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import os.path
import cv2
import random
from tqdm import trange

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn
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

from numpy import genfromtxt


# get frames in a video, return in shape of (height, width, frame_num)
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
    def __init__(self, input_size, hidden_size):
        super(Network, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2),
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
    net.train()
    optimizer.zero_grad()
    X = x_train
    y = y_train
    output = net(X)[0]
    err = criterion(output, y)
    # Calculate gradients for D in backward pass
    err.backward()
    optimizer.step()
    return err.item()

def eval_single(net, x_val, y_val):
    net.eval()
    with torch.no_grad():
        output = net(x_val)[0]
        err = criterion(output, y_val)
    return err.item()

def test_single(net, x_test):
    net.eval()
    with torch.no_grad():
        output = net(x_test)[0]
    np.save("result", output.cpu().detach().numpy())

# parameters
manualSeed = 775
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
workers = 2
batch_size = 128
image_size = 68 # All images will be resized to this size using a transformer.
nc = 1 # Number of channels in the training images. For color images this is 3
nz = 768 # Size of input vector
num_epochs = 15
lr = 0.0002
ngpu = 1 # Number of GPUs available. Use 0 for CPU mode.
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

criterion = nn.MSELoss()
for speaker in ['F2']:
    dir_mri = './data/' + speaker + '/avi/'
    dir_sound = './data/' + speaker + '/mockingjay/'

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
    files = dict()
    files['all'] = [] # all file names in strings
    if os.path.isdir(dir_mri):
        for file in sorted(os.listdir(dir_mri)):
            if ".avi" in file:
                files['all'] += [file[:-4]]
    # randomize file order
    random.seed(17)
    random.shuffle(files['all'])
    
    files['valid'] = files['all'][0:4]
    files['test'] = files['all'][4:6]
    files['train'] = files['all'][6:]
    
    print('valid files', files['valid'])
    print('test files', files['test'])

    files_sound = dict() # all audio feature file names in strings
    files_sound ['all'] = []
    if os.path.isdir(dir_sound):
        for file in sorted(os.listdir(dir_sound )):
            if ".npy" in file:
                files_sound ['all'] += [file]

    mri = dict()    # MRI data, labels in training
    sound = dict()  # audio feature, input data in training
    template = torch.zeros(size=(n_height, n_width))
    for train_valid in ['train', 'valid', "test"]:
        mri[train_valid] = []
        mri[train_valid+"_framenums"] = []
        sound[train_valid] = []
        # read MRI and audio features
        for file in files[train_valid]:
            try:
                print('starting', train_valid, file)
                mri_data = load_video_3D(dir_mri + file + ".avi", framesPerSec)
                audio_data = np.load(dir_sound + file + ".npy")
            except ValueError as e:
                print("wrong data, check manually!", e)
            else:
                print('minmax:', np.min(mri_data), np.max(mri_data))

            mri_data = torch.from_numpy(mri_data).permute(2, 0, 1)
            video_frame_num = mri_data.shape[0]
            mri[train_valid].append(mri_data.reshape(video_frame_num, -1))
            mri[train_valid+"_framenums"].append(video_frame_num)
            template += torch.mean(mri_data, dim=0)

            audio_data = torch.from_numpy(audio_data)
            audio_frame_num = audio_data.shape[0]
            audio_indices = []
            # align audio features and MRI frames, choose closest audio feature frame for each MRI frame
            for i in range(video_frame_num):
                audio_indices.append(int(i*audio_frame_num/video_frame_num))
            sound[train_valid].append(audio_data[audio_indices])
    template /= len(files['all'])
    plt.imsave("template.png", template)
    template = template.reshape(-1)
    
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    net = Network(nz+n_width*n_height, n_width*n_height).to(device)
    optimizerLSTM = optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        train_err_avg = 0
        for i in trange(len(mri["train"])):
            x_train = sound["train"][i].to(device)
            x_train = torch.cat((x_train, template.repeat(x_train.shape[0], 1)), axis=1)
            y_train = mri["train"][i].to(device)
            train_err = train_single(net, optimizerLSTM, x_train.float(), y_train.float())
            train_err_avg += train_err
        train_err_avg /= len(mri["train"])
        val_err_avg = 0
        for i in range(len(mri["valid"])):
            x_val = sound["valid"][i].to(device)
            y_val = mri["valid"][i].to(device)
            val_err = eval_single(net, x_val, y_val)
            val_err_avg += val_err
        val_err_avg /= len(mri["valid"])
        print(f"epoch {epoch}: training error {train_err_avg}, validation error {val_err_avg}")
    x_test = sound["test"][0].to(device)
    test_single(net, x_test)