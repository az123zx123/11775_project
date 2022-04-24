import torch
import pandas as pd
import os
from numpy import genfromtxt
import numpy as np

speaker = 'f2'
dir_sound = './data/'+speaker + '/soundnet/avg_pooling/y_scns/'
files_sound = dict()
mri = dict()
sound = dict()
files_sound ['all'] = []
if os.path.isdir(dir_sound ):
        for file in sorted(os.listdir(dir_sound )):
            if ".csv" in file:
                files_sound ['all'] += [file]

files_sound ['valid'] = files_sound ['all'][0:4]
files_sound ['test'] = files_sound ['all'][4:6]
files_sound ['train'] = files_sound ['all'][6:]
nz = 1024
for train_valid in ['train', 'valid']:
	n_files = len(files_sound[train_valid])
	sound[train_valid] = np.empty((n_files, nz))
	n_file = 0
	for file in files_sound[train_valid]:
              		sound[train_valid][n_file] = genfromtxt(dir_sound+file)
              		n_file += 1


train_tensor = torch.tensor(sound['train'])
print(train_tensor.shape)