# function imports
import numpy as np
import keras
import pickle
from keras.utils.io_utils import HDF5Matrix
import pandas as pd
from keras.models import Model
from keras.models import Model, load_model
from keras.layers import Dense, Flatten, concatenate
from keras.layers import Conv3D, MaxPooling3D, Dropout,Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, MaxPool3D
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dropout, Input, BatchNormalization
from keras.activations import relu
from keras.layers import Dense, Activation
from keras.constraints import max_norm
from keras import backend as K
import h5py
from scipy.stats import zscore, pearsonr
from skimage import measure
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

import os

# detect environment, local vs argon (hpc)
if os.path.exists('/Dedicated'):
    environ = 'hpc'
    sdata = '/Dedicated/jmichaelson-sdata'
    wdata = '/Dedicated/jmichaelson-wdata/lbrueggeman'
else:
    environ = 'local'
    sdata = '/sdata'
    wdata = '/wdata/lbrueggeman'


#######
print('here')
import sys
struct_num = int(sys.argv[1]) - 1

struct_to_model = list((10,11,12,13,16,17,18,26,49,50,51,52,53,54,58,1,2,3))
atlas_to_model = list(np.concatenate((np.repeat('first',15), np.repeat('fast',3))))

struct_to_model = struct_to_model[struct_num]
atlas_to_model = atlas_to_model[struct_num]


model_fpath = 'Brain-Region-Model-Evaluations/Data/model_sex_' + atlas_to_model + '_' + str(struct_to_model) + '.mod'
pred_fpath = 'Brain-Region-Model-Evaluations/Data/pred_sex_' + atlas_to_model + '_' + str(struct_to_model) + '.npy'
mod = load_model(os.path.join(wdata, model_fpath))

print('it')

if atlas_to_model == 'first':
    atlas = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/first_atlas_t1.npy')
    atlas_labels = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/first_atlas_t1_labels.npy')
else:
    atlas = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/fast_atlas_t1.npy')
    atlas_labels = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/fast_atlas_t1_labels.npy')

# MRIs 
x = np.load('/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512_float16.npy')
x = np.moveaxis(x, 1, -1)
file_name = '/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512.h5'
h5f = h5py.File(file_name, 'r')
labels = h5f['labels'][:]
labels = labels.astype(str)
sex = h5f['sex'][:]

print('goes')

# get overlap between mris, segmentations, and sex resid scores
shared_labels = np.intersect1d(labels, atlas_labels)

# get all items in same order as shared_labels
x_ind = np.array([np.where(x == labels)[0][0] for x in shared_labels])
atlas_ind = np.array([np.where(x == atlas_labels)[0][0] for x in shared_labels])

test_ind = np.arange((15000+3070),(15000+6140))
x_ind = x_ind[test_ind]
atlas_ind = atlas_ind[test_ind]

# apply order
x = x[x_ind]
sex = sex[x_ind]
atlas = atlas[atlas_ind]

print('the')

# filter for structures
atlas = atlas == struct_to_model
x = x[:,:,:,:,0] * atlas
x = np.expand_dims(x,4)

print('long')

# predict
preds = mod.predict(x, batch_size=16)
np.save(os.path.join(wdata,pred_fpath), preds)

print('step')


