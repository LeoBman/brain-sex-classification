#################### SETUP


# array job batch index
import sys
batch_num = str(sys.argv[1])

# function imports
import numpy as np
import keras
from keras.utils.io_utils import HDF5Matrix
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Flatten, concatenate
from keras.layers import Conv3D, MaxPooling3D, Dropout,Conv1D, MaxPooling1D, Conv2D, MaxPooling2D, MaxPool3D
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dropout, Input, BatchNormalization
from keras.activations import relu
from keras.layers import Dense, Activation
from keras.constraints import max_norm
import h5py
from scipy.stats import zscore

import os
# detect environment, local vs argon (hpc)
if os.path.exists('/Dedicated'):
    environ = 'hpc'
    sdata = '/Dedicated/jmichaelson-sdata'
    wdata = '/Dedicated/jmichaelson-wdata'
else:
    environ = 'local'
    sdata = '/sdata'
    wdata = '/wdata'


#################### DATA IMPORT


# import data
file_name = '/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512.h5'

# load data for a normal fit procedure
x = np.load('/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512_float16.npy')
x = np.moveaxis(x, 1, -1)
file_name = '/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512.h5'
h5f = h5py.File(file_name, 'r')
labels = h5f['labels'][:]
labels = labels.astype(str)
sex = h5f['sex'][:]

# import brain region volumes
#sex_resid = pd.read_csv(os.path.join(wdata, 'lbrueggeman/ukbb_sex/data/bvol_table.csv'))
#sex_resid.set_index(sex_resid['subject_id'].astype(str), inplace=True)
#sex_resid = sex_resid['sex_resid']
#sex_resid = sex_resid[labels.astype(str)]

# get indices which are >3 SD (outliers)
#drop_thres = np.where(np.invert(abs(zscore(sex_resid)) > 2))

# load index splits for 5 fold CV
folds = pd.read_csv(os.path.join(wdata,'lbrueggeman/ukbb_sex/data/folds.tsv'), sep='\t')
folds = folds[np.invert(folds['set'] == 'test')]

val_labels = folds['labels'][folds['set'] == 'fold' + batch_num].astype(str)
train_labels = folds['labels'][np.invert(folds['set'] == 'fold' + batch_num)].astype(str)

val_ind = np.hstack(np.hstack([np.where(labels.astype(str) == i) for i in val_labels]))
train_ind = np.hstack(np.hstack([np.where(labels.astype(str) == i) for i in train_labels]))

#val_ind = np.array([x for x in val_ind if x in drop_thres[0]])
#train_ind = np.array([x for x in train_ind if x in drop_thres[0]])


#################### MODEL DEFINITION


# model definition
def con_net():
## input layer
    keras.backend.clear_session()
    input_layer = Input((128,128,128,1))
    ## conv 1
    conv1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu', data_format='channels_last')(input_layer)
    pool1 = MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv1)
    batch1 = BatchNormalization()(pool1)
    ## conv 2
    conv2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(batch1)
    pool2 = MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv2)
    batch2 = BatchNormalization()(pool2)
    ## conv 3
    conv3 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(batch2)
    conv4 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(conv3)
    pool3 = MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv4)
    batch3 = BatchNormalization()(pool3)
    ## conv 4
    conv5 = Conv3D(filters=128, kernel_size=(2, 2, 2), activation='relu')(batch3)
    conv6 = Conv3D(filters=256, kernel_size=(2, 2, 2), activation='relu')(conv5)
    pool4 = MaxPool3D(pool_size=(1,2,2), strides=(1,2,2))(conv6)
    flatten1 = Flatten()(pool4)
    ## 20 features 
    dense1 = Dense(19, activation='linear')(flatten1)
    output_layer = Dense(1, activation='sigmoid')(dense1)
    ## define the model with input layer and output layer
    adam = keras.optimizers.Adam(lr=1e-5)
    model = Model(inputs=input_layer, outputs=output_layer) 
    model.compile(loss='binary_crossentropy', metrics=['binary_crossentropy','accuracy'], optimizer=adam)     
    return model

mod = con_net()

# model callbacks
es = keras.callbacks.EarlyStopping(monitor='binary_crossentropy',
                    min_delta=0,
                    patience=3,
                    verbose=0, mode='auto')
cp = keras.callbacks.ModelCheckpoint(filepath='/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/trained_models/2020-01-22-cnn-set{}_sex_LRe5'.format(batch_num),
                                verbose=1,
                                save_best_only=True)


#################### FIT


history = mod.fit(x=x[train_ind],
          y=sex[train_ind],
          validation_data = [x[val_ind], sex[val_ind]],
          batch_size=16,
          epochs=50,
          verbose=1,
         shuffle=False,
         callbacks=[es, cp])

pd.DataFrame(history.history).to_csv(
    filepath='/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/trained_models/2020-01-22-cnn-set{}_sex_LR1e-5.tsv'.format(batch_num),
    index=False, sep='\t')

