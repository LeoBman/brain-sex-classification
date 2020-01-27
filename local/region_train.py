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
    wdata = '/Dedicated/jmichaelson-wdata'
else:
    environ = 'local'
    sdata = '/sdata'
    wdata = '/wdata'

#### GET STRUCTURE AND ATLAS TO MODEL

import sys
struct_num = int(sys.argv[1]) - 1

struct_to_model = list((10,11,12,13,16,17,18,26,49,50,51,52,53,54,58,1,2,3))
atlas_to_model = list(np.concatenate((np.repeat('first',15), np.repeat('fast',3))))

struct_to_model = struct_to_model[struct_num]
atlas_to_model = atlas_to_model[struct_num]

#### PREP PATHS FOR SAVING

model_save_fpath = os.path.join(wdata, 'lbrueggeman/Brain-Region-Model-Evaluations/Data', 'model_1sex_' + str(atlas_to_model) + '_' + str(struct_to_model) + '.mod') 

pred_save_fpath = os.path.join(wdata, 'lbrueggeman/Brain-Region-Model-Evaluations/Data', 'preds_1sex_' + str(atlas_to_model) + '_' + str(struct_to_model) + '.npy') 

trainhist_save_fpath = os.path.join(wdata, 'lbrueggeman/Brain-Region-Model-Evaluations/Data', 'trainhist_sex_' + str(atlas_to_model) + '_' + str(struct_to_model) + '.csv') 

#################### DATA IMPORT

# MRIs 
x = np.load('/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512_float16.npy')
x = np.moveaxis(x, 1, -1)
file_name = '/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512.h5'
h5f = h5py.File(file_name, 'r')
labels = h5f['labels'][:]
labels = labels.astype(str)
sex = h5f['sex'][:]

# residualized sex values (y)
#sex_resid = pd.read_csv(os.path.join(wdata, 'lbrueggeman/ukbb_sex/data/bvol_table.csv'))
#sex_resid.set_index(sex_resid['subject_id'].astype(str), inplace=True)
#sex_resid = sex_resid['sex_resid']
#sex_resid = sex_resid[labels.astype(str)]
# get indices which are >2 SD (outliers)
#drop_thres = np.where(np.invert(abs(zscore(sex_resid)) > 2))
#sex_resid = sex_resid.iloc[drop_thres]

# import first & fast atlas
if atlas_to_model == 'first':
    atlas = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/first_atlas_t1.npy')
    atlas_labels = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/first_atlas_t1_labels.npy')
else:
    atlas = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/fast_atlas_t1.npy')
    atlas_labels = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/fast_atlas_t1_labels.npy')

#################### REORDER & OVERLAP DATA

# get overlap between mris, segmentations, and sex resid scores
shared_labels = np.intersect1d(labels, atlas_labels)

# get all items in same order as shared_labels
x_ind = np.array([np.where(x == labels)[0][0] for x in shared_labels])
atlas_ind = np.array([np.where(x == atlas_labels)[0][0] for x in shared_labels])

# apply order
x = x[x_ind]
labels = labels[x_ind]
sex = sex[x_ind]
#sex_resid = sex_resid.iloc[sex_ind]
atlas = atlas[atlas_ind]

# CV split
train_ind = np.arange(0,15000)
val_ind = np.arange(15000,(15000+3070))
test_ind = np.arange((15000+3070),(15000+6140))

#################### MODEL DEFINITION

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
cp = keras.callbacks.ModelCheckpoint(filepath=model_save_fpath,
                                verbose=1,
                                save_best_only=True)


#################### FIT

atlas = atlas == struct_to_model
x = x[:,:,:,:,0] * atlas
x = np.expand_dims(x,4)

history = mod.fit(x=x[train_ind],
          y=sex[train_ind],
          validation_data = [x[val_ind], sex[val_ind]],
          batch_size=16,
          epochs=50,
          verbose=1,
         shuffle=False,
         callbacks=[es, cp])

# save train history
#pd.DataFrame(history.history).to_csv(
#    filepath=trainhist_save_fpath,
#    index=False, sep='\t')

# load model, predict & save
#mod = load_model(model_save_fpath)
#test_pred = mod.predict(x[test_ind], batch_size=16)
#np.save(pred_save_fpath, test_pred)

print("DONE!")

