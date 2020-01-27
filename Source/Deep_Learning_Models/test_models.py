# array job batch index
#import sys
#batch_num = int(sys.argv[1])

# function imports
import numpy as np
import keras
from keras.utils.io_utils import HDF5Matrix
import pandas as pd
from keras.models import Model, load_model
from keras.layers import Dense, Flatten
from keras.layers import Conv3D, MaxPooling3D, Dropout,Conv1D, MaxPooling1D,Conv2D, MaxPooling2D,MaxPool3D
from keras.callbacks import Callback
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dropout, Input, BatchNormalization
from keras.activations import relu
from keras.layers import Dense, Activation
from keras.constraints import max_norm
import h5py
from scipy.stats import zscore, pearsonr
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


############# PREP DATA


# load data for a normal fit procedure
x = np.load('/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512_float16.npy')
x = np.moveaxis(x, 1, -1)
file_name = '/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512.h5'
h5f = h5py.File(file_name, 'r')
labels = h5f['labels'][:]
labels = labels.astype(str)
sex = h5f['sex'][:]

# load index splits for 5 fold CV
folds = pd.read_csv(os.path.join(wdata,'lbrueggeman/ukbb_sex/data/folds.tsv'), sep='\t')
folds = folds[folds['set'] == 'test']
test_labels = folds['labels']
test_ind = np.hstack(np.hstack([np.where(labels.astype(str) == str(i)) for i in test_labels]))

mod1 = load_model('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/trained_models/2020-01-22-cnn-set1_sex_LRe5')
preds = mod1.predict(x[test_ind], batch_size=16)

results = pd.DataFrame({'labels' : test_labels, 'CNN':preds[:,0]})


from sklearn.metrics import roc_auc_score
roc_auc_score(sex[test_ind], preds[:,0])















# load region masks
first = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/first_atlas_t1.npy')
first_labels = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/first_atlas_t1_labels.npy')

fast = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/fast_atlas_t1.npy')
fast_labels = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/fast_atlas_t1_labels.npy')

# load images
file_name = '/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512.h5'
h5f = h5py.File(file_name, 'r')

x = h5f['mri'][]
x = np.expand_dims(x, axis=4)
x = np.divide(x, x.max(axis=(1,2,3,4), keepdims=True))
labels = h5f['labels'][]
labels = labels.astype(str)

# residualized sex values (y)
sex_resid = pd.read_csv(os.path.join(wdata, 'lbrueggeman/ukbb_sex/data/bvol_table.csv'))
sex_resid.set_index(sex_resid['subject_id'].astype(str), inplace=True)
sex_resid = sex_resid['sex_resid']
sex_resid = sex_resid[labels.astype(str)]

# get indices which are >2 SD (outliers)
drop_thres = np.where(np.invert(abs(zscore(sex_resid)) > 2))

# load validation and test set splits
folds = pd.read_csv(os.path.join(wdata,'lbrueggeman/ukbb_sex/data/folds.tsv'), sep='\t')
folds_test = folds[folds['set'] == 'test']
folds = folds[np.invert(folds['set'] == 'test')]

test_labels = folds_test['labels'].astype(str)
val_labels1 = folds['labels'][folds['set'] == 'fold' + '1'].astype(str)
val_labels2 = folds['labels'][folds['set'] == 'fold' + '2'].astype(str)
val_labels3 = folds['labels'][folds['set'] == 'fold' + '3'].astype(str)
val_labels4 = folds['labels'][folds['set'] == 'fold' + '4'].astype(str)
val_labels5 = folds['labels'][folds['set'] == 'fold' + '5'].astype(str)

test_ind = np.hstack(np.hstack([np.where(labels.astype(str) == i) for i in test_labels]))
val_ind1 = np.hstack(np.hstack([np.where(labels.astype(str) == i) for i in val_labels1]))
val_ind2 = np.hstack(np.hstack([np.where(labels.astype(str) == i) for i in val_labels2]))
val_ind3 = np.hstack(np.hstack([np.where(labels.astype(str) == i) for i in val_labels3]))
val_ind4 = np.hstack(np.hstack([np.where(labels.astype(str) == i) for i in val_labels4]))
val_ind5 = np.hstack(np.hstack([np.where(labels.astype(str) == i) for i in val_labels5]))

test_ind = np.array([x for x in test_ind if x in drop_thres[0]])
val_ind1 = np.array([x for x in val_ind1 if x in drop_thres[0]])
val_ind2 = np.array([x for x in val_ind2 if x in drop_thres[0]])
val_ind3 = np.array([x for x in val_ind3 if x in drop_thres[0]])
val_ind4 = np.array([x for x in val_ind4 if x in drop_thres[0]])
val_ind5 = np.array([x for x in val_ind5 if x in drop_thres[0]])


########### PERFORMANCE COMPARISON


# we need to get 5 cross validated values, and the test set performance

mod1 = load_model(os.path.join(wdata, 'lbrueggeman/ukbb_sex/trained_models/2019-10-21-cnn-set1_sexresid_LRe5'))
mod2 = load_model(os.path.join(wdata, 'lbrueggeman/ukbb_sex/trained_models/2019-10-21-cnn-set2_sexresid_LRe5'))
mod3 = load_model(os.path.join(wdata, 'lbrueggeman/ukbb_sex/trained_models/2019-10-21-cnn-set3_sexresid_LRe5'))
mod4 = load_model(os.path.join(wdata, 'lbrueggeman/ukbb_sex/trained_models/2019-10-21-cnn-set4_sexresid_LRe5'))
mod5 = load_model(os.path.join(wdata, 'lbrueggeman/ukbb_sex/trained_models/2019-10-21-cnn-set5_sexresid_LRe5'))

cv1 = mod1.predict(x[val_ind1], batch_size=32)
cv2 = mod2.predict(x[val_ind2], batch_size=32)
cv3 = mod3.predict(x[val_ind3], batch_size=32)
cv4 = mod4.predict(x[val_ind4], batch_size=32)
cv5 = mod5.predict(x[val_ind5], batch_size=32)
test_pred = mod5.predict(x[test_ind], batch_size=32)

cv1_out = np.square(pearsonr(sex_resid[val_ind1], cv1[:,0])[0])
cv2_out = np.square(pearsonr(sex_resid[val_ind2], cv2[:,0])[0])
cv3_out = np.square(pearsonr(sex_resid[val_ind3], cv3[:,0])[0])
cv4_out = np.square(pearsonr(sex_resid[val_ind4], cv4[:,0])[0])
cv5_out = np.square(pearsonr(sex_resid[val_ind5], cv5[:,0])[0])
test_out = np.square(pearsonr(sex_resid[test_ind], test_pred[:,0])[0])

perf_dat = pd.DataFrame({'set' : np.array(('cv','cv','cv','cv','cv','test')),
                         'value' : np.array((cv1_out, cv2_out, cv3_out, cv4_out, cv5_out, test_out))})

perf_dat.to_csv(os.path.join(wdata, 'lbrueggeman/ukbb_sex/data/cnn_performance.tsv'), sep='\t', index=False)


########### REGION PRIORITIZATION


# get first & fast atlas in same order as test images
nix = [np.where(x == first_labels)[0] for x in labels[test_ind].astype(str)]
nix = np.hstack(nix)

first = first[nix]
first_labels = first_labels[nix]

nix = [np.where(x == fast_labels)[0] for x in labels[test_ind].astype(str)]
nix = np.hstack(nix)

fast = fast[nix]
fast_labels = fast_labels[nix]

##### 1) saliency
import vis
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay, visualize_cam

def get_gradient(x, model, num_imgs):
    grads_out = np.empty((num_imgs,128, 128, 128))
    for i in range(num_imgs):
        print(i)
        # 20 is the imagenet index corresponding to `ouzel`
        grads_out[i] = visualize_saliency(model, layer_idx=-1, filter_indices=None, 
                                seed_input=x[i], backprop_modifier=None)
    return(grads_out)

sal = get_gradient(x[test_ind[0:50]], mod5, 50) 

sal_out = []
reg_out = []

for i in np.unique(first[0:3])[1:]:
    sal_out.append(np.mean(sal[first[0:50] == i]))
    reg_out.append(i)

for i in np.unique(fast[0:3])[1:]:
    sal_out.append(np.mean(sal[fast[0:50] == i]))
    reg_out.append(i)

foo = pd.DataFrame({'region':reg_out,
              'val':sal_out,
              'atlas':np.hstack((np.repeat('first',15), np.repeat('fast',3))),
              'method':'saliency'
              })

##### 2) occlusion (voxel corrected)

# first
marker = []
r_out = []
predictions = []

for i in np.unique(first[0:3])[1:]:
    print(i)
    xcp = x[test_ind].copy()
    xcp[first == i] = 0
    shuff_pred = mod5.predict(xcp)
    
    marker.append(i)
    predictions.append(shuff_pred)
    r_out.append( pearsonr(shuff_pred[:,0], sex_resid[test_ind]) )
    print(r_out)

# fast
markerf = []
r_outf = []
predictionsf = []

for i in np.unique(fast[0:3])[1:]:
    print(i)
    xcp = x[test_ind].copy()
    xcp[fast == i] = 0
    shuff_pred = mod5.predict(xcp)
    
    markerf.append(i)
    predictionsf.append(shuff_pred)
    r_outf.append( pearsonr(shuff_pred[:,0], sex_resid[test_ind]) )
    print(r_outf)

r_out = np.array(r_out)
r_outf = np.array(r_outf)

# get avg num voxels for each region type

i_out = []
vox_num = []

for i in np.unique(fast[0])[1:]:
    print(i)
    i_out.append(i)
    vox_num.append( np.sum(fast == i)/fast.shape[0] )

i_outf = []
vox_numf = []

for i in np.unique(first[0])[1:]:
    print(i)
    i_outf.append(i)
    vox_numf.append( np.sum(first == i)/first.shape[0] )

bar = pd.DataFrame({'region' : marker, 'val' : r_out[:,0], 'atlas' : 'first', 'method':'occlusion_decreaser2'})
nix = pd.DataFrame({'region' : markerf, 'val' : r_outf[:,0], 'atlas' : 'fast','method':'occlusion_decreaser2'})

region_out = pd.concat([nix,bar])
region_out['val'] = np.sqrt(test_out) - region_out['val']

bar = pd.DataFrame({'region' : region_out['region'],
                    'val' : (region_out['val']/np.hstack(np.array((vox_num, vox_numf)))),
                    'atlas' : region_out['atlas'],
                    'method':'occlusion_occlusion_descreaser2_s'})

region_out = pd.concat([region_out, bar, foo])
region_out.to_csv(os.path.join(wdata, 'lbrueggeman/ukbb_sex/data/2019-10-21-cnn_region_scores.tsv'), sep='\t', index=False)
