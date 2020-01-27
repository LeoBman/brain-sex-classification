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
from keras import backend as K
import h5py
from scipy.stats import zscore
import glob
import re
import nibabel as nib

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.style.use('ggplot')
import seaborn as sns
sns.set()
from matplotlib import gridspec

############ get brain image
nifti_files = glob.glob('/Dedicated/jmichaelson-sdata/UK_Biobank/image_brain/20252/unbiased_image/*.nii.gz', recursive=True)
split_string = re.split(r'/', nifti_files[0])
id = re.split(r'.n',split_string[7])[0]
x = nib.load(nifti_files[0])
x = x.get_data()

# get its atlas
fn = '/Dedicated/jmichaelson-sdata/UK_Biobank/image_brain/20252/first_seg/' + id + '.nii.gz'
first_atlas = nib.load(fn)
first_atlas = first_atlas.get_data()
fn = '/Dedicated/jmichaelson-sdata/UK_Biobank/image_brain/20252/fast_seg/' + id + '.nii.gz'
fast_atlas = nib.load(fn)
fast_atlas = fast_atlas.get_data()

# find best axis, 102 to 106
#for i in range(211):
#    print(i)
#    print(np.unique(first_atlas[:,i,:]))

brain_px = (first_atlas != 0) | (fast_atlas != 0)

roc = pd.read_csv('/Dedicated/jmichaelson-wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/region_aurocs.csv')

cnn_first = first_atlas.copy()
cnn_first = cnn_first.astype(np.float)
cnn_first[cnn_first == 0] = np.nan

cnn_first[cnn_first == 10] = roc[(roc['struct'] == 'thalamus_left') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 11] = roc[(roc['struct'] == 'caudate_left') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 12] = roc[(roc['struct'] == 'putamen_left') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 13] = roc[(roc['struct'] == 'pallidum_left') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 16] = roc[(roc['struct'] == 'brain_stem_fourth_ventricle') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 17] = roc[(roc['struct'] == 'hippocampus_left') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 18] = roc[(roc['struct'] == 'amygdala_left') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 26] = roc[(roc['struct'] == 'accumbens_left') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 49] = roc[(roc['struct'] == 'thalamus_right') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 50] = roc[(roc['struct'] == 'caudate_right') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 51] = roc[(roc['struct'] == 'putamen_right') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 52] = roc[(roc['struct'] == 'pallidum_right') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 53] = roc[(roc['struct'] == 'hippocampus_right') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 54] = roc[(roc['struct'] == 'amygdala_right') & (roc['source'] == 'CNN')]['auroc']
cnn_first[cnn_first == 58] = roc[(roc['struct'] == 'accumbens_right') & (roc['source'] == 'CNN')]['auroc']

# first - bvol
bvol_first = first_atlas.copy()
bvol_first = bvol_first.astype(np.float)
bvol_first[bvol_first == 0] = np.nan

bvol_first[bvol_first == 10] = roc[(roc['struct'] == 'thalamus_left') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 11] = roc[(roc['struct'] == 'caudate_left') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 12] = roc[(roc['struct'] == 'putamen_left') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 13] = roc[(roc['struct'] == 'pallidum_left') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 16] = roc[(roc['struct'] == 'brain_stem_fourth_ventricle') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 17] = roc[(roc['struct'] == 'hippocampus_left') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 18] = roc[(roc['struct'] == 'amygdala_left') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 26] = roc[(roc['struct'] == 'accumbens_left') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 49] = roc[(roc['struct'] == 'thalamus_right') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 50] = roc[(roc['struct'] == 'caudate_right') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 51] = roc[(roc['struct'] == 'putamen_right') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 52] = roc[(roc['struct'] == 'pallidum_right') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 53] = roc[(roc['struct'] == 'hippocampus_right') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 54] = roc[(roc['struct'] == 'amygdala_right') & (roc['source'] == 'volume-based')]['auroc']
bvol_first[bvol_first == 58] = roc[(roc['struct'] == 'accumbens_right') & (roc['source'] == 'volume-based')]['auroc']

# 1 csf, 2 white, 3 gray
# cnn - fast 1
cnn_fast1 = fast_atlas.copy()
cnn_fast1 = cnn_fast1.astype(np.float)
cnn_fast1[cnn_fast1 != 1] = np.nan
cnn_fast1[cnn_fast1 == 1] = roc[(roc['struct'] == 'csf') & (roc['source'] == 'CNN')]['auroc']

cnn_fast2 = fast_atlas.copy()
cnn_fast2 = cnn_fast2.astype(np.float)
cnn_fast2[cnn_fast2 != 2] = np.nan
cnn_fast2[cnn_fast2 == 2] = roc[(roc['struct'] == 'white') & (roc['source'] == 'CNN')]['auroc']

cnn_fast3 = fast_atlas.copy()
cnn_fast3 = cnn_fast3.astype(np.float)
cnn_fast3[cnn_fast3 != 3] = np.nan
cnn_fast3[cnn_fast3 == 3] = roc[(roc['struct'] == 'grey') & (roc['source'] == 'CNN')]['auroc']

# bvol - fast 1
bvol_fast1 = fast_atlas.copy()
bvol_fast1 = bvol_fast1.astype(np.float)
bvol_fast1[bvol_fast1 != 1] = np.nan
bvol_fast1[bvol_fast1 == 1] = roc[(roc['struct'] == 'csf') & (roc['source'] == 'volume-based')]['auroc']

bvol_fast2 = fast_atlas.copy()
bvol_fast2 = bvol_fast2.astype(np.float)
bvol_fast2[bvol_fast2 != 2] = np.nan
bvol_fast2[bvol_fast2 == 2] = roc[(roc['struct'] == 'white') & (roc['source'] == 'volume-based')]['auroc']

bvol_fast3 = fast_atlas.copy()
bvol_fast3 = bvol_fast3.astype(np.float)
bvol_fast3[bvol_fast3 != 3] = np.nan
bvol_fast3[bvol_fast3 == 3] = roc[(roc['struct'] == 'grey') & (roc['source'] == 'volume-based')]['auroc']

# show regs
first_nozero = first_atlas.copy()
first_nozero = first_nozero.astype(np.float)
first_nozero[first_nozero == 0] = np.nan
first_nozero[first_nozero == 10] = 1
first_nozero[first_nozero == 11] = 2
first_nozero[first_nozero == 12] = 3
first_nozero[first_nozero == 13] = 4
first_nozero[first_nozero == 16] = 5
first_nozero[first_nozero == 17] = 6
first_nozero[first_nozero == 18] = 7
first_nozero[first_nozero == 26] = 8
first_nozero[first_nozero == 49] = 1
first_nozero[first_nozero == 50] = 2
first_nozero[first_nozero == 51] = 3
first_nozero[first_nozero == 52] = 4
first_nozero[first_nozero == 53] = 6
first_nozero[first_nozero == 54] = 7
first_nozero[first_nozero == 58] = 8

fast_nozero = fast_atlas.copy()
fast_nozero = fast_nozero.astype(np.float)
fast_nozero[fast_nozero == 0] = np.nan


norm = plt.Normalize(vmin=0.5,vmax=.85)
cmap = plt.cm.gray
cmap2 = plt.cm.tab10
cmap3 = plt.cm.viridis

x[x < .5] = np.nan

plt.close()

nrow=5
ncol=2
plt.style.use('dark_background')
plt.rcParams['axes.facecolor'] = 'black'
plt.rcParams['savefig.facecolor'] = 'black'

fig= plt.figure(figsize=(ncol+1,nrow+1))
gs = gridspec.GridSpec(nrow, ncol,
         wspace=0.0, hspace=0.0, 
         top=1.-0.5/(nrow+1), bottom=0.5/(nrow+1), 
         left=0.5/(ncol+1), right=1-0.5/(ncol+1)) 

# brain view 1
axes = plt.subplot(gs[0,0])
axes.imshow(np.array(x[:,102,:]).astype(np.float64), cmap=cmap, alpha=0.8)
axes.imshow(np.array(first_nozero[:,102,:]).astype(np.float64), cmap=cmap2)

# brain view 1 - other copy
axes = plt.subplot(gs[0,1])
axes.imshow(np.array(x[:,102,:]).astype(np.float64), cmap=cmap, alpha=0.8)
axes.imshow(np.array(fast_nozero[:,102,:]).astype(np.float64), cmap=cmap2)

# view 1 - cnn first
axes = plt.subplot(gs[1,0])
axes.imshow(np.array(x[:,102,:]).astype(np.float64), cmap=cmap, alpha=0.8)
axes.imshow(np.array(cnn_first[:,102,:]).astype(np.float64), cmap=cmap3, norm=norm)

# view 1 - bvol first
axes = plt.subplot(gs[1,1])
axes.imshow(np.array(x[:,102,:]).astype(np.float64), cmap=cmap, alpha=0.8)
axes.imshow(np.array(bvol_first[:,102,:]).astype(np.float64), cmap=cmap3, norm=norm)

# view 1 - cnn fast 1
axes = plt.subplot(gs[2,0])
axes.imshow(np.array(x[:,102,:]).astype(np.float64), cmap=cmap, alpha=0.8)
axes.imshow(np.array(cnn_fast1[:,102,:]).astype(np.float64), cmap=cmap3, norm=norm)

# view 1 - bvol fast 1
axes = plt.subplot(gs[2,1])
axes.imshow(np.array(x[:,102,:]).astype(np.float64), cmap=cmap, alpha=0.8)
axes.imshow(np.array(bvol_fast1[:,102,:]).astype(np.float64), cmap=cmap3, norm=norm)

# view 1 - cnn fast 2
axes = plt.subplot(gs[3,0])
axes.imshow(np.array(x[:,102,:]).astype(np.float64), cmap=cmap, alpha=0.8)
axes.imshow(np.array(cnn_fast2[:,102,:]).astype(np.float64), cmap=cmap3, norm=norm)

# view 1 - bvol fast 2
axes = plt.subplot(gs[3,1])
axes.imshow(np.array(x[:,102,:]).astype(np.float64), cmap=cmap, alpha=0.8)
axes.imshow(np.array(bvol_fast2[:,102,:]).astype(np.float64), cmap=cmap3, norm=norm)

# view 1 - cnn fast 3
axes = plt.subplot(gs[4,0])
axes.imshow(np.array(x[:,102,:]).astype(np.float64), cmap=cmap, alpha=0.8)
axes.imshow(np.array(cnn_fast3[:,102,:]).astype(np.float64), cmap=cmap3, norm=norm)

# view 1 - bvol fast 3
axes = plt.subplot(gs[4,1])
axes.imshow(np.array(x[:,102,:]).astype(np.float64), cmap=cmap, alpha=0.8)
axes.imshow(np.array(bvol_fast3[:,102,:]).astype(np.float64), cmap=cmap3, norm=norm)

#delete axis ticks
for i in range(10):
    fig.axes[i].xaxis.set_major_formatter(plt.NullFormatter())
    fig.axes[i].yaxis.set_major_formatter(plt.NullFormatter())
    fig.axes[i].grid(False)
    fig.axes[i].axis('off')

plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/Brain-Region-Model-Evaluations/Paper/Figures/brain_fig3.png', dpi=300)

foo = x.copy()
foo[np.isnan(foo)] = 0
bar = first_atlas == 49
bar[bar==False] = 0
nix = foo * bar
nix[nix == 0] = np.nan

plt.close()
fig = plt.subplot()
fig.imshow(np.array(nix[:,104,:]).astype(np.float64))
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/Brain-Region-Model-Evaluations/Paper/Figures/first_only_thalamus.png')

nix[np.invert(np.isnan(nix))] = 1
plt.close()
fig = plt.subplot()
fig.imshow(np.array(nix[:,104,:]).astype(np.float64), cmap=cmap2)
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/Brain-Region-Model-Evaluations/Paper/Figures/first_only_thalamus_color.png')


bar = first_atlas.copy()
bar = bar.astype(np.float)
bar[bar == 0] = np.nan

plt.close()
fig = plt.subplot()
fig.imshow(np.array(x[:,104,:]).astype(np.float64))
fig.imshow(np.array(bar[:,104,:]).astype(np.float64), cmap=cmap2)
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/Brain-Region-Model-Evaluations/Paper/Figures/first_all_struct.png')

plt.close()
fig = plt.subplot()
fig.imshow(np.array(x[:,104,:]).astype(np.float64))
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/Brain-Region-Model-Evaluations/Paper/Figures/first_just_brain.png')









np.unique(cnn_fast)


def static_plot(grads=0, avg_brain=0, fpath='', ncol=22, fig_ht=10, fig_wd=80, grads_col='Reds', brain_col='Greys', brain_alpha=0.8):
    
    fig, axes = plt.subplots(6,ncol, dpi=150)
    fig.set_figheight(fig_ht)
    fig.set_figwidth(fig_wd)
    
    norm = plt.Normalize(0, 1)
    
    for i in range(0,ncol,1): # col
        axes[0,i].imshow(avg_brain[:,(1+i)*5,:], cmap=brain_col, alpha=brain_alpha)
        axes[0,i].axis('off')
    
        axes[1,i].imshow(grads[:,(1+i)*5,:], norm=None, cmap=grads_col)
        axes[1,i].axis('off')
            
        axes[2,i].imshow(avg_brain[(1+i)*5,:,:], cmap=brain_col, alpha=brain_alpha)
        axes[2,i].axis('off')
    
        axes[3,i].imshow(grads[(1+i)*5,:,:], norm=None, cmap=grads_col)
        axes[3,i].axis('off')
            
        axes[4,i].imshow(avg_brain[:,:,(1+i)*5], cmap=brain_col, alpha=brain_alpha)
        axes[4,i].axis('off')
    
        axes[5,i].imshow(grads[:,:,(1+i)*5], norm=None, cmap=grads_col)
        axes[5,i].axis('off')
    
    plt.savefig(fpath)








np.unique(first_atlas[:,71,:])

np.sum(first_atlas[:,71,:] == 54)

plt.close()
fig = plt.figure("Lesion mask 3D" ,figsize=(20, 14))
ax = fig.add_subplot(1,1,1, projection='3d')
edge_color = [.5,.5,1,0.1]
alpha=0.5
fc='grey'

np.unique(first_atlas)

for i in np.unique(first_atlas[:,71,:])[1:]:
    if i > 26:
        continue
    foo = first_atlas.copy()
    foo = (foo == i).astype(np.int)
    verts, faces, normals, values = measure.marching_cubes_lewiner(foo)
    mesh = Poly3DCollection(verts[faces], alpha = alpha, facecolor=fc, linewidths=.1)
    
    mesh.set_edgecolor(edge_color)
    ax.add_collection3d(mesh)

ax.set_xlim3d(0, 128)
ax.set_ylim3d(0, 128)
ax.set_zlim3d(0, 128)


ax.view_init(elev=15, azim=135.)
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/3d_first_az315_ele15.png')



for i in range(0,360,45):
    ax.view_init(elev=0, azim=i)
    plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/3d_first_az' + str(i) +'.png')




ax.view_init(elev=0, azim=135.)
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/3d_first.png')

ax.view_init(elev=0, azim=180.)
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/3d_first.png')

ax.view_init(elev=0, azim=225.)
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/3d_first.png')

ax.view_init(elev=0, azim=265.)
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/3d_first.png')

ax.view_init(elev=0, azim=315.)
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/3d_first.png')

ax.view_init(elev=0, azim=360.)
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/3d_first.png')

ax.view_init(elev=0, azim=90.)
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/3d_first.png')



















plt.close()
fig = plt.subplot()
fig.imshow(np.array(first_atlas[:,71,:]).astype(np.float64))
fig.imshow(np.array(first_atlas[:,71,:]).astype(np.float64))
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/first_all_71.png')



plt.close()
fig = plt.figure("Lesion mask 3D" ,figsize=(20, 14))

color='red'
for i in range(1,7,1):
    
    foo = img[0].copy()
    nix = foo.copy()
    if(i > 1):
        color = 'blue'
        bar = rotate_structure(foo)
        nix = shift_structure(bar, brain_px)
    
    mesh = Poly3DCollection(verts[faces], alpha = alpha, facecolor=fc, linewidths=.1)
    ax = fig.add_subplot(2,3,i, projection='3d')
    edge_color = [.5,.5,1,0.1]
    mesh.set_edgecolor(edge_color)
    ax.add_collection3d(mesh)
    ax.set_xlim3d(0, 128)
    ax.set_ylim3d(0, 128)
    ax.set_zlim3d(0, 128)
    ax.scatter(np.where(nix > 0)[0], np.where(nix > 0)[1], np.where(nix > 0)[2], c=color)
    ax.view_init(elev=180.)

plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/brain_structure_permute.png')












plt.close()
fig = plt.subplot()
fig.imshow(np.array(fast_atlas[0,:,:,80]).astype(np.float64))
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/fast_all.png')

plt.close()
fig = plt.subplot()
fig.imshow(np.array(first_atlas[0,64,:,:]).astype(np.float64))
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/first_all.png')


#1 csf 
#2 grey
#3 white



verts, faces, normals, values = measure.marching_cubes_lewiner(brain_px)

plt.close()
fig = plt.figure("Lesion mask 3D" ,figsize=(20, 14))

color='red'
for i in range(1,7,1):
    
    foo = img[0].copy()
    nix = foo.copy()
    if(i > 1):
        color = 'blue'
        bar = rotate_structure(foo)
        nix = shift_structure(bar, brain_px)
    
    mesh = Poly3DCollection(verts[faces], alpha = alpha, facecolor=fc, linewidths=.1)
    ax = fig.add_subplot(2,3,i, projection='3d')
    edge_color = [.5,.5,1,0.1]
    mesh.set_edgecolor(edge_color)
    ax.add_collection3d(mesh)
    ax.set_xlim3d(0, 128)
    ax.set_ylim3d(0, 128)
    ax.set_zlim3d(0, 128)
    ax.scatter(np.where(nix > 0)[0], np.where(nix > 0)[1], np.where(nix > 0)[2], c=color)
    ax.view_init(elev=180.)

plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/brain_structure_permute.png')


plt.close()
fig = plt.figure("Lesion mask 3D" ,figsize=(20, 14))

verts, faces, normals, values = measure.marching_cubes_lewiner(brain_px)
mesh = Poly3DCollection(verts[faces], alpha = alpha, facecolor=fc, linewidths=.1)
ax = fig.add_subplot(1,1,1, projection='3d')
edge_color = [.5,.5,1,0.1]
mesh.set_edgecolor(edge_color)
ax.add_collection3d(mesh)
ax.set_xlim3d(0, 128)
ax.set_ylim3d(0, 128)
ax.set_zlim3d(0, 128)

for i in range(1,1000,1):
    foo = img[0].copy()
    color = 'blue'
    nix = shift_and_rotate(foo,brain_px)
    verts, faces, normals, values = measure.marching_cubes_lewiner(nix)
    mesh = Poly3DCollection(verts[faces], alpha = 1, facecolor=fc, linewidths=.1)
    ax.add_collection3d(mesh)

verts, faces, normals, values = measure.marching_cubes_lewiner(img[0])
mesh = Poly3DCollection(verts[faces], alpha = 1, facecolor='red', linewidths=.1)
ax.add_collection3d(mesh)

ax.view_init(elev=180.)
plt.savefig('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/figs/brain_structure_permute1000.png')



