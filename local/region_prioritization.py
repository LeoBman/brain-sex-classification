import numpy as np
import h5py

######### LABEL PREP

first_atlas_labels = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/first_atlas_t1_labels.npy')    
fast_atlas_labels = np.load('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/fast_atlas_t1_labels.npy')

file_name = '/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512.h5'
h5f = h5py.File(file_name, 'r')
labels = h5f['labels'][:]
labels = labels.astype(str)
sex = h5f['sex'][:]

# first
first_shared_labels = np.intersect1d(labels, first_atlas_labels) #LABELS
x_ind = np.array([np.where(x == labels)[0][0] for x in first_shared_labels]) 
first_sex = sex[x_ind] #SEX
test_ind = np.arange((15000+3070),(15000+6140))
first_sex = first_sex[test_ind] #SEX TEST
first_shared_labels = first_shared_labels[test_ind] #LABELS TEST

# fast
fast_shared_labels = np.intersect1d(labels, fast_atlas_labels) #LABELS
x_ind = np.array([np.where(x == labels)[0][0] for x in fast_shared_labels]) 
fast_sex = sex[x_ind] #SEX
test_ind = np.arange((15000+3070),(15000+6140))
fast_sex = fast_sex[test_ind] #SEX TEST
fast_shared_labels = fast_shared_labels[test_ind] #LABELS TEST

######### LOAD PREDICTIONS

import glob
import pandas as pd

fn = glob.glob('/Dedicated/jmichaelson-wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/pred_sex_first*')

first_results = [np.load(x)[:,0] for x in fn]
first_results.append(first_shared_labels)

first_names = list(('first_10',
                   'first_11',
                   'first_12',
                   'first_13',
                   'first_16',
                   'first_17',
                   'first_18',
                   'first_26',
                   'first_49',
                   'first_50',
                   'first_51',
                   'first_52',
                   'first_53',
                   'first_54',
                   'first_58',
                   'labels'))

foo = pd.DataFrame.from_items(zip(first_names, first_results))
foo.to_csv('/Dedicated/jmichaelson-wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/first_pred_df.csv', index=False)


fn = glob.glob('/Dedicated/jmichaelson-wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/pred_sex_fast*')

fast_results = [np.load(x)[:,0] for x in fn]
fast_results.append(fast_shared_labels)

fast_names = list(('fast_1',
                   'fast_2',
                   'fast_3',
                   'labels'))

foo = pd.DataFrame.from_items(zip(fast_names, fast_results))
foo.to_csv('/Dedicated/jmichaelson-wdata/lbrueggeman/Brain-Region-Model-Evaluations/Data/fast_pred_df.csv', index=False)
