import pandas as pd
import h5py
import numpy as np

# import data
file_name = '/Dedicated/jmichaelson-sdata/UK_Biobank/Processed_unbias_T1/ukbb_dataset_20190512.h5'

# load data for a normal fit procedure
h5f = h5py.File(file_name, 'r')
labels = h5f['labels'][:]

np.random.shuffle(labels)

df = pd.DataFrame({'labels': labels,
              'set': np.hstack(np.array((np.repeat('test',2390), np.repeat('fold1',3800), np.repeat('fold2',3800), np.repeat('fold3',3800), np.repeat('fold4',3800), np.repeat('fold5',3800))))})

df.to_csv('/Dedicated/jmichaelson-wdata/lbrueggeman/ukbb_sex/data/folds.tsv', index=False, sep='\t')