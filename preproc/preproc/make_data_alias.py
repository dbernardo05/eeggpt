# Copies data from ext HD to local dummy aliases
# to eliminate need for HD once features are generated

import re
import os
import yaml
import numpy as np
import pandas as pd
import h5py
import sys

#from scipy.io import loadmat
from glob import glob

remote_data_dir = '/Volumes/BOBO/__SPQR_data/'
local_data_dir = '../data/'

subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
for subject in subjects:
	base = remote_data_dir + 'data/%s/%s_' % (str(subject).zfill(3), str(subject).zfill(3))
	fnames = (sorted(glob(base + '*_0.h5'),
					 key=lambda x: int(x.replace(base, '')[:-7])) +
			  sorted(glob(base + '*_1.h5'),
					 key=lambda x: int(x.replace(base, '')[:-7])))
	
	loc_subj_dir = os.path.join(local_data_dir, str(subject).zfill(3))
	if not os.path.exists(loc_subj_dir):
		os.makedirs(loc_subj_dir)

	for fname in fnames:
		subj = fname.split('/')[-2]
		fn = fname.split('/')[-1]
		out_fname = os.path.join(local_data_dir, subj, fn)
		with h5py.File(out_fname, 'w') as hf:
			hf.create_dataset(fn,  data=np.array([True], dtype=bool))




