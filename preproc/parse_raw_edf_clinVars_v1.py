###
# parse_raw_edf.py
# segments data from EDFs for use in SPQR
# v1 - annotations added
# v2 - adding flexibility for differently named EDFs (Never Responders)
# v3 - for NMT dataset
# v4 - for TUH dataset
# clinVars_v1 - for TUH dataset, just get clinvars


## NOTE, label of '1' means that the patient ABNORMAL
### 

from __future__ import print_function

import pickle as pkl
import fnmatch
import glob
import h5py
# import matplotlib as mpl
# mpl.use('TkAgg')
# import matplotlib.pyplot as plt

import numpy as np
import os
import pandas as pd
import random
import re
import sys


from datetime import datetime
from EDFlib.edfreader import EDFreader
from tqdm import tqdm


def find_files(search_dir, ext, subdirs, datatype):
	matches = []
	ignored = []

	for subdir in subdirs:
		search_subdir = os.path.join(search_dir, datatype, subdir, '01_tcp_ar')
		print('Curr subdir:', search_subdir)
		for root, dirnames, filenames in os.walk(search_subdir):
			for filename in fnmatch.filter(filenames, '*' + ext):
				fname = os.path.join(root, filename)
				if fname.split('/')[-1][0] == '.':
					ignored.append(fname)
				else:
					matches.append(fname)
		print("Ignored:", ignored)

	return matches

def process_labels(edfs, verbose=False):
	# Create subjs dict and labels
	pre_subjs = { }
	for edf in edfs:

		if 'abnormal' in edf:
			lbl='1'
		else:
			lbl='0'

		subj = edf.split("/")[-1].split(".")[0]

		subj = f'{subj}_{lbl}'

		if subj not in pre_subjs:
			pre_subjs[subj] = [ edf ]
		else:
			pre_subjs[subj].append(edf)
	if verbose:
		print('\nPre- subjs:\n', pre_subjs)

	doSomeSubjectOrganizing = True
	if doSomeSubjectOrganizing:
		subjs = pre_subjs

	if verbose:
		print('Post- subjs:\n')
		for subj, edf_files in subjs.items():
			print('\t', subj)
			for edf_file in edf_files:
				print('\t\t', edf_file)
	return subjs


def check_eeg(raw_edf, srate):
	errors = []

	# replace assertions by conditions
	if len(raw_edf.ch_names) != 21:
		errors.append(f"Missing:{list(set(ref_chans) - set(raw_edf.ch_names))}")
	if int(raw_edf.info['sfreq']) != srate:
		errors.append(f"SR != {srate}")
	if raw_edf.info['sfreq'] != get_srate(raw_edf):
		errors.append(f"Sampling freq mismatch (reported/calculated): {raw_edf.info['sfreq'], get_srate(raw_edf)}")

	# assert no error message has been registered, else print messages
	assert not errors, "errors occured:\n{}".format("\n".join(errors))

def read_edf(edf_file):
	hdl = EDFreader(edf_file)

	# print("\nStartdate: %02d-%02d-%04d" %(hdl.getStartDateDay(), hdl.getStartDateMonth(), hdl.getStartDateYear()))
	# print("Starttime: %02d:%02d:%02d" %(hdl.getStartTimeHour(), hdl.getStartTimeMinute(), hdl.getStartTimeSecond()))
	filetype = hdl.getFileType()
	if (filetype == hdl.EDFLIB_FILETYPE_EDF) or (filetype == hdl.EDFLIB_FILETYPE_BDF):
		subj_bytearr = hdl.getPatient()
		# print("Patient: %s" %(subj_bytearr))
		
		# Regular expressions to find sex and age
		sex_pattern = re.compile(rb'\b(M|F)\b')
		age_pattern = re.compile(rb'Age:(\d+)')

		# Search for sex and age
		sex_match = sex_pattern.search(subj_bytearr)
		age_match = age_pattern.search(subj_bytearr)

		# Extract the values
		sex = sex_match.group(1).decode("utf-8") if sex_match else None
		age = int(age_match.group(1).decode("utf-8")) if age_match else None
		
		assert age is not None and sex is not None

		return age, sex

if __name__ == '__main__':
	startTime = datetime.now()
	data_dir = '/Users/dbernardo/Documents/pyres/TUH/edf/'

	# leaving out 'train' for now
	for datatype in ['train']:
		subj_clinvars = []
		out_dir = f'data/{datatype}/'

		# Get EDF paths
		edf_paths = find_files(data_dir, '*.edf', ['normal', 'abnormal'], datatype)

		# Processes paths
		subjs = process_labels(edf_paths)

		for subj, edf_files in (pbar := tqdm(subjs.items())):
			pbar.set_description(f"Processing {subj}")

			subj_num = subj.split("_")[0]
			j = 0

			for edf_file in edf_files:

				age, sex = read_edf(edf_file)
				subj_clinvars.append([subj, edf_file, age, sex])

				sys.stdout.flush()

		subj_clinvars_df = pd.DataFrame(subj_clinvars, columns=['subject', 'edf_file', 'age', 'sex'])
		subj_clinvars_df.to_csv(f'{datatype}_clinvars.csv', index=False)

	print('\nTotal time taken:')
	print(datetime.now() - startTime)
