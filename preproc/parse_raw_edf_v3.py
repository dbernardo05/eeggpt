###
# parse_raw_edf.py
# segments data from EDFs for use in SPQR
# v1 - annotations added
# v2 - adding flexibility for differently named EDFs (Never Responders)
# v3 - for NMT dataset


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
import mne
import numpy as np
import os
import pandas as pd
import random
import sys

from datetime import datetime
from tqdm import tqdm

def get_srate(raw):
	data, times = raw[:]  
	srate = 1.0 / abs(times[1] - times[0])
	final_srate = 1.0 / abs(times[-1] - times[-2])
	assert abs(final_srate - srate) < 0.001
	return srate

def find_files(search_dir, ext, subdirs, datatype):
	matches = []
	ignored = []

	for subdir in subdirs:
		search_subdir = os.path.join(search_dir, subdir, datatype)
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

def check_sr(edf_file):
	raw_edf = mne.io.read_raw_edf(edf_file, preload=True, verbose='ERROR')
	return get_srate(raw_edf)

def check_eeg(raw_edf):
	errors = []

	# replace assertions by conditions
	if len(raw_edf.ch_names) != 21:
		errors.append(f"Missing:{list(set(ref_chans) - set(raw_edf.ch_names))}")
	if int(raw_edf.info['sfreq']) != 200:
		errors.append("SR != 200")
	if raw_edf.info['sfreq'] != get_srate(raw_edf):
		errors.append(f"Sampling freq mismatch (reported/calculated): {raw_edf.info['sfreq'], get_srate(raw_edf)}")

	# assert no error message has been registered, else print messages
	assert not errors, "errors occured:\n{}".format("\n".join(errors))

def load_edf(edf_file):
	raw_edf = mne.io.read_raw_edf(edf_file, preload=True, verbose='ERROR')
	
	# fix chans
	new_ch_map = {}
	for ch in raw_edf.ch_names:
		n_ch = ch.replace('FP1', 'Fp1').replace('FP2','Fp2').replace('Z','z').replace(' ', '')
		new_ch_map[ch] = n_ch
	raw_edf.rename_channels(new_ch_map)

	ref_chans = ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 
		'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz', 'A1', 'A2']
	raw_edf.pick_channels(ref_chans)

	check_eeg(raw_edf)

	hp01 = get_hp_data(raw_edf, 1.)

	if np.any(np.isnan(hp01.get_data())):
		print('############# ERROR NAN WHEN FILTERING')
		sys.exit()

	# print('\t\tLength data (min):', (hp01.get_data().shape[1]/200.0)/60.0)

	raw_ipsiear_ref, _ = mne.set_eeg_reference(hp01, ref_channels=['A1', 'A2'], verbose='warning')
	if np.any(np.isnan(raw_ipsiear_ref.get_data())):
		print('############# ERROR NAN WHEN MONTAGING')
		sys.exit()

	return raw_ipsiear_ref.get_data(), [None]

def get_hp_data(eeg_data, lf):
	# raw.notch_filter(np.arange(60, 361, 60), picks=picks, filter_length=2048*20, phase='zero')
	eeg_data.filter(lf, None, fir_design='firwin', verbose='warning')
	# curr_hp1_data.plot_psd(area_mode='range', tmax=10.0, average=False)
	# eeg_data = eeg_data.get_data()
	return eeg_data

def get_anns(eeg_data):
	pre_annotations = mne.events_from_annotations(eeg_data)
	anns = []
	for ann in pre_annotations:
		if '#' in ann[2]:
			anns.append([ann[0]*srate, ann[2]])
	return anns

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


def check_data(subjs):
	print('Checking Sampling Rates')
	sampling_rates = []
	for subj, edf_files in tqdm(subjs.items()):
		for edf_file in edf_files:
			sampling_rates.append(check_sr(edf_file))
	print(f'Unique sampling rates:{set(sampling_rates)}')
	assert len(set(sampling_rates)) == 1

	print('Checking Filters')
	filters = []
	for subj, edf_files in tqdm(subjs.items()):
		for edf_file in edf_files:
			raw_edf = mne.io.read_raw_edf(edf_file, preload=True, verbose='ERROR')
			filters.append((raw_edf.info['highpass'], raw_edf.info['lowpass']))
	print(f'Unique filters:{set(filters)}')
	assert len(set(filters)) == 1


if __name__ == '__main__':
	startTime = datetime.now()

	data_dir = 'nmt_scalp_eeg_dataset/'

	srate = 200
	winsize = 10 * 60  # 10 minute window
	chunk_eeg = False



	for datatype in ['train', 'eval']:

		out_dir = f'data/{datatype}/'

		# Get EDF paths
		edf_paths = find_files(data_dir, '*.edf', ['normal', 'abnormal'], datatype)

		# Processes paths
		subjs = process_labels(edf_paths)

		# Set Montage
		montage = mne.channels.make_standard_montage('standard_1020')

		if False:
			check_data(subjs)

		for subj, edf_files in (pbar := tqdm(subjs.items())):
			pbar.set_description(f"Processing {subj}")

			subj_num = subj.split("_")[0]
			j = 0

			for edf_file in edf_files:
				lbl = subj.split("_")[-1]

				hp01, anns = load_edf(edf_file)

				if chunk_eeg:
					# Mainly for long (24hr) EEGs
					num_epochs = int(np.floor(hp01.shape[1] / (srate*winsize)))
				else:
					num_epochs = 1

				for n in range(num_epochs):
					# print('\t\t\tCurrent epoch:', n, )

					if chunk_eeg:
						range_start = n*srate*winsize
						range_end = n*srate*winsize + winsize*srate
						eeg_dataseg = hp01[:, range_start: range_end]
					else:
						eeg_dataseg = hp01

					# Code for parsing annotations
					# epoch_ann = 0
					# ann_locs = np.where(np.logical_and((ann_idx>range_start), (ann_idx<range_end))==True)[0]
					# if np.any(ann_locs):
					# 	# print('\t\t\t\tAnns idx:', ann_locs)
					# 	print('\t\t\t\tAnns:', [anns[i] for i in ann_locs]) 
					# 	for ann in [anns[i] for i in ann_locs]:
					# 		if 'mvmt_art' in ann[1]:
					# 			epoch_ann = 1
					# 		elif 'spasms' in ann[1]:
					# 			epoch_ann = 2

					#eeg_dataseg = np.random.random((10,1000))

					# Edit dataset_name if needed
					dataset_name = subj 

					out_path = os.path.join(out_dir, dataset_name)
					if not os.path.exists(out_path):
						os.makedirs(out_path)

					h5_outpath = os.path.join(out_path, dataset_name + '.h5')
					with h5py.File(h5_outpath, 'w') as hf:
						hf.create_dataset(dataset_name,  data=eeg_dataseg)
					j+=1


				sys.stdout.flush()


	print('\nTotal time taken:')
	print(datetime.now() - startTime)
