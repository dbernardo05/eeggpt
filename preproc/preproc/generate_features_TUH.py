import re
import os
import yaml
import numpy as np
import pandas as pd
import h5py
import sys
import tqdm

#from scipy.io import loadmat
from glob import glob

from progressbar import Bar, ETA, Percentage, ProgressBar
from joblib import Parallel, delayed
from optparse import OptionParser

from sklearn.pipeline import make_pipeline


def from_yaml_to_func(method, params):
	"""Convert yaml to function"""
	prm = dict()
	if params is not None:
		for key, val in params.items():
			prm[key] = eval(str(val))
	return eval(method)(**prm)

def process_data(fname, ii, reg, preproc):
	subj, label = reg.findall(fname)[0]
	subj_str= str(subj)

	# print fname, subj, indice, label
	# sys.exit()

	# data = loadmat(fname, squeeze_me=True, struct_as_record=False,
	#                verify_compressed_data_integrity=False)['dataStruct']
	with h5py.File(fname, 'r') as hf:
		data = hf[f'{subj_str}_{label}'][:]

	# print(data.shape)
	if data.shape[1] <= 20 * 200.:
		return np.nan, np.nan, np.nan, subj

	#print "\tave:", np.average(data), "\tmax", np.max(data), '\tmin:', np.min(data)
	#out = preproc.fit_transform(np.array([data.data.T]))
	out = preproc.fit_transform(np.array([data]))

	if len(out) == 1:
		out = out[0]
	val = np.sum(np.isnan(out)) == 0
	return out, val, int(label), subj


def gen_subject_features(subject, yml, preproc, output, inner_parallel_mode):
	
	subject_root = subject.split('/')[-1]

	# parse pipe function from parameters
	if 'postproc' in yml.keys():
		pipe = []
		for item in yml['postproc']:
			for method, params in item.items():
				pipe.append(from_yaml_to_func(method, params))

		# create pipeline
		postproc = make_pipeline(*pipe)

	# reg = re.compile('.*/(\d*)_(\d*)\.h5') 
	reg = re.compile(r'.*/([a-zA-Z]*s\d{3}XXXt\d{3})_(\d)\.h5')

	fnames = (sorted(glob(subject + '/*_0.h5'),
					 key=lambda x: x.split('/')[-1]) +
			  sorted(glob(subject + '/*_1.h5'),
					 key=lambda x: x.split('/')[-1]))


	debug_mode = False
	if debug_mode:
		for ii, fname in enumerate(fnames):
			print('#', fname)
			process_data(fname, ii, reg, preproc)


	if inner_parallel_mode:
		# There is no benefit to using inner parallel with non-chunk mode b/c only one chunk
		res = Parallel(n_jobs=njobs)(delayed(process_data)(fname=fname, ii=ii, reg=reg, preproc=preproc)
									 for ii, fname in enumerate(tqdm.tqdm(fnames)))
		features, valid, y, fnames = zip(*res)
	else:
		assert len(fnames) == 1
		fname = fnames[0]
		features, valid, y, fnames = process_data(fname=fname, ii=0, reg=reg, preproc=preproc)

	features = np.array(features)
	y = np.array(y)
	valid = np.array(valid)
	fnames = np.array(fnames)

	if 'postproc' in yml.keys():
		print("\npost process training data")
		features = postproc.fit_transform(features[valid], y[valid])
		out_shape = list(features.shape)
		out_shape[0] = len(valid)
		features_final = np.ones(tuple(out_shape)) * np.nan
		features_final[valid] = features
	else:
		features_final = features

	np.savez('%s/%s.npz' % (output, subject_root), features=features_final,
			 y=y, valid=valid,
			 fnames=fnames)
	# clear memory
	res = []
	features = []



if __name__ == "__main__":
	parser = OptionParser()
	parser.add_option("-s", "--subject",
					  dest="subject", default=None,
					  help="The subject")
	parser.add_option("-c", "--config",
					  dest="config", default="config.yml",
					  help="The config file")
	parser.add_option("-o", "--old",
					  dest="old", default=False, action="store_true",
					  help="process the old test set")
	parser.add_option("-n", "--njobs",
					  dest="njobs", default=1,
					  help="the number of jobs")
	parser.add_option("-m", "--multimode",
					  dest="multimode", default=0,
					  help="Enable multiple subject mode")
	parser.add_option("-r", "--serialmode",
					  dest="serialmode", default=0,
					  help="Enable serial single subject mode")
	parser.add_option("-d", "--datatype",
					  dest="datatype", default='eval',
					  help="Choose eval or train")
	(options, args) = parser.parse_args()

	spqr_data_dir = '/Users/dbernardo/Documents/pyres/eeggpt/preproc/data/'

	njobs = int(options.njobs)

	# load yaml file
	yml = yaml.load(open(options.config), Loader=yaml.Loader)

	# imports
	for pkg, functions in yml['imports'].items():
		stri = 'from ' + pkg + ' import ' + ','.join(functions)
		exec(stri)

	print(yml)
	# parse pipe function from parameters
	pipe = []
	for item in yml['preproc']:
		for method, params in item.items():
			from_yaml_to_func(method, params)
			pipe.append(from_yaml_to_func(method, params))

	# create pipeline
	preproc = make_pipeline(*pipe)

	# output of the script
	output = f"./features_{options.datatype}/{yml['output']}"
	# create forlder if it does not exist
	if not os.path.exists(output):
		os.makedirs(output)
		

	if options.multimode:
		fnames = sorted(glob(f'{spqr_data_dir}{options.datatype}/*'))
		res = Parallel(n_jobs=njobs)(delayed(gen_subject_features)(subject=subject, yml=yml, preproc=preproc, output=output, inner_parallel_mode=False)
									 for ii, subject in enumerate(tqdm.tqdm(fnames)))

	elif options.serialmode:
		fnames = sorted(glob(f'{spqr_data_dir}{options.datatype}/*'))
		for ii, fname in enumerate(tqdm.tqdm(fnames)):
			print(ii, fname)
			gen_subject_features(fname, yml, preproc, output, inner_parallel_mode=False)
	else:
		# Old usage
		subject = str(options.subject)

		gen_subject_features(subject, yml, preproc, output, inner_parallel_mode=True)

