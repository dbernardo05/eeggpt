import re
import os
import yaml
import numpy as np
import pandas as pd
import h5py
import sys

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
        for key, val in params.iteritems():
            prm[key] = eval(str(val))
    return eval(method)(**prm)


parser = OptionParser()

parser.add_option("-s", "--subject",
                  dest="subject", default=1,
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

(options, args) = parser.parse_args()

#arf_data_dir = '../data/'
arf_data_dir = '/Volumes/BOBO/__SPQR_data/'

subject = int(options.subject)
njobs = int(options.njobs)

# load yaml file
yml = yaml.load(open(options.config))

# output of the script
output = './arf/features/%s' % yml['output']
# create forlder if it does not exist
if not os.path.exists(output):
    os.makedirs(output)

# imports
for pkg, functions in yml['imports'].iteritems():
    stri = 'from ' + pkg + ' import ' + ','.join(functions)
    exec(stri)

# parse pipe function from parameters
pipe = []
for item in yml['preproc']:
    for method, params in item.iteritems():
        pipe.append(from_yaml_to_func(method, params))

# create pipeline
preproc = make_pipeline(*pipe)

# parse pipe function from parameters
if 'postproc' in yml.keys():
    pipe = []
    for item in yml['postproc']:
        for method, params in item.iteritems():
            pipe.append(from_yaml_to_func(method, params))

    # create pipeline
    postproc = make_pipeline(*pipe)


reg_arf = re.compile('.*/(\d*)_(\d*)_(\d)_(\d).h5')
reg = re.compile('.*/(\d*)_(\d*)_(\d).h5')

reg_test = re.compile('.*(new_%s_\d*.h5)' % str(subject).zfill(3))
#reg_old_test = re.compile('.*(%s_\d*.h5)' % str(subject).zfill(3))
reg_old_test = re.compile('.*(%s_\d*.h5)' % str(subject).zfill(3))

reg_arf_fname = re.compile('.*(%s_\d*_\d_\d.h5)' % str(subject).zfill(3))
reg_fname = re.compile('.*(%s_\d*_\d.h5)' % str(subject).zfill(3))


def process_data_train(fname, ii):
    if fname.split('/')[-1].count('_') == 3:
        fn_parts = reg_arf.findall(fname)[0]
        subj, indice, ann, label = fn_parts
        fn = reg_arf_fname.findall(fname)[0]
    else:
        subj, indice, label = reg.findall(fname)[0]
        fn = reg_fname.findall(fname)[0]
        ann = '0'

    subj_str= str(subj).zfill(3)
    indice_str = str(indice).zfill(3)
    # print fname, subj, indice, label
    # sys.exit()
    data_sequence = 0  # sequence used to be number 1 to 6 (I think, indicating order within hour of data)
    pbar.update(ii)
    # data = loadmat(fname, squeeze_me=True, struct_as_record=False,
    #                verify_compressed_data_integrity=False)['dataStruct']
    with h5py.File(fname, 'r') as hf:
        data = hf[subj_str + '_' + indice_str + '_' + ann + '_' + label][:]

    #print "\tave:", np.average(data), "\tmax", np.max(data), '\tmin:', np.min(data)
    #out = preproc.fit_transform(np.array([data.data.T]))
    out = preproc.fit_transform(np.array([data]))

    if len(out) == 1:
        out = out[0]
    val = np.sum(np.isnan(out)) == 0
    return out, val, int(label), int(ann), int(indice), (int(indice) - 1) / 6, data_sequence, fn


def process_data_test(fname, ii, reg_test=reg_test):
    # idx = reg_test.findall(fname)[0]
    subj, indice, ann, label = reg_test.findall(fname)[0]
    subj_str= str(subj).zfill(3)
    indice_str = str(indice).zfill(3)

    pbar.update(ii)
    # data = loadmat(fname, squeeze_me=True, struct_as_record=False,
    #                verify_compressed_data_integrity=False)['dataStruct']
    with h5py.File(fname, 'r') as hf:
        data = hf[subj_str + '_' + indice_str + '_' + ann + '_' + label][:]

    #out = preproc.fit_transform(np.array([data.data.T]))
    out = preproc.fit_transform(np.array([data]))

    if len(out) == 1:
        out = out[0]
    val = np.sum(np.isnan(out)) == 0

    return out, val, idx

base = arf_data_dir + 'data/%s/%s_' % (str(subject).zfill(3), str(subject).zfill(3))
fnames = (sorted(glob(base + '*_0.h5'),
                 key=lambda x: int(x.replace(base, '')[:-7])) +
          sorted(glob(base + '*_1.h5'),
                 key=lambda x: int(x.replace(base, '')[:-7])))

# ignore file not safe
# ignore = pd.read_csv('../csv_files/train_and_test_data_labels_safe.csv', index_col=0)

fnames_finals = []
for fname in fnames:
    ba = arf_data_dir + 'data/%s/' % str(subject).zfill(3)
    fn = fname.replace(ba, '')
    # if ignore.loc[fn, 'safe'] == 1:
    fnames_finals.append(fname)

fnames = fnames_finals

pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=len(fnames)).start()

res = Parallel(n_jobs=njobs)(delayed(process_data_train)(fname=fname, ii=ii)
                             for ii, fname in enumerate(fnames))

features, valid, y, ann, idx, clips, sequence, fnames = zip(*res)

features = np.array(features)
sequence = np.array(sequence)
idx = np.array(idx)
y = np.array(y)
clips = np.array(clips)
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

np.savez('%s/%s.npz' % (output, str(subject).zfill(3)), features=features_final,
         y=y, sequence=sequence, idx=idx, clips=clips, valid=valid,
         fnames=fnames)
# clear memory
res = []
features = []

print('Done Pre-Proc Training !!!')


sys.exit()



if options.old:
    base = '../data/test_%s/%s_' % (str(subject).zfill(3), str(subject).zfill(3))
    fnames = sorted(glob(base + '*.h5'),
                    key=lambda x: int(x.replace(base, '')[:-5]))

    #ignore = pd.read_csv('../csv_files/train_and_test_data_labels_safe.csv', index_col=0)

    fnames_finals = []
    for fname in fnames:
        ba = '../data/test_%s/' % str(subject).zfill(3)
        fn = fname.replace(ba, '')
        # if fn in ignore.index.values:
        fnames_finals.append(fname)

    fnames = fnames_finals

    pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()],
                       maxval=len(fnames)).start()

    res = Parallel(n_jobs=njobs)(delayed(process_data_test)(fname=fname, ii=ii, reg_test=reg)
                                 for ii, fname in enumerate(fnames))

    features, valid, idx = zip(*res)

    features = np.array(features)
    idx = np.array(idx)
    valid = np.array(valid)
    if 'postproc' in yml.keys():
        print("\npost process test data")
        features = postproc.transform(features[valid])
        out_shape = list(features.shape)
        out_shape[0] = len(valid)
        features_final = np.ones(tuple(out_shape)) * np.nan
        features_final[valid] = features
    else:
        features_final = features

    np.savez('%s/test%s.npz' % (output, str(subject).zfill(3)), features=features_final,
             fnames=idx, valid=valid)
    print('Done Old Test !!!')
    # clear memory
    res = []
    features = []



# Code below not needed
base = '../data/test_%s_new/new_%s_' % (str(subject).zfill(3), str(subject).zfill(3))
fnames = sorted(glob(base + '*.h5'),
                key=lambda x: int(x.replace(base, '')[:-5]))

pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()],
                   maxval=len(fnames)).start()

res = Parallel(n_jobs=njobs)(delayed(process_data_test)(fname=fname, ii=ii, reg_test=reg)
                             for ii, fname in enumerate(fnames))

features, valid, idx = zip(*res)

features = np.array(features)
idx = np.array(idx)
valid = np.array(valid)
if 'postproc' in yml.keys():
    print("\npost process test data")
    features = postproc.transform(features[valid])
    out_shape = list(features.shape)
    out_shape[0] = len(valid)
    features_final = np.ones(tuple(out_shape)) * np.nan
    features_final[valid] = features
else:
    features_final = features

np.savez('%s/new_test%d.npz' % (output, subject), features=features_final,
         fnames=idx, valid=valid)
print('Done New Test!!!')
