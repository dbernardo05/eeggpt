# generate_submission.py
# version 0.1 - working version with simple train/test split
# version 0.2 - incorporate per subject basis kfold split


import os
import cPickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import xgboost as xgb
import xxhash
import yaml

from copy import deepcopy
from glob import glob
from progressbar import Bar, ETA, Percentage, ProgressBar
from joblib import Parallel, delayed
from optparse import OptionParser
from time import time
from scipy import interp
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.pipeline import make_pipeline

np.random.seed(10)

def from_yaml_to_func(method, params):
    """Convert yaml to function"""
    prm = dict()
    if params is not None:
        for key, val in params.iteritems():
            prm[key] = eval(str(val))
    return eval(method)(**prm)

parser = OptionParser()

parser.add_option("-c", "--config",
                  dest="config", default="config.yml",
                  help="The config file")
parser.add_option("-p", "--preds",
                  dest="preds_only", default=False, action="store_true",
                  help="Compute only predictions")
parser.add_option("-v", "--validation",
                  dest="val_only", default=False, action="store_true",
                  help="Compute only validation")
parser.add_option("-l", "--loo",
                  dest="loo_split", default=False, action="store_true",
                  help="Perform Leave One Out cross validation instead of stratifiedKfold")

(options, args) = parser.parse_args()

spqr_data_dir = '/Volumes/BOBO/__SPQR_data/'

# global options
protomode = True
preds_mode = 'pool'

# load yaml file
yml = yaml.load(open(options.config))
modelname = str(options.config).split("/")[-1].split('.')[0]

# imports
for pkg, functions in yml['imports'].iteritems():
    stri = 'from ' + pkg + ' import ' + ','.join(functions)
    exec(stri)

# parse pipe function from parameters
pipe = []
for item in yml['model']:
    for method, params in item.iteritems():
        pipe.append(from_yaml_to_func(method, params))

# create pipeline
model = make_pipeline(*pipe)

datasets = yml['datasets']
n_jobs = yml['n_jobs']
safe_old = yml['safe_old']

if 'ignore_na' in yml.keys():
    ignore_na = yml['ignore_na']
    if ignore_na == False:
        print 'Error: ignore_na set to False'
        sys.exit()
else:
    ignore_na = True

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]


def classify(split, ii, model, features, features_test):
    """classify one split"""
    # print 'split', split
    # print 'ii', ii
    # print 'model', model

    # clean test features
    numSlices = features.shape[1]
    #print '\nfeatures shape:', features.shape

    # train fold
    feat_tr_fold = features
    print 'feat_tr_fold shape 1:', feat_tr_fold.shape
    feat_tr_fold = np.concatenate(feat_tr_fold, 0)
    print 'feat_tr_fold shape 2:', feat_tr_fold.shape

    if protomode:
        #feat_tr_fold = feat_tr_fold.reshape((feat_tr_fold.shape[0], np.prod(feat_tr_fold.shape[1:])))
        d1, d2 = (feat_tr_fold.shape[0], feat_tr_fold.shape[1:])
        rs_dim = [d1, np.prod(d2)]
        feat_tr_fold = feat_tr_fold.reshape(rs_dim)
        print 'feat_tr_fold shape 3:', feat_tr_fold.shape

    y_tr_fold = split['labels_train']
    y_tr_fold = y_tr_fold.repeat(numSlices)
    #print 'y_tr_fold shape:', y_tr_fold.shape

    fnames_tr_fold = split['fnames_train']
    fnames_tr_fold = fnames_tr_fold.repeat(numSlices)
    index_tr_fold = np.arange(len(y_tr_fold))

    # test fold
    feat_te_fold = features_test
    feat_te_fold = np.concatenate(feat_te_fold, 0)
    if protomode:
        #feat_te_fold = feat_te_fold.reshape((feat_te_fold.shape[0], np.prod(feat_te_fold.shape[1:])))
        d1, d2 = (feat_te_fold.shape[0], feat_te_fold.shape[1:])
        rs_dim = [d1, np.prod(d2)]
        feat_te_fold = feat_te_fold.reshape(rs_dim)
        # feat_te_fold = feat_te_fold.reshape((feat_te_fold.shape[0] * feat_te_fold.shape[1], feat_te_fold.shape[-1]))

    print 'feat_te_fold.shape', feat_te_fold.shape

    fnames_te_fold = split['fnames_test']
    fnames_te_fold = fnames_te_fold.repeat(numSlices)

    y_te_fold = split['labels_test']
    y_te_fold = y_te_fold.repeat(numSlices)
    #print 'y_te_fold shape:', y_te_fold.shape

    index_te_fold = np.arange(len(y_te_fold))
    #print 'index_te_fold shape:', index_te_fold.shape

    cv_fold1 = pd.DataFrame(0, index=index_te_fold, columns=['Preds'])
    cv_fold1['CV'] = ii
    cv_fold1['Labels'] = y_te_fold
    cv_fold1['File'] = fnames_te_fold

    # cv_fold2 = pd.DataFrame(0, index=index_tr_fold, columns=['Preds'])
    # cv_fold2['CV'] = ii
    # cv_fold2['Labels'] = y_tr_fold
    # cv_fold2['File'] = fnames_tr_fold

    if ignore_na:
        ix_good = np.array([np.sum(np.isnan(f)) == 0 for f in feat_tr_fold])
        feat_tr_fold = feat_tr_fold[ix_good]
        y_tr_fold = y_tr_fold[ix_good]
        fnames_tr_fold = fnames_tr_fold[ix_good]
        index_tr_fold = index_tr_fold[ix_good]

        ix_good = np.array([np.sum(np.isnan(f)) == 0 for f in feat_te_fold])
        y_te_fold = y_te_fold[ix_good]
        feat_te_fold = feat_te_fold[ix_good]
        fnames_te_fold = fnames_te_fold[ix_good]
        index_te_fold = index_te_fold[ix_good]
    
    print 'feat_tr_fold shape final:', feat_tr_fold.shape
    print 'feat_te_fold shape final:', feat_te_fold.shape


    if not options.preds_only:
        if protomode:
            #clf = xgb.XGBClassifier()
            #clf = svm.SVC(gamma=0.001, C=100.)
            clf = RandomForestClassifier(n_estimators=10)
        else:
            clf = deepcopy(model)

        clf.fit(feat_tr_fold, y_tr_fold)
        qq = clf.predict_proba(feat_te_fold)
        cv_fold1.loc[index_te_fold, 'Preds'] = qq[:, 1]

        if getattr(clf, 'predict_proba', None):
            cv_fold1.loc[index_te_fold, 'Preds'] = clf.predict_proba(feat_te_fold)[:, 1]
        else:
            cv_fold1.loc[index_te_fold, 'Preds'] = clf.predict(feat_te_fold)
        pbar.update(1)

        # Previous code for testing Train/Test splits
        # if protomode:
        #     #clf = xgb.XGBClassifier()
        #     clf = svm.SVC(gamma=0.001, C=100.)
        #     #clf = RandomForestClassifier(n_estimators=10)
        # else:
        #     clf = deepcopy(model)
        # clf.fit(feat_te_fold, y_te_fold)
        # if getattr(clf, 'predict_proba', None):
        #     cv_fold2.loc[index_tr_fold, 'Preds'] = clf.predict_proba(feat_tr_fold)[:, 1]
        # else:
        #     cv_fold2.loc[index_tr_fold, 'Preds'] = clf.predict(feat_tr_fold)

        pbar.update(2)

    if not options.val_only:
        feat_tr = np.concatenate([feat_tr_fold, feat_te_fold])
        y_tr = np.concatenate([y_tr_fold, y_te_fold])

        del(feat_tr_fold)
        del(feat_te_fold)

        clf = deepcopy(model)
        clf.fit(feat_tr, y_tr)

    return cv_fold1, clf


def get_cv(train, test, gen_ix_mode=False):
    cv = {}
    for dtype, kf in zip(['train', 'test'], [train, test]):
        temp_fnames = []
        temp_lbls = []
        temp_idcs = []
        for subject in kf:
            base = spqr_data_dir + 'data/%s/%s_' % (str(subject).zfill(3), str(subject).zfill(3))
            fnames = sorted(glob(base + '*.h5'),
                            key=lambda x: int(x.replace(base, '')[:-7]))
            fnames_finals = []
            for fname in fnames:
                ba = spqr_data_dir + 'data/%s/' % (str(subject).zfill(3))
                fn = fname.replace(ba, '')
                fnames_finals.append(fn)
            temp_fnames.extend(fnames_finals)

            # load safe_idcs
            with open('arf/safe_idcs/safeidx_' + str(subject).zfill(3) + '.pkl', 'rb') as fp:
                clust_rank, safe_idcs = pkl.load(fp)

            # if gen_ix_mode:
            #     safe_idcs = gen_safe_idx(clust_rank, 2)

            for fname in fnames_finals:
                temp_lbls.append(int(fname.split('.')[0][-1]))
                temp_idcs.append(safe_idcs[fname])

        cv['fnames_%s' % dtype] = np.array(temp_fnames)
        cv['labels_%s' % dtype] = np.array(temp_lbls)
        cv['indices_%s' % dtype] = np.array(temp_idcs)

    #print cv
    return cv



def load_features(split, cv, dtype):
    features = []
    for subject in subjects[split]:
        temp_features = []
        for jj, dataset in enumerate(datasets):
            a = np.load('./features/%s/%s.npz' % (dataset, str(subject).zfill(3)))
            temp_features.append(a['features'])
        temp_features = np.concatenate(temp_features, -1)

        fnames = cv['fnames_%s' % dtype]
        fn_idcs = []
        for fn, fname in enumerate(fnames):
            if str(subject).zfill(3) in fname.split('_')[0]:
                fn_idcs.append(fn)
        assert fn_idcs == sorted(fn_idcs)
        safe_idcs = cv['indices_%s' % dtype][fn_idcs]
        art_idcs = np.logical_not(safe_idcs)
        temp_features[art_idcs, :] = np.nan
        #print 'temp_features.shape:', temp_features.shape

        features.append(temp_features)
    del(a)

    return features


tot_preds = []
tot_cv_fold1 = []
tot_cv_fold2 = []
t_init = time()

# plt.figure(1)
# plt.plot([0, 1], [0, 1], 'k--')

roc_data = []
kf_tprs = []
kf_fprs = []
base_fpr = np.linspace(0, 1, 101)

subjects = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
#subjects = np.array([1, 2, 3, 4, 7, 8, 9, 10])

if options.loo_split:
    loo = LeaveOneOut()
    splitter = loo.split(subjects)
else:
    kf_lbls = np.array([1, 1, 0, 0, 1, 1, 1, 1, 0, 0 ])
    skf = StratifiedKFold(n_splits=4)
    splitter = skf.split(subjects, kf_lbls)

kf_num = 0


for train, test in splitter:

    cv = get_cv(subjects[train], subjects[test], True)
    pbar = ProgressBar(widgets=[Percentage(), Bar(), ETA()], maxval=3).start()

    features = load_features(train, cv, 'train')
    features_test = load_features(test, cv, 'test')

    # print 'len features:', len(features)
    # for n, feat in enumerate(features):
    #     print 'features shape:', feat.shape

    # print 'len features_test:', len(features_test)
    # for n, feat in enumerate(features_test):
    #     print 'features shape:', feat.shape

    # features = np.concatenate(features, -1)
    # features_test = np.concatenate(features_test, -1)
    features = np.vstack(features)
    features_test = np.vstack(features_test)

    # print 'features shape:', features.shape
    # print 'features_test shape:', features_test.shape

    cv_fold1, clf = classify(cv, 0, model, features, features_test)

    # average prediction accross CV splits
    del(features)
    del(features_test)

    if preds_mode == 'pool':
        cv_fold1_preds = cv_fold1.groupby('File')['Preds'].mean()
        cv_fold1_preds = np.array([float(x) for x in cv_fold1_preds[:]])
        cv_fold1_preds_final = cv_fold1_preds>0.5
        cv_fold1_labels = cv_fold1.groupby('File')['Labels'].unique()
        cv_fold1_labels = [x[0] for x in cv_fold1_labels[:]]
    else:
        cv_fold1_preds = cv_fold1['Preds']
        cv_fold1_preds_final = cv_fold1_preds>0.5
        cv_fold1_labels = cv_fold1['Labels']

    assert len(cv_fold1_labels) == len(cv_fold1_preds)

    target_names = ['Nonresponder', 'Responder']
    print('Cross-validation, kfold:', kf_num)
    print(classification_report(cv_fold1_labels, cv_fold1_preds_final, target_names=target_names))

    fpr, tpr, _ = roc_curve(cv_fold1_labels, cv_fold1_preds)
    # plt.plot(fpr, tpr, label=modelname+'-kf' + str(kf_num))
    roc_data.append([fpr,tpr,modelname+'-kf' + str(kf_num)])

    tpr = interp(base_fpr, fpr, tpr)
    tpr[0] = 0.0

    kf_tprs.append(tpr)

    kf_num += 1

kf_tprs = np.array(kf_tprs)

mean_tprs = kf_tprs.mean(axis=0)

# plt.xlabel('False positive rate')
# plt.ylabel('True positive rate')
# plt.title('ROC curve')
# plt.legend(loc='best')
# plt.savefig(os.path.join('metrics', '20180323_test.png'))

# Plot average ROC
# tprs = np.array(tprs)
# mean_tprs = tprs.mean(axis=0)
# std = tprs.std(axis=0)
# tprs_upper = np.minimum(mean_tprs + std, 1)
# tprs_lower = mean_tprs - std

# plt.figure(2)
# plt.plot(base_fpr, mean_tprs, 'b')
# plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([-0.01, 1.01])
# plt.ylim([-0.01, 1.01])
# plt.ylabel('True Positive Rate')
# plt.savefig(os.path.join('metrics', '20180323_avROC_test.png'))

roc_summary = {'individual': roc_data, 'average': [mean_tprs, base_fpr, modelname]}
with open(os.path.join('metrics', 'roc_data_' + modelname + '.pkl'), 'wb') as handle:
    pkl.dump(roc_summary, handle, protocol=pkl.HIGHEST_PROTOCOL)


if not options.val_only:
    preds = pd.concat(tot_preds).groupby('File').max()
    output = '../submissions/%s.csv' % yml['output']
    preds.to_csv(output)

print '\n\ntime taken:', time() - t_init, '\n\n'
