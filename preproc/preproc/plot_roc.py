

import cPickle as pkl
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from glob import glob
from sklearn import metrics

roc_datas = glob('metrics/*.pkl')

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')

linestyles = ['-', '--', '-.', ':']

for roc_fname in roc_datas:

	with open(roc_fname, 'rb') as handle:
	    roc_data = pkl.load(handle)

	for fpr, tpr, modelname in roc_data['individual']:
		kf = int(modelname.split('-')[-1][-1])
		auc = metrics.auc(fpr, tpr)
		plt.plot(fpr, tpr, label='%s (AUC = %.2f)' % (modelname, auc) , linestyle=linestyles[kf], alpha=0.6)

plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, fontsize=7)

plt.savefig(os.path.join('metrics', '20180401_all.png'), dpi=300, bbox_inches="tight")
plt.clf()


tprs = []
base_fprs = []
plt.figure(2)
for roc_fname in roc_datas:

	with open(roc_fname, 'rb') as handle:
	    roc_data = pkl.load(handle)

	print len(roc_data['average'])
	mean_tprs, base_fpr, modelname = roc_data['average']
	# for tpr, fpr in zip(mean_tprs, base_fpr):
	# kf = int(modelname.split('-')[-1][-1])
	auc = metrics.auc(base_fpr, mean_tprs)
	plt.plot(base_fpr, mean_tprs, alpha=0.2, label='%s (AUC = %.2f)' % (modelname, auc) )
	
	tprs.append(mean_tprs)
	base_fprs.append(base_fpr)

tprs = np.array(tprs)

mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)

tprs_upper = np.minimum(mean_tprs + std, 1)
tprs_lower = mean_tprs - std

plt.figure(2)
av_auc = metrics.auc(base_fpr, mean_tprs)
plt.plot(base_fpr, mean_tprs, 'b', linewidth=2.0)
plt.fill_between(base_fpr, tprs_lower, tprs_upper, color='grey', alpha=0.3)

plt.plot([0, 1], [0, 1],'r--')
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.ylabel('True Positive Rate')
plt.title('Average ROC curve (AUC = %.2f)' % (av_auc))
plt.legend(bbox_to_anchor=(1.04,0), loc="lower left", borderaxespad=0, fontsize=7)

plt.savefig(os.path.join('metrics', '20180401_avROC_all.png'), dpi=300, bbox_inches="tight")

plt.clf()
