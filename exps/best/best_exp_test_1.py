"""
Best settings on test
"""

import sys
import os

sys.path.append(os.getcwd())

from sie.crf import PruneCRF
from sie.exp import run_exp_test, eval_exp_train
from sie.postproc import postproc_labels

from best_feats import make_feats

preds = {}

# ----------------------------------------------------------------------------
# Material
# ----------------------------------------------------------------------------

label = 'Material'
crf = PruneCRF()
train_feat_dirs = make_feats('train', label)
dev_feat_dirs = make_feats('dev', label)
test_feat_dirs = make_feats('test', label)
preds[label] = run_exp_test(crf, train_feat_dirs, dev_feat_dirs, test_feat_dirs, label)

# ----------------------------------------------------------------------------
# Process
# ----------------------------------------------------------------------------

label = 'Process'
crf = PruneCRF()
train_feat_dirs = make_feats('train', label)
dev_feat_dirs = make_feats('dev', label)
test_feat_dirs = make_feats('test', label)
preds[label] = run_exp_test(crf, train_feat_dirs, dev_feat_dirs, test_feat_dirs, label)

# ----------------------------------------------------------------------------
# Task
# ----------------------------------------------------------------------------

label = 'Task'
crf = PruneCRF()
train_feat_dirs = make_feats('train', label)
dev_feat_dirs = make_feats('dev', label)
test_feat_dirs = make_feats('test', label)
preds[label] = run_exp_test(crf, train_feat_dirs, dev_feat_dirs, test_feat_dirs, label)

# ----------------------------------------------------------------------------
# Evaluate
# ----------------------------------------------------------------------------

eval_exp_train(preds, 'test', postproc=postproc_labels, zip_fname='best_exp_test_1.zip')
