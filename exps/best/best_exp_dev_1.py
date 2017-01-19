"""
Best settings on dev
"""

import sys
import os

sys.path.append(os.getcwd())

from sie.crf import PruneCRF
from sie.exp import run_exp_dev, eval_exp_train

from best_feats import make_feats

preds = {}

# ----------------------------------------------------------------------------
# Material
# ----------------------------------------------------------------------------

label = 'Material'
crf = PruneCRF()
train_feat_dirs = make_feats('train', label)
dev_feat_dirs = make_feats('dev', label)
preds[label] = run_exp_dev(crf, train_feat_dirs, dev_feat_dirs, label)

# ----------------------------------------------------------------------------
# Process
# ----------------------------------------------------------------------------

label = 'Process'
crf = PruneCRF()
train_feat_dirs = make_feats('train', label)
dev_feat_dirs = make_feats('dev', label)
preds[label] = run_exp_dev(crf, train_feat_dirs, dev_feat_dirs, label)

# ----------------------------------------------------------------------------
# Task
# ----------------------------------------------------------------------------

label = 'Task'
crf = PruneCRF()
train_feat_dirs = make_feats('train', label)
dev_feat_dirs = make_feats('dev', label)
preds[label] = run_exp_dev(crf, train_feat_dirs, dev_feat_dirs, label)

# ----------------------------------------------------------------------------
# Evaluate
# ----------------------------------------------------------------------------

eval_exp_train(preds, 'dev')  # , postproc=postproc_labels)
