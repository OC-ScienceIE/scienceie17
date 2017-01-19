"""
Best settings on train with cv
"""

import sys
import os

sys.path.append(os.getcwd())

from sie.crf import PruneCRF
from sie.exp import run_exp_train_cv, eval_exp_train

from best_feats import make_feats

preds = {}

# ----------------------------------------------------------------------------
# Material
# ----------------------------------------------------------------------------

label = 'Material'
crf = PruneCRF()
feat_dirs = make_feats('train', label)
preds[label] = run_exp_train_cv(crf, feat_dirs, label, n_folds=5, n_jobs=-1)

# ----------------------------------------------------------------------------
# Process
# ----------------------------------------------------------------------------

label = 'Process'
crf = PruneCRF()
feat_dirs = make_feats('train', label)
preds[label] = run_exp_train_cv(crf, feat_dirs, label, n_folds=5, n_jobs=-1)

# ----------------------------------------------------------------------------
# Task
# ----------------------------------------------------------------------------

label = 'Task'
crf = PruneCRF()
feat_dirs = make_feats('train', label)
preds[label] = run_exp_train_cv(crf, feat_dirs, label, n_folds=5, n_jobs=-1)

# ----------------------------------------------------------------------------
# Evaluate
# ----------------------------------------------------------------------------

eval_exp_train(preds)  # , postproc=postproc_labels)
