"""
Extension of simple CRF exp from crf1-cv-exp.py
with pruning
with word features
"""

from os.path import join

from sie.crf import PruneCRF
from sie import ENTITIES, LOCAL_DIR, EXPS_DIR
from sie.feats import generate_feats, word_feats
from sie.exp import run_exp_train_cv, eval_exp_train


# Step 1: Generate features

spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
word_feats_dir = join('_train', 'wordfeats1')

generate_feats(spacy_dir, word_feats_dir, lambda sent: word_feats(sent, context_size=1))


# Step 2: Run experiments

crf = PruneCRF()#c1=0.1, c2=0.1, all_possible_transitions=True)

base_feats_dir = join(EXPS_DIR, 'crf1/_train/features1')
feat_dirs = [base_feats_dir, word_feats_dir]
preds = {}

for label in ENTITIES:
    preds[label] = run_exp_train_cv(crf, feat_dirs, label)


# Step 3: Evaluate

eval_exp_train(preds)
