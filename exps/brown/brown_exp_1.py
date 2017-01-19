"""
Brown features

This seems to be informative in combination with word features,
but doesn't add anything on top of word features + lempos feats
"""

from os.path import join

from sie import ENTITIES, LOCAL_DIR, EXPS_DIR
from sie.crf import PruneCRF
from sie.exp import run_exp_train_cv, eval_exp_train
from sie.feats import generate_feats, brown_feats

# Step 1: Generate features

spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
brown_feats_dir = join('_train', 'brown_feats')
base_feats_dir = join(EXPS_DIR, 'crf1/_train/features1')
word_feats_dir = join(EXPS_DIR, 'wordfeats/_train/wordfeats1')

generate_feats(spacy_dir, brown_feats_dir, brown_feats)


# Step 2: Run experiments

crf = PruneCRF()#c1=0.1, c2=0.1, all_possible_transitions=True)

base_feats_dir = join(EXPS_DIR, 'crf1/_train/features1')
feat_dirs = [base_feats_dir, brown_feats_dir, word_feats_dir]
preds = {}

for label in ENTITIES:
    preds[label] = run_exp_train_cv(crf, feat_dirs, label)


# Step 3: Evaluate

eval_exp_train(preds)
