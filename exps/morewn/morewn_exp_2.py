"""
wordnet features, hypernyms only, without synsets of current token
"""

from os.path import join

from sie import ENTITIES, LOCAL_DIR, EXPS_DIR
from sie.crf import PruneCRF
from sie.exp import run_exp_train_cv, eval_exp_train
from sie.feats import generate_feats, wordnet_feats


# Step 1: Generate features

base_feats_dir = join(EXPS_DIR, 'crf1/_train/features1')
word_feats_dir = join(EXPS_DIR, 'wordfeats/_train/wordfeats1')

spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
wn_feats_dir = join('_train', 'wnfeats1')
generate_feats(spacy_dir, wn_feats_dir, wordnet_feats)


# Step 2: Run experiments

crf = PruneCRF()#c1=0.1, c2=0.1, min_freq=5)#, all_possible_transitions=True)
feat_dirs = [base_feats_dir, wn_feats_dir, word_feats_dir]
preds = {}

for label in ENTITIES[:1]:
    preds[label] = run_exp_train_cv(crf, feat_dirs, label, n_folds=5, n_jobs=-1)


# Step 3: Evaluate

eval_exp_train(preds)
