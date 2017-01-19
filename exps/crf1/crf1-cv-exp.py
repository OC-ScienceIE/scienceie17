"""
Simple CRF exp with basic features (lemma & POS in a 5-token window),
using 5-fold CV for evaluation
"""


from os.path import join

from sklearn_crfsuite import CRF

from sie import ENTITIES, LOCAL_DIR
from sie.feats import generate_feats, features1
from sie.exp import run_exp_train_cv, eval_exp_train


# Step 1: Generate features

spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
base_feats_dir = join('_train', 'features1')

# If you want to save time by reusing existing feats, comment out the line below:
generate_feats(spacy_dir, base_feats_dir, features1)


# Step 2: Run experiments

crf = CRF()#c1=0.1, c2=0.1, all_possible_transitions=True)
feat_dirs = [base_feats_dir]
preds = {}

for label in ENTITIES:
    preds[label] = run_exp_train_cv(crf, feat_dirs, label, n_folds=5)


# Step 3: Evaluate

eval_exp_train(preds)


