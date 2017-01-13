"""
Simple CRF exp with basic features (lemma & POS in a 5-token window),
training on train data and testing on dev data.
"""

from os.path import join

from sklearn_crfsuite import CRF

from sie import ENTITIES, LOCAL_DIR
from sie.feats import generate_feats, features1
from sie.exp import run_exp_dev, eval_exp_train

# Step 1: Generate features

train_spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
train_base_feats_dir = join('_train', 'features1')

# If you want to save time by reusing existing feats, comment out the line below:
generate_feats(train_spacy_dir, train_base_feats_dir, features1)

dev_spacy_dir = join(LOCAL_DIR, 'dev', 'spacy')
dev_base_feats_dir = join('_dev', 'features1')

generate_feats(dev_spacy_dir, dev_base_feats_dir, features1)


# Step 2: Run experiments

crf = CRF(c1=0.1, c2=0.1, all_possible_transitions=True)
train_feat_dirs = [train_base_feats_dir]
dev_feat_dirs = [dev_base_feats_dir]
preds = {}

for label in ENTITIES:
    preds[label] = run_exp_dev(crf, train_feat_dirs, dev_feat_dirs, label)


# Step 3: Evaluate

eval_exp_train(preds, 'dev')
