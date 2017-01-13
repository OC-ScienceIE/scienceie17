"""
Simple CRF exp with basic features (lemma & POS in a 5-token window),
training on train and dev data combined and testing on test data.
"""

from os.path import join

from sklearn_crfsuite import CRF

from sie import ENTITIES, LOCAL_DIR
from sie.feats import generate_feats, features1
from sie.exp import run_exp_test, eval_exp_train

# Step 1: Generate features

train_spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
train_base_feats_dir = join('_train', 'features1')

# If you want to save time by reusing existing feats, comment out the line below:
generate_feats(train_spacy_dir, train_base_feats_dir, features1)


dev_spacy_dir = join(LOCAL_DIR, 'dev', 'spacy')
dev_base_feats_dir = join('_dev', 'features1')

# If you want to save time by reusing existing feats, comment out the line below:
generate_feats(dev_spacy_dir, dev_base_feats_dir, features1)


test_spacy_dir = join(LOCAL_DIR, 'test', 'spacy')
test_base_feats_dir = join('_test', 'features1')

generate_feats(test_spacy_dir, test_base_feats_dir, features1)


# Step 2: Run experiments

crf = CRF(c1=0.1, c2=0.1, all_possible_transitions=True)
train_feat_dirs = [train_base_feats_dir]
dev_feat_dirs = [dev_base_feats_dir]
test_feat_dirs = [test_base_feats_dir]
preds = {}

for label in ENTITIES:
    preds[label] = run_exp_test(crf, train_feat_dirs, dev_feat_dirs, test_feat_dirs, label)


# Step 3: Evaluate

# Even though test data is unlabeled, but generates teh Brat files to submit
eval_exp_train(preds, 'test')
