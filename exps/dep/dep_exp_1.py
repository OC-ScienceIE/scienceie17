"""
Extension of simple CRF exp from crf1-cv-exp.py
with pruning
with dependency features
"""

from os.path import join

from sie import ENTITIES, LOCAL_DIR, EXPS_DIR
from sie.crf import PruneCRF
from sie.exp import run_exp_train_cv, eval_exp_train
from sie.feats import generate_feats, dep_feats


# Step 1: Generate features


base_feats_dir = join(EXPS_DIR, 'best/_train/Material/lempos_feats')
word_feats_dir = join(EXPS_DIR, 'best/_train/Material/word_feats')
wordnet_feats_dir = join(EXPS_DIR, 'best/_train/Material/wordnet_feats')

spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
dep_feats_dir = join('_train', 'dep_feats')
generate_feats(spacy_dir, dep_feats_dir, lambda sent: dep_feats(sent, context_size=1))

# Step 2: Run experiments

crf = PruneCRF()  # c1=0.1, c2=0.1, all_possible_transitions=True)

feat_dirs = [
    #base_feats_dir,
    #word_feats_dir,
    #wordnet_feats_dir,
    dep_feats_dir
]

preds = {}

for label in ENTITIES:
    preds[label] = run_exp_train_cv(crf, feat_dirs, label)

# Step 3: Evaluate

eval_exp_train(preds)
