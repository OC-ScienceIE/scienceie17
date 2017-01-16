"""
Extension of simple CRF exp from crf1-cv-exp.py
with pruning
with word features
"""

from os.path import join

from sie.crf import PruneCRF
from sie import ENTITIES, LOCAL_DIR, EXPS_DIR
from sie.feats import generate_feats, features1
from sie.exp import run_exp_train_cv, eval_exp_train


def wordfeats1(sent):
    sent_feats = []

    for token in sent:
        l = len(token.orth_)
        token_feats = {
            'word': token.orth_,
            'shape': token.shape_,
            'is_alpha': token.is_alpha,
            'is_lower': token.is_lower,
            'is_ascii': token.is_ascii,
            'is_capitalized': token.orth_.istitle(),
            'is_upper': token.orth_.isupper(),
            'is_punct': token.is_punct,
            'like_num': token.like_num,
            'prefix2': token.orth_[:2] if l > 1 else '',
            'suffix2': token.orth_[-2:] if l > 1 else '',
            'prefix3': token.orth_[:3] if l > 2 else '',
            'suffix3': token.orth_[-3:] if l > 2 else '',
            'prefix4': token.orth_[:4] if l > 3 else '',
            'suffix4': token.orth_[-4:] if l > 3 else '',
            'char_size': l,
            'is_stop': token.is_stop,
            # This numerical feature has large range and causes weird behaviour...
            #'rank': token.rank
        }

        sent_feats.append(token_feats)

    return sent_feats


# Step 1: Generate features

spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
word_feats_dir = join('_train', 'wordfeats1')

generate_feats(spacy_dir, word_feats_dir, wordfeats1)


# Step 2: Run experiments

crf = PruneCRF(c1=0.1, c2=0.1, all_possible_transitions=True)

base_feats_dir = join(EXPS_DIR, 'crf1/_train/features1')
feat_dirs = [base_feats_dir, word_feats_dir]
preds = {}

for label in ENTITIES:
    preds[label] = run_exp_train_cv(crf, feat_dirs, label)


# Step 3: Evaluate

eval_exp_train(preds)
