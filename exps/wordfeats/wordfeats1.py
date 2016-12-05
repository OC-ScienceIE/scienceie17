"""
Extension of simple CRF exp from crf1-cv-exp.py
with a number of word features
"""


from os.path import join

from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

from sklearn.model_selection import cross_val_predict, GroupKFold

from sie import ENTITIES, LOCAL_DIR, DATA_DIR, EXPS_DIR
from sie.crf import collect_crf_data, pred_to_iob
from sie.feats import generate_feats, features1
from sie.brat import iob_to_brat

from eval import calculateMeasures


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
            'char_size': l
        }

        sent_feats.append(token_feats)

    return sent_feats


# Step 1: Generate features

spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
word_feats_dir = join('_train', 'wordfeats1')

generate_feats(spacy_dir, word_feats_dir, wordfeats1)

# Step 2: Collect data for running CRF classifier

base_feats_dir = join(EXPS_DIR, 'crf1/_train/features1')
true_iob_dir = join(LOCAL_DIR, 'train', 'iob')

data = collect_crf_data(true_iob_dir, base_feats_dir, word_feats_dir)

# Step 3: Create folds

# create folds from complete texts only (i.e. instances of the same text
# are never in different folds)
# TODO How to set seed for random generator?
#group_k_fold = GroupKFold(n_splits=5)
group_k_fold = GroupKFold(n_splits=3)

# use same split for all three entities
splits = list(
    group_k_fold.split(data['feats'], data['Material'], data['filenames']))

# Step 4: Run CRF classifier
crf = CRF(c1=0.1, c2=0.1, all_possible_transitions=True)
pred = {}

for ent in ENTITIES:
    crf.fit(data['feats'], data[ent])
    pred[ent] = cross_val_predict(crf, data['feats'], data[ent], cv=splits)
    # Report scores directly on I and B tags,
    # disregard 'O' because it is by far the most frequent class
    print('\n' + ent + ':\n')
    print(flat_classification_report(data[ent], pred[ent], digits=3,
                                     labels=('B', 'I')))


# Step 5: Convert CRF prediction to IOB tags
pred_iob_dir = '_train/iob'

pred_to_iob(pred, data['filenames'], true_iob_dir, pred_iob_dir)

# Step 6: Convert predicted IOB tags to predicted Brat annotations
txt_dir = join(DATA_DIR, 'train')
brat_dir = '_train/brat'

iob_to_brat(pred_iob_dir, txt_dir, brat_dir)

# Step 7: Evaluate
calculateMeasures(txt_dir, brat_dir, 'rel')

