"""
prune sentences without annotation from training

with basic features, word features and synvec features
"""


from os.path import join

from sklearn_crfsuite.metrics import flat_classification_report

from sklearn.model_selection import cross_val_predict, GroupKFold

from sie import ENTITIES, EXPS_DIR, LOCAL_DIR, DATA_DIR
from sie.crf import PruneCRF, collect_crf_data, pred_to_iob
from sie.brat import iob_to_brat

from eval import calculateMeasures


# Step 1: Reuse features

base_feats_dir = join(EXPS_DIR, 'crf1/_train/features1')
word_feats_dir = join('_train', join(EXPS_DIR, 'wordfeats/_train/wordfeats1'))
synvec_feats_dir = join('_train', join(EXPS_DIR, 'synvecs/_train/synvec_feats'))


# Step 2: Collect data for running CRF classifier

true_iob_dir = join(LOCAL_DIR, 'train', 'iob')

data = collect_crf_data(true_iob_dir, base_feats_dir, word_feats_dir, synvec_feats_dir)

# Step 3: Create folds

# create folds from complete texts only (i.e. instances of the same text
# are never in different folds)
# TODO How to set seed for random generator?
group_k_fold = GroupKFold(n_splits=5)

# use same split for all three entities
splits = list(
    group_k_fold.split(data['feats'], data['Material'], data['filenames']))

# Step 4: Run CRF classifier
crf = PruneCRF(c1=0.1, c2=0.1, all_possible_transitions=True)
pred = {}

for ent in ENTITIES:
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

