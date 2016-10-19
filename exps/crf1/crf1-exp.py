"""
Simple CRF exp with basic features (lemma & POS in a 5-token window),
training & testing on the *same* data.
"""

from os.path import join

from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

from sie import ENTITIES, LOCAL_DIR
from sie.crf import collect_crf_data
from sie.feats import generate_feats, features1

# Step 1: Generate features

spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
feats_dir = join('_train', 'features1')

generate_feats(spacy_dir, feats_dir, features1)

# Step 2: Collect data for running CRF classifier

iob_dir = join(LOCAL_DIR, 'train', 'iob')
data = collect_crf_data(iob_dir, feats_dir)

# Step 3: Run CRF classifier
crf = CRF()

for ent in ENTITIES:
    crf.fit(data['feats'], data[ent])
    # score = crf.score(data['feats'], data[ent])
    # print(ent, score)
    pred = crf.predict(data['feats'])
    print('\n' + ent + ':\n')
    print(flat_classification_report(data[ent], pred, digits=3,
                                     labels=('B', 'I')))
