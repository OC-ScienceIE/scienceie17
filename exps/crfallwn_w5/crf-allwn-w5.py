"""
Simple CRF exp with basic features (lemma & POS in a 5-token window),
using 5-fold CV for evaluation
"""

from os.path import join

from sklearn_crfsuite import CRF
from sklearn_crfsuite.metrics import flat_classification_report

from sklearn.model_selection import cross_val_predict, GroupKFold

from sie import ENTITIES, LOCAL_DIR, DATA_DIR
from sie.crf import collect_crf_data, pred_to_iob
from sie.feats import generate_feats
from sie.brat import iob_to_brat

from eval import calculateMeasures

# Function to add WordNet features
from nltk.corpus import wordnet as wn
import re

def features2(sent, context_size=2, undefined='__'):
    """
    Example of a simple feature function which generates, for each token in
    the sentence, its lemma, POS and all Wordnet synsets of the lemma as well as that of the two
    preceding/following tokens (i.e. window size=5).
    
    Example:

    [
        {
            "-1:None": "true",
            "-1:lemma": "__",
            "-1:pos": "__",
            "-2:None": "true",
            "-2:lemma": "__",
            "-2:pos": "__",
            "0:Synset('hapless.s.01')": "true",
            "0:Synset('inadequate.s.02')": "true",
            "0:Synset('poor.a.02')": "true",
            "0:Synset('poor.a.03')": "true",
            "0:Synset('poor.a.04')": "true",
            "0:Synset('poor.s.06')": "true",
            "0:lemma": "poor",
            "0:pos": "ADJ",
            "1:Synset('oxidation.n.01')": "true",
            "1:lemma": "oxidation",
            "1:pos": "NOUN",
            "2:Synset('behavior.n.01')": "true",
            "2:Synset('behavior.n.02')": "true",
            "2:Synset('behavior.n.04')": "true",
            "2:Synset('demeanor.n.01')": "true",
            "2:lemma": "behavior",
            "2:pos": "NOUN"
        },
        {
            "-1:Synset('hapless.s.01')": "true",
            "-1:Synset('inadequate.s.02')": "true",
            "-1:Synset('poor.a.02')": "true",
            "-1:Synset('poor.a.03')": "true",
            "-1:Synset('poor.a.04')": "true",
            "-1:Synset('poor.s.06')": "true",
            "-1:lemma": "poor",
            "-1:pos": "ADJ",
            "-2:None": "true",
            "-2:lemma": "__",
            "-2:pos": "__",
            "0:Synset('oxidation.n.01')": "true",
            "0:lemma": "oxidation",
            "0:pos": "NOUN",
            "1:Synset('behavior.n.01')": "true",
            "1:Synset('behavior.n.02')": "true",
            "1:Synset('behavior.n.04')": "true",
            "1:Synset('demeanor.n.01')": "true",
            "1:lemma": "behavior",
            "1:pos": "NOUN",
            "2:Synset('be.v.01')": "true",
            "2:Synset('be.v.02')": "true",
            "2:Synset('be.v.03')": "true",
            "2:Synset('be.v.05')": "true",
            "2:Synset('be.v.08')": "true",
            "2:Synset('be.v.10')": "true",
            "2:Synset('be.v.11')": "true",
            "2:Synset('be.v.12')": "true",
            "2:Synset('constitute.v.01')": "true",
            "2:Synset('cost.v.01')": "true",
            "2:Synset('embody.v.02')": "true",
            "2:Synset('equal.v.01')": "true",
            "2:Synset('exist.v.01')": "true",
            "2:lemma": "be",
            "2:pos": "VERB"
        },
        {
            "-1:Synset('oxidation.n.01')": "true",
            "-1:lemma": "oxidation",
            "-1:pos": "NOUN",
            "-2:Synset('hapless.s.01')": "true",
            "-2:Synset('inadequate.s.02')": "true",
            "-2:Synset('poor.a.02')": "true",
            "-2:Synset('poor.a.03')": "true",
            "-2:Synset('poor.a.04')": "true",
            "-2:Synset('poor.s.06')": "true",
            "-2:lemma": "poor",
            "-2:pos": "ADJ",
            "0:Synset('behavior.n.01')": "true",
            "0:Synset('behavior.n.02')": "true",
            "0:Synset('behavior.n.04')": "true",
            "0:Synset('demeanor.n.01')": "true",
            "0:lemma": "behavior",
            "0:pos": "NOUN",
            "1:Synset('be.v.01')": "true",
            "1:Synset('be.v.02')": "true",
            "1:Synset('be.v.03')": "true",
            "1:Synset('be.v.05')": "true",
            "1:Synset('be.v.08')": "true",
            "1:Synset('be.v.10')": "true",
            "1:Synset('be.v.11')": "true",
            "1:Synset('be.v.12')": "true",
            "1:Synset('constitute.v.01')": "true",
            "1:Synset('cost.v.01')": "true",
            "1:Synset('embody.v.02')": "true",
            "1:Synset('equal.v.01')": "true",
            "1:Synset('exist.v.01')": "true",
            "1:lemma": "be",
            "1:pos": "VERB",
            "2:None": "true",
            "2:lemma": "the",
            "2:pos": "DET"
        },
        {
            "-1:Synset('behavior.n.01')": "true",
            "-1:Synset('behavior.n.02')": "true",
            "-1:Synset('behavior.n.04')": "true",
            "-1:Synset('demeanor.n.01')": "true",
            "-1:lemma": "behavior",
            "-1:pos": "NOUN",
            "-2:Synset('oxidation.n.01')": "true",
            "-2:lemma": "oxidation",
            "-2:pos": "NOUN",
            "0:Synset('be.v.01')": "true",
            "0:Synset('be.v.02')": "true",
            "0:Synset('be.v.03')": "true",
            "0:Synset('be.v.05')": "true",
            "0:Synset('be.v.08')": "true",
            "0:Synset('be.v.10')": "true",
            "0:Synset('be.v.11')": "true",
            "0:Synset('be.v.12')": "true",
            "0:Synset('constitute.v.01')": "true",
            "0:Synset('cost.v.01')": "true",
            "0:Synset('embody.v.02')": "true",
            "0:Synset('equal.v.01')": "true",
            "0:Synset('exist.v.01')": "true",
            "0:lemma": "be",
            "0:pos": "VERB",
            "1:None": "true",
            "1:lemma": "the",
            "1:pos": "DET",
            "2:Synset('major.a.01')": "true",
            "2:Synset('major.a.02')": "true",
            "2:Synset('major.a.03')": "true",
            "2:Synset('major.a.04')": "true",
            "2:Synset('major.a.05')": "true",
            "2:Synset('major.a.06')": "true",
            "2:Synset('major.a.07')": "true",
            "2:Synset('major.s.08')": "true",
            "2:lemma": "major",
            "2:pos": "ADJ"
        },
        {
            "-1:Synset('be.v.01')": "true",
            "-1:Synset('be.v.02')": "true",
            "-1:Synset('be.v.03')": "true",
            "-1:Synset('be.v.05')": "true",
            "-1:Synset('be.v.08')": "true",
            "-1:Synset('be.v.10')": "true",
            "-1:Synset('be.v.11')": "true",
            "-1:Synset('be.v.12')": "true",
            "-1:Synset('constitute.v.01')": "true",
            "-1:Synset('cost.v.01')": "true",
            "-1:Synset('embody.v.02')": "true",
            "-1:Synset('equal.v.01')": "true",
            "-1:Synset('exist.v.01')": "true",
            "-1:lemma": "be",
            "-1:pos": "VERB",
            "-2:Synset('behavior.n.01')": "true",
            "-2:Synset('behavior.n.02')": "true",
            "-2:Synset('behavior.n.04')": "true",
            "-2:Synset('demeanor.n.01')": "true",
            "-2:lemma": "behavior",
            "-2:pos": "NOUN",
            "0:None": "true",
            "0:lemma": "the",
            "0:pos": "DET",
            "1:Synset('major.a.01')": "true",
            "1:Synset('major.a.02')": "true",
            "1:Synset('major.a.03')": "true",
            "1:Synset('major.a.04')": "true",
            "1:Synset('major.a.05')": "true",
            "1:Synset('major.a.06')": "true",
            "1:Synset('major.a.07')": "true",
            "1:Synset('major.s.08')": "true",
            "1:lemma": "major",
            "1:pos": "ADJ",
            "2:Synset('barrier.n.01')": "true",
            "2:Synset('barrier.n.02')": "true",
            "2:Synset('barrier.n.03')": "true",
            "2:lemma": "barrier",
            "2:pos": "NOUN"
        },
        {
            "-1:None": "true",
            "-1:lemma": "the",
            "-1:pos": "DET",
            "-2:Synset('be.v.01')": "true",
            "-2:Synset('be.v.02')": "true",
            "-2:Synset('be.v.03')": "true",
            "-2:Synset('be.v.05')": "true",
            "-2:Synset('be.v.08')": "true",
            "-2:Synset('be.v.10')": "true",
            "-2:Synset('be.v.11')": "true",
            "-2:Synset('be.v.12')": "true",
            "-2:Synset('constitute.v.01')": "true",
            "-2:Synset('cost.v.01')": "true",
            "-2:Synset('embody.v.02')": "true",
            "-2:Synset('equal.v.01')": "true",
            "-2:Synset('exist.v.01')": "true",
            "-2:lemma": "be",
            "-2:pos": "VERB",
            "0:Synset('major.a.01')": "true",
            "0:Synset('major.a.02')": "true",
            "0:Synset('major.a.03')": "true",
            "0:Synset('major.a.04')": "true",
            "0:Synset('major.a.05')": "true",
            "0:Synset('major.a.06')": "true",
            "0:Synset('major.a.07')": "true",
            "0:Synset('major.s.08')": "true",
            "0:lemma": "major",
            "0:pos": "ADJ",
            "1:Synset('barrier.n.01')": "true",
            "1:Synset('barrier.n.02')": "true",
            "1:Synset('barrier.n.03')": "true",
            "1:lemma": "barrier",
            "1:pos": "NOUN",
            "2:None": "true",
            "2:lemma": "to",
            "2:pos": "ADP"
        },
        {
            "-1:Synset('major.a.01')": "true",
            "-1:Synset('major.a.02')": "true",
            "-1:Synset('major.a.03')": "true",
            "-1:Synset('major.a.04')": "true",
            "-1:Synset('major.a.05')": "true",
            "-1:Synset('major.a.06')": "true",
            "-1:Synset('major.a.07')": "true",
            "-1:Synset('major.s.08')": "true",
            "-1:lemma": "major",
            "-1:pos": "ADJ",
            "-2:None": "true",
            "-2:lemma": "the",
            "-2:pos": "DET",
            "0:Synset('barrier.n.01')": "true",
            "0:Synset('barrier.n.02')": "true",
            "0:Synset('barrier.n.03')": "true",
            "0:lemma": "barrier",
            "0:pos": "NOUN",
            "1:None": "true",
            "1:lemma": "to",
            "1:pos": "ADP",
            "2:None": "true",
            "2:lemma": "the",
            "2:pos": "DET"
        },
        {
            "-1:Synset('barrier.n.01')": "true",
            "-1:Synset('barrier.n.02')": "true",
            "-1:Synset('barrier.n.03')": "true",
            "-1:lemma": "barrier",
            "-1:pos": "NOUN",
            "-2:Synset('major.a.01')": "true",
            "-2:Synset('major.a.02')": "true",
            "-2:Synset('major.a.03')": "true",
            "-2:Synset('major.a.04')": "true",
            "-2:Synset('major.a.05')": "true",
            "-2:Synset('major.a.06')": "true",
            "-2:Synset('major.a.07')": "true",
            "-2:Synset('major.s.08')": "true",
            "-2:lemma": "major",
            "-2:pos": "ADJ",
            "0:None": "true",
            "0:lemma": "to",
            "0:pos": "ADP",
            "1:None": "true",
            "1:lemma": "the",
            "1:pos": "DET",
            "2:Synset('increase.v.01')": "true",
            "2:Synset('increase.v.02')": "true",
            "2:lemma": "increase",
            "2:pos": "VERB"
        },
      ],
    """
    sent_feats = []

    for i, token in enumerate(sent):
            
        token_feats = {}
        
        
        for j in range(-context_size, context_size + 1):
            k = j + i

            wnsynsets = []
            
            if 0 <= k < len(sent):
                lemma = sent[k].lemma_
                pos = sent[k].pos_
                if re.match(r"VERB|ADJ|NOUN|ADV",pos):  
                    for synset in eval("wn.synsets(" + repr(lemma) + ", pos=wn." + pos  +")"):  
                        wnsynsets.append(str(synset))
                else:
                    wnsynsets = ["None"]
            else:
                lemma = pos = undefined
                wnsynsets = ["None"]
                
                
            token_feats['{}:lemma'.format(j)] = lemma
            token_feats['{}:pos'.format(j)] = pos
            for synset in wnsynsets:
                token_feats['{}:{}'.format(j, synset)] = "true"

        sent_feats.append(token_feats)

    return sent_feats



# Step 1: Generate features

spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
feats_dir = join('_train', 'features2')

# If you want to save time by resusing feats from crf1-exp/py,
# comment out the line below:
generate_feats(spacy_dir, feats_dir, features2)

# Step 2: Collect data for running CRF classifier

true_iob_dir = join(LOCAL_DIR, 'train', 'iob')

data = collect_crf_data(true_iob_dir, feats_dir)

# Step 3: Create folds

# create folds from complete texts only (i.e. instances of the same text
# are never in different folds)
# TODO How to set seed for random generator?
group_k_fold = GroupKFold(n_splits=2)

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

