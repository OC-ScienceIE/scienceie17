"""
wordnet features, hypernyms, including synsets of current token
"""


from os.path import join

from sklearn_crfsuite import CRF

from nltk.corpus import wordnet as wn

from sie import ENTITIES, LOCAL_DIR, EXPS_DIR
from sie.feats import generate_feats, features1
from sie.exp import run_exp_train_cv, eval_exp_train
from sie.crf import PruneCRF


# Step 1: Generate features

base_feats_dir = join(EXPS_DIR, 'crf1/_train/features1')
word_feats_dir = join(EXPS_DIR, 'wordfeats/_train/wordfeats1')

spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
wn_feats_dir = join('_train', 'wnfeats1')

def wnfeats1(sent, context_size=2, undefined='__'):
    """
    wordnet features
    """
    sent_feats = []

    for i, token in enumerate(sent):
        token_feats = {}

        for j in range(-context_size, context_size + 1):
            k = j + i

            if 0 <= k < len(sent):
                lemma = sent[k].lemma_
                pos = sent[k].pos_
                wn_pos = getattr(wn, pos, None)
                # include synsets of currrent token
                synsets = wn.synsets(lemma, pos=wn_pos) or wn.synsets(lemma)

                for synset in synsets:
                    try:
                        token_feats['{}:{}'.format(j, synset.hypernyms()[0].hypernyms()[0].hypernyms()[0].name())] = 1
                    except:
                        pass

            #token_feats['{}:lemma'.format(j)] = lemma
            #token_feats['{}:pos'.format(j)] = pos

        sent_feats.append(token_feats)

    return sent_feats



generate_feats(spacy_dir, wn_feats_dir, wnfeats1)


# Step 2: Run experiments

crf = PruneCRF(c1=0.1, c2=0.1, all_possible_transitions=True)
feat_dirs = [base_feats_dir, wn_feats_dir, word_feats_dir]
preds = {}

for label in ENTITIES:
    preds[label] = run_exp_train_cv(crf, feat_dirs, label, n_folds=5)


# Step 3: Evaluate

eval_exp_train(preds)
