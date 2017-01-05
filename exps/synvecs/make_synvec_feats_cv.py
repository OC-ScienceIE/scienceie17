"""
Using Wordnet featurs with an external classifier.

The intuition is that using the synsets of a word and its hypernyms as features willl NER.
E.g. 'iron' and 'carbon' both share the hypernym 'chemical element'; if 'iron' is annotated as Material in the training
data, then the classifeir might infer that unseen 'carbon' in the test data must also be Material.
However, CRF can not handle too many features.
Basic idea: Use a faster classifier to predict to which class (Process, Material, Task, Other) a token belongs.
Then use this prediction (or the probability dist over the classes) as a feature for the CRF.

Technical complication:
The external classifier can not be trained on the whole training data, which would cause leaking of the test data.
Instead it has to be trained on the training folds during CV and then applied to the test fold.
Since this is complicated and expensive, we do it off-line.
The external classifier is used the predict features with CV using *exactly the same 5 folds* as used in teh CRF exps!

Best results for external classifier with LogisticRegression (MaxEnt) and weighting of both samples and classes.

Limited to content words present in Wordnet, otherwise predicted features is undefined.

Feature can be either most likely class or probability dist over the classes (use prob=True).

Bottom line: does not work :-( Adding this features to the CRF does not improve performance over
basic features + word features + pruning.

"""

from os.path import join, exists, basename
from os import makedirs
from glob import glob
from collections import Counter
import json
import pickle

import numpy as np

import spacy

from nltk.corpus import wordnet as wn

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GroupKFold
from sklearn.feature_selection import VarianceThreshold

from sie.spacynlp import read_doc
from sie import LOCAL_DIR, ENTITIES
from sie.crf import collect_crf_data
from sie.feats import Features

CONTENT_POS = set(['NOUN', 'PROPN', 'VERB', 'ADJ', 'ADV'])

LABELS = ENTITIES + ('Other',)


def get_train_test_fold_filenames(true_iob_dir, use_pickle=True):
    pickle_fname = '_train_test_fold_fnames.pkl'

    if use_pickle:
        try:
            return pickle.load(open(pickle_fname, 'rb'))
        except IOError:
            pass

    # Misuse data collecting function to get X, y and filenames.
    # Since we are not interested in the actual features, we pretend true_iob_dir is a feature dir.
    data = collect_crf_data(true_iob_dir, true_iob_dir)

    # Now create
    group_k_fold = GroupKFold(n_splits=5)

    # Create folds from complete texts only (i.e. instances of the same text are never in different folds)
    # Use same split for all three entities.
    # Note that there is no random seed, because the output of group_k_fold.split is deterministic
    # as long as the iob files are globbed in exactly the same order
    splits = group_k_fold.split(data['feats'], data['Material'], data['filenames'])

    fnames = np.array(data['filenames'])
    train_test_fold_fnames = []

    for train_idx, test_idx in splits:
        train_fnames = np.unique(fnames[train_idx])
        test_fnames = np.unique(fnames[test_idx])

        train_test_fold_fnames.append((train_fnames, test_fnames))

    pickle.dump(train_test_fold_fnames, open(pickle_fname, 'wb'))

    return train_test_fold_fnames


def get_entity_lempos_counts(iob_dir, spacy_dir, nlp=None, use_pickle=True):
    pickle_fname = '_counts.pkl'

    if use_pickle:
        try:
            return pickle.load(open(pickle_fname, 'rb'))
        except IOError:
            pass

    iob_fnames = glob(join(iob_dir, '*'))
    spacy_fnames = glob(join(spacy_dir, '*'))
    counts = {}

    if not nlp:
        nlp = spacy.load('en')

    for iob_fname, spacy_fname in zip(iob_fnames, spacy_fnames):
        iob_doc = json.load(open(iob_fname, encoding='utf8'))
        spacy_doc = read_doc(spacy_fname, nlp)
        count = dict((e, Counter()) for e in LABELS)

        for spacy_sent, iob_sent in zip(spacy_doc.sents, iob_doc):
            for spacy_tok, iob_tok in zip(spacy_sent, iob_sent):
                if spacy_tok.pos_ in CONTENT_POS and not spacy_tok.is_stop:
                    lempos = spacy_tok.lemma_ + '#' + spacy_tok.pos_
                    other = True

                    for label in ENTITIES:
                        if iob_tok[label] in 'BI':
                            count[label][lempos] += 1
                            other = False

                    if other:
                        count['Other'][lempos] += 1

        counts[basename(iob_fname)] = count

    pickle.dump(counts, open(pickle_fname, 'wb'))

    return counts


def sum_counts(counts, fold_fname):
    count_sum = dict((e, Counter()) for e in LABELS)

    for fname in fold_fname:
        for label, counter in counts[fname].items():
            count_sum[label].update(counter)

    return count_sum


def make_training_data(counts):
    feat_dicts = []
    y = []
    weights = []

    for label, counter in counts.items():
        for lempos, count in counter.items():
            lemma, pos = lempos.rsplit('#', 1)
            feat = make_wn_feats(lemma, pos)
            # skip if not found in wordnet
            if feat:
                feat_dicts.append(feat)
                weights.append(count)
                y.append(label)

    return feat_dicts, y, weights


def make_wn_feats(lemma, pos):
    # print(lemma, pos)
    feat = {}
    wn_pos = getattr(wn, pos, None)
    synsets = wn.synsets(lemma, pos=wn_pos) or wn.synsets(lemma)

    for synset in synsets:
        # print('SYNSET: ', synset.name())
        feat[synset.name()] = 1
        for hyp_synset in synset.closure(lambda s: s.hypernyms()):
            # print('\tHYPERNYM: ', hyp_synset.name())
            feat[hyp_synset.name()] = 1

    # print(80 * '-')
    return feat


def fit_classifier(feat_dicts=None, y_true=None, weights=None):
    # clf = MultinomialNB()
    clf = LogisticRegression(class_weight='balanced')

    pipeline = Pipeline([
        ('vectorizer', DictVectorizer()),
        ('selection', VarianceThreshold()),
        ('classifier', clf)
    ])

    # cf. http://stackoverflow.com/questions/36205850/sklearn-pipeline-applying-sample-weights-after-applying-a-polynomial-feature-t
    pipeline.fit(feat_dicts, y_true, **{'classifier__sample_weight': weights})

    return pipeline


def generate_synvec_feats(spacy_dir, feat_dir, test_fnames, clf, nlp, probs=False):
    labels = clf.named_steps['classifier'].classes_

    if probs:
        undefined = dict((l, 0.0) for l in labels)
    else:
        undefined = {'Pred': '__'}

    hyp_rel = lambda s: s.hypernyms()

    makedirs(feat_dir, exist_ok=True)

    for fname in test_fnames:
        spacy_fname = join(spacy_dir, fname.replace('.json', '.spacy'))
        spacy_doc = read_doc(spacy_fname, nlp)
        doc_feats = []

        for sent in spacy_doc.sents:
            sent_feats = []

            for token in sent:
                synvec_feats = undefined

                if token.pos_ in CONTENT_POS and not token.is_stop:
                    feat = make_wn_feats(token.lemma_, token.pos_)

                    # if not found in wordnet, then synvec stays undefined
                    if feat:
                        if probs:
                            preds = clf.predict_proba(feat)[0]
                            synvec_feats = dict(zip(labels, preds))
                        else:
                            pred = clf.predict(feat)[0]
                            synvec_feats = {'Pred': pred}

                token_feats = {'synvec': synvec_feats}
                sent_feats.append(token_feats)

            doc_feats.append(sent_feats)

        text_feats = Features(doc_feats)
        feat_fname = join(feat_dir, fname)
        print()
        text_feats.to_file(feat_fname)


true_iob_dir = join(LOCAL_DIR, 'train', 'iob')
spacy_dir = join(LOCAL_DIR, 'train', 'spacy')
synvec_feats_dir = join('_train', 'synvec_feats')

train_test_fold_fnames = get_train_test_fold_filenames(true_iob_dir, use_pickle=True)

counts = get_entity_lempos_counts(true_iob_dir, spacy_dir, use_pickle=True)

nlp = spacy.load('en')

for train_fnames, test_fnames in train_test_fold_fnames:
    # 1. create counts
    train_counts = sum_counts(counts, train_fnames)
    test_counts = sum_counts(counts, test_fnames)

    # 2. create training and test data
    X_train, y_train, weights_train = make_training_data(train_counts)
    X_test, y_test, weights_test = make_training_data(test_counts)

    # 3. fit classifier
    clf = fit_classifier(X_train, y_train, weights_train)

    # 4. predict
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, digits=3, labels=LABELS, sample_weight=weights_test))

    # 5. generate synvec features
    generate_synvec_feats(spacy_dir, synvec_feats_dir, test_fnames, clf, nlp, probs=True)
