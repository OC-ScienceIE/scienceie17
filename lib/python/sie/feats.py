"""
reading & writing feature files in json format
"""

import json
from glob import glob
from os import makedirs
from os.path import join, splitext, basename

import spacy
from nltk.corpus import wordnet as wn

from sie.spacynlp import read_doc
from sie.utils import sorted_glob


class Features(list):
    """
    List of list of dicts representing features for sentences in a text
    """

    @classmethod
    def from_file(self, *filenames):
        """
        read features from json file(s)
        """
        feats = Features._from_file(filenames[0])
        for fname in filenames[1:]:
            feats.update(Features._from_file(fname))
        return feats

    @classmethod
    def _from_file(self, filename):
        print('reading features from ' + filename)
        return Features(json.load(open(filename)))

    def to_file(self, filename, **kwargs):
        """
        write features to json file(s)
        """
        print('writing features to ' + filename)
        json.dump(self, open(filename, 'w'), indent=4, sort_keys=True, ensure_ascii=False, **kwargs)

    def update(self, other):
        """
        update these features with other features
        """
        assert len(self) == len(other)
        for l1, l2 in zip(self, other):
            assert len(l1) == len(l2)
            for d1, d2 in zip(l1, l2):
                # no checking of feature clashes
                d1.update(d2)


def generate_feats(spacy_dir, feat_dir, feat_func, nlp=None):
    """
    Generate features and save to file

    :param spacy_dir: dir with serialized Spacy analyses
    :param feat_dir: output dir for generated feature files in json format
    :param feat_func: function for generating
    :return:
    """
    if not nlp:
        nlp = spacy.load('en')
    makedirs(feat_dir, exist_ok=True)

    for spacy_fname in sorted_glob(join(spacy_dir, '*.spacy')):
        doc = read_doc(spacy_fname, nlp)

        feat_fname = join(feat_dir,
                          splitext(basename(spacy_fname))[0] + '.json')

        text_feats = Features([feat_func(sent) for sent in doc.sents])

        text_feats.to_file(feat_fname)


def lemma_pos_feats(sent, context_size=2, undefined='__'):
    """
    Examle of a simple feature function which generates, for each token in
    the sentence, its lemma and POS as well as that of the two
    preceding/following tokens (i.e. window size=5).

    Example:

    [
        {
          "-1:lemma": "__",
          "-1:pos": "__",
          "-2:lemma": "__",
          "-2:pos": "__",
          "0:lemma": "poor",
          "0:pos": "ADJ",
          "1:lemma": "oxidation",
          "1:pos": "NOUN",
          "2:lemma": "behavior",
          "2:pos": "NOUN"
        },
        {
          "-1:lemma": "poor",
          "-1:pos": "ADJ",
          "-2:lemma": "__",
          "-2:pos": "__",
          "0:lemma": "oxidation",
          "0:pos": "NOUN",
          "1:lemma": "behavior",
          "1:pos": "NOUN",
          "2:lemma": "be",
          "2:pos": "VERB"
        },
        .
        .
        .
        {
          "-1:lemma": "structural",
          "-1:pos": "ADJ",
          "-2:lemma": "temperature",
          "-2:pos": "NOUN",
          "0:lemma": "application",
          "0:pos": "NOUN",
          "1:lemma": ".",
          "1:pos": "PUNCT",
          "2:lemma": "__",
          "2:pos": "__"
        },
        {
          "-1:lemma": "application",
          "-1:pos": "NOUN",
          "-2:lemma": "structural",
          "-2:pos": "ADJ",
          "0:lemma": ".",
          "0:pos": "PUNCT",
          "1:lemma": "__",
          "1:pos": "__",
          "2:lemma": "__",
          "2:pos": "__"
        }
      ],
    """
    sent_feats = []

    for i, token in enumerate(sent):
        token_feats = {}

        for j in range(-context_size, context_size + 1):
            k = j + i

            if 0 <= k < len(sent):
                lemma = sent[k].lemma_
                pos = sent[k].pos_
            else:
                lemma = pos = undefined

            token_feats['{}:lemma'.format(j)] = lemma
            token_feats['{}:pos'.format(j)] = pos

        sent_feats.append(token_feats)

    return sent_feats


# DEPRECATED, for backward compatabiity only
features1 = lemma_pos_feats


def word_feats(sent, context_size=0, undefined='__'):
    """
    word form features
    """
    sent_feats = []

    for i in range(len(sent)):
        token_feats = {}

        for j in range(-context_size, context_size + 1):
            k = j + i

            if 0 <= k < len(sent):
                token = sent[k]
                l = len(token.orth_)

                token_feats['{}:word'.format(j)] = token.orth_
                token_feats['{}:shape'.format(j)] = token.shape_
                token_feats['{}:is_alpha'.format(j)] = token.is_alpha
                token_feats['{}:is_lower'.format(j)] = token.is_lower
                token_feats['{}:is_ascii'.format(j)] = token.is_ascii
                token_feats['{}:is_capitalized'.format(j)] = token.orth_.istitle()
                token_feats['{}:is_upper'.format(j)] = token.orth_.isupper()
                token_feats['{}:is_punct'.format(j)] = token.is_punct
                token_feats['{}:like_num'.format(j)] = token.like_num
                token_feats['{}:prefix2'.format(j)] = token.orth_[:2] if l > 1 else ''
                token_feats['{}:suffix2'.format(j)] = token.orth_[-2:] if l > 1 else ''
                token_feats['{}:prefix3'.format(j)] = token.orth_[:3] if l > 2 else ''
                token_feats['{}:suffix3'.format(j)] = token.orth_[-3:] if l > 2 else ''
                token_feats['{}:prefix4'.format(j)] = token.orth_[:4] if l > 3 else ''
                token_feats['{}:suffix4'.format(j)] = token.orth_[-4:] if l > 3 else ''
                # Crfsuite does not support numerical features (are converted to "one hot")
                # token_feats['{}:char_size'.format(j)] =  l
                token_feats['{}:is_stop'.format(j)] = token.is_stop
                # This numerical feature has large range and causes weird behaviour...
                # token_feats['{}:rank'.format(j)] =  token.rank

        sent_feats.append(token_feats)

    return sent_feats


def wordnet_feats(sent, context_size=2, undefined='__'):
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
                synsets = wn.synsets(lemma, pos=wn_pos)  # or wn.synsets(lemma)

                for synset in synsets:
                    # token_feats['{}:{}'.format(j, synset.name())] = 1

                    for hyp_synset in synset.closure(lambda s: s.hypernyms()):
                        token_feats['{}:hyp:{}'.format(j, hyp_synset.name())] = 1

        sent_feats.append(token_feats)

    return sent_feats


def brown_feats(sent, context_size=1, undefined='__'):
    """
    Brown cluster features
    """
    sent_feats = []

    for i in range(len(sent)):
        token_feats = {}

        for j in range(-context_size, context_size + 1):
            k = j + i

            if 0 <= k < len(sent):
                bit_string = '{0:016b}'.format(sent[k].cluster)

                for p in [2,4,6,8,10,12,16]:
                    token_feats['{}:brown:{}'.format(j, p)] = bit_string[-p:]

        sent_feats.append(token_feats)

    return sent_feats