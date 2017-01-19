"""
Create features for best exps
"""

from os.path import join, exists

import spacy

from sie import LOCAL_DIR
from sie.feats import lemma_pos_feats, word_feats, wordnet_feats, generate_feats

nlp = spacy.load('en')


def make_feats(part, label, force=False):
    if label == 'Material':
        return make_material_feats(part, force)
    elif label == 'Process':
        return make_process_feats(part, force)
    elif label == 'Task':
        return make_task_feats(part, force)


def make_material_feats(part, force):
    spacy_dir = join(LOCAL_DIR, part, 'spacy')

    lempos_feats_dir = '_{}/Material/lempos_feats'.format(part)
    if force or not exists(lempos_feats_dir):
        generate_feats(spacy_dir, lempos_feats_dir, lemma_pos_feats, nlp=nlp)

    word_feats_dir = '_{}/Material/word_feats'.format(part)
    if force or not exists(word_feats_dir):
        generate_feats(spacy_dir,
                       word_feats_dir,
                       lambda sent: word_feats(sent, context_size=1),
                       nlp=nlp)

    wn_feats_dir = '_{}/Material/wordnet_feats'.format(part)
    if force or not exists(wn_feats_dir):
        generate_feats(spacy_dir,
                       wn_feats_dir,
                       lambda s: wordnet_feats(s, context_size=2),
                       nlp=nlp)

    return [lempos_feats_dir, word_feats_dir, wn_feats_dir]


def make_process_feats(part, force):
    spacy_dir = join(LOCAL_DIR, part, 'spacy')

    lempos_feats_dir = '_{}/Process/lempos_feats'.format(part)
    if force or not exists(lempos_feats_dir):
        generate_feats(spacy_dir, lempos_feats_dir, lemma_pos_feats, nlp=nlp)

    word_feats_dir = '_{}/Process/word_feats'.format(part)
    if force or not exists(word_feats_dir):
        generate_feats(spacy_dir,
                       word_feats_dir,
                       lambda sent: word_feats(sent, context_size=1),
                       nlp=nlp)

    return [lempos_feats_dir, word_feats_dir]


def make_task_feats(part, force):
    spacy_dir = join(LOCAL_DIR, part, 'spacy')

    lempos_feats_dir = '_{}/Task/lempos_feats'.format(part)
    if force or not exists(lempos_feats_dir):
        generate_feats(spacy_dir,
                       lempos_feats_dir,
                       lambda s: lemma_pos_feats(s, context_size=1),
                       nlp=nlp)

    word_feats_dir = '_{}/Task/word_feats'.format(part)
    if force or not exists(word_feats_dir):
        generate_feats(spacy_dir,
                       word_feats_dir,
                       lambda sent: word_feats(sent, context_size=0),
                       nlp=nlp)

    return [lempos_feats_dir, word_feats_dir]
