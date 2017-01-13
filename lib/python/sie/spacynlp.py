"""
NLP processing of annotated data with Spacy
"""

from glob import glob
from os.path import join, splitext, basename, exists
from os import makedirs
import pickle
import json

import numpy as np

import spacy
from spacy.tokens.doc import Doc
from spacy.tokens import Span

from sie import ENTITIES


def run_nlp(txt_dir, spacy_dir, nlp=None):
    """
    Process text files in directory txt_dir with Spacy NLP pipeline and
    serialize analyses to directory spacy_dir
    """
    if not nlp:
        nlp = spacy.load('en')

    makedirs(spacy_dir, exist_ok=True)

    for txt_fname in glob(join(txt_dir, '*.txt')):
        print('reading ' + txt_fname)
        text = open(txt_fname).read()
        # Spacy considers '\n' as a separate token.
        # That causes problems when writing tokens in column format,
        # so we strip the final '\n'.
        doc = nlp(text.rstrip('\n'))
        spacy_fname = join(spacy_dir,
                           splitext(basename(txt_fname))[0] + '.spacy')
        print('writing ' + spacy_fname)
        write_doc(spacy_fname, doc)


def generate_iob_tags(ann_dir, spacy_dir, iob_dir, nlp=None):
    """
    Generate files with IOB tags from Brat .ann files in ann_dir,
    Spacy serialized analyses in spacy_dir, writing to output files to iob_dir
    """
    #TODO This does not correctly handle embedded entities of the same type!
    # E.g. Material inside another Material

    if not nlp:
        nlp = spacy.load('en')

    makedirs(iob_dir, exist_ok=True)
    correct = incorrect = 0
    txt_count = ann_count = iob_count = 0

    for txt_fname in glob(join(ann_dir, '*.txt')):
        txt_count += 1
        spacy_fname = join(spacy_dir, splitext(basename(txt_fname))[0] + '.spacy')
        doc = read_doc(spacy_fname, nlp)
        char2token = map_chars_to_tokens(doc)

        iob_tags = {}
        for label in ENTITIES:
            iob_tags[label] = len(doc) * ['O']

        ann_fname = txt_fname.replace('.txt', '.ann')

        if exists(ann_fname):
            print('reading ' + ann_fname)
            ann_count += 1

            for line in open(ann_fname):
                if line.startswith('T'):
                    try:
                        label, begin_char, end_char = line.split('\t')[1].split()
                    except ValueError:
                        print('Oh no! Malformed annotation:\n' + line)
                        continue

                    begin_char, end_char = int(begin_char), int(end_char)
                    start_token = char2token[begin_char]
                    end_token = char2token[end_char]

                    span = Span(doc, start_token, end_token, label=14)

                    if span.start_char != begin_char or span.end_char != end_char:
                        print('BRAT SPAN:   ', doc.text[begin_char:end_char])
                        print('SPACY SPAN:  ', span)
                        toks = [t.text
                                for t in doc[max(0, start_token - 3):end_token + 3]]
                        print('SPACY TOKENS:', toks)
                        print()
                        incorrect += 1
                    else:
                        iob_tags[label][start_token] = 'B'
                        for i in range(start_token + 1, end_token):
                            iob_tags[label][i] = 'I'
                        correct += 1
        else:
            # test data has no annotation
            print('WARNING: no annotation file ' + ann_fname)

        iob_fname = join(iob_dir, splitext(basename(ann_fname))[0] + '.json')
        iob_count += 1
        write_iob_file(iob_fname, doc, iob_tags)

    print('\n#succesful spans: {}\n#failed spans: {}'.format(correct, incorrect))
    print('\n#txt files: {}\n#ann files: {}\n#iob files: {}'.format(txt_count, ann_count, iob_count))


def write_iob_file(iob_fname, doc, iob_tags):
    doc_iob = []

    for sent in doc.sents:
        sent_iob = []

        for token in sent:
            token_iob = {'token': token.orth_,
                         'begin': token.idx,
                         'end': token.idx + len(token.orth_),
                         'Material': iob_tags['Material'][token.i],
                         'Process': iob_tags['Process'][token.i],
                         'Task': iob_tags['Task'][token.i]
                         }
            sent_iob.append(token_iob)
        doc_iob.append(sent_iob)

    with open(iob_fname, 'w') as outf:
        print('writing ' + iob_fname)
        json.dump(doc_iob, outf, indent=4, sort_keys=True, ensure_ascii=False)


def map_chars_to_tokens(doc):
    """
    Creates a mapping from input characters to corresponding input tokens

    For instance, given the input:

    Nuclear theory ...
    |||||||||||||||
    012345678911111...
              01234

    it returns an array of size equal to the number of input chars plus one,
    whcih looks like this:

    000000011111112...

    This means that the first 7 chars map to the first token ("Nuclear"),
    the next 7 chars (including the initial whitespace) map to the second
    token ("theory") and so on.
    """
    n_chars = len(doc.text_with_ws)
    char2token = np.zeros(n_chars + 1, 'int')
    start_char = 0
    for token in doc:
        end_char = token.idx + len(token)
        char2token[start_char:end_char] = token.i
        start_char = end_char
    char2token[-1] = char2token[-2] + 1
    return char2token


def write_doc(spacy_fname, doc):
    print('writing ' + spacy_fname)
    byte_string = doc.to_bytes()
    open(spacy_fname, 'wb').write(byte_string)


def read_doc(spacy_fname, nlp):
    print('reading ' + spacy_fname)
    byte_string = next(Doc.read_bytes(open(spacy_fname, 'rb')))
    doc = Doc(nlp.vocab)
    doc.from_bytes(byte_string)
    return doc


# ******************************************************************************
# Code below is not functional yet, because of a bug in Spacy that prevents
# serialization to byte strings of Docs with new entity labels.
# ******************************************************************************


def add_entities(ann_dir, spacy_dir, ents_dir=None, nlp=None):
    """
    Add Method, Process and Task entities from .ann files as entity spans
    to the corresponding serialized Spacy analyses in directory spacy_dir
    """
    if not nlp:
        nlp = spacy.load('en')

    if ents_dir:
        makedirs(ents_dir, exist_ok=True)

    register_entities(nlp)
    correct = incorrect = 0

    for ann_fname in glob(join(ann_dir, '*.ann')):
        print('reading ' + ann_fname)
        spacy_fname = join(spacy_dir,
                           splitext(basename(ann_fname))[0] + '.spacy')
        doc = read_doc(spacy_fname, nlp)

        # see https://github.com/spacy-io/spaCy/issues/461
        entities = [(e.label, e.start, e.end) for e in doc.ents]

        char2token = map_chars_to_tokens(doc)

        for line in open(ann_fname):
            if line.startswith('T'):
                try:
                    label, begin, end = line.split('\t')[1].split()
                except ValueError:
                    print('Oh no! Malformed annotation:\n' + line)
                    continue

                start_token = char2token[int(begin)]
                end_token = char2token[int(end)]

                span = Span(doc, start_token, end_token, label=14)

                if span.start_char != int(begin) or span.end_char != int(end):
                    print('BRAT SPAN:   ', doc.text[int(begin):int(end)])
                    print('SPACY SPAN:  ', span)
                    toks = [t.text
                            for t in doc[max(0, start_token - 3):end_token + 3]]
                    print('SPACY TOKENS:', toks)
                    print()
                    incorrect += 1
                else:
                    label_id = nlp.vocab.strings[label]
                    entities.append((label_id, start_token, end_token))
                    correct += 1

        if ents_dir:
            ents_fname = join(ents_dir,
                              splitext(basename(ann_fname))[0] + '_ents.pkl')
            print('writing ' + ents_fname)
            pickle.dump(entities, open(ents_fname, 'wb'))
        else:
            # Save ents in doc
            # FIXME: this currently fails with KeyError!
            # See https://github.com/explosion/spaCy/issues/514
            # doc.ents behaves like a set, so adding duplicates is harmless
            doc.ents = entities
            write_doc(spacy_fname, doc)

    print('\n#succesful spans: {}\n#failed spans: {}'.format(
        correct, incorrect))


def register_entities(nlp):
    """
    Register ScienceIE entities with Spacy
    """
    # integer ID can be obtained with nlp.vocab.strings[label]
    for label in 'Material', 'Process', 'Task':
        nlp.entity.add_label(label)
