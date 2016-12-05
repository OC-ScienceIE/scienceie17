"""
Postprocessing on output of wordfeats1.py

The idea is to keep labeling consistent across a text.
E.g. if the token string "chemical reaction" has been labeled as Process in the majority
of the labeled cases, then any unlabeled instances of the same token string are also
labeled as Process.

Unless the token string is a substring of a larger token string already labeled as
Process (e.g. "chemical reaction  enhancement").

We take the majority label. If there is a draw, then we skip it.

Reads IOB files and writes new IOB files.
"""


from glob import glob
from os.path import join, basename
from os import makedirs
import json

from sie import ENTITIES, EXPS_DIR, DATA_DIR
from sie.brat import iob_to_brat

from eval import calculateMeasures


def get_token_labels(text_iob):
    tokens2labels = {}

    for label in ENTITIES:
        for sent_iob in text_iob:
            tokens = []

            for token_iob in sent_iob:
                tag = token_iob[label]

                if tokens and tag in 'OB':
                    # close open span
                    tok_str = ' '.join(tokens)
                    label2count = tokens2labels.setdefault(tok_str, {})
                    label2count[label] = label2count.get(label, 0) + 1
                    tokens = []

                if tag in 'IB':
                    # continue span or open new span
                    tokens.append(token_iob['token'])

            if tokens:
                # there is still an open span when last tag is B
                tok_str = ' '.join(tokens)
                label2count = tokens2labels.setdefault(tok_str, {})
                label2count[label] = label2count.get(label, 0) + 1

    # for tokens, labels in tokens2labels.items():
    #     if len(labels) > 1:
    #         print(tokens)
    #         print(labels)
    #         print()

    return tokens2labels


def resolve_labels(tokens2labels):
    # Select label with max count
    # In case of a draw, delete tokens from dict.
    for tokens in list(tokens2labels.keys()):
        max_count, max_label = 0, None

        for label, count in tokens2labels[tokens].items():
            if count > max_count:
                max_count, max_label = count, label
            elif count == max_count:
                del tokens2labels[tokens]
                break
        else:
            tokens2labels[tokens] = max_label


def relabel(text_iob, tokens2labels):
    for sent_iob in text_iob:
        for n in range(1, 10):
            for i in range(max(len(sent_iob) - n, 0)):
                elems = sent_iob[i:i + n]
                tokens = ' '.join([e['token'] for e in elems])

                try:
                    label = tokens2labels[tokens]
                except KeyError:
                    continue

                # Check that tokens are not already labeled, or
                # part of a larget token sequence, with the same label
                if all(e[label] not in 'BI' for e in elems):
                    print(tokens, '==>', label)
                    elems[0][label] = 'B'
                    for e in elems[1:]:
                        e[label] = 'I'


def all_token_labels(in_iob_dir, out_iob_dir):
    makedirs(out_iob_dir, exist_ok=True)

    for in_iob_fname in glob(join(in_iob_dir, '*.json')):
        print('reading ' + in_iob_fname)
        text_iob = json.load(open(in_iob_fname))
        tokens2labels = get_token_labels(text_iob)
        resolve_labels(tokens2labels)
        relabel(text_iob, tokens2labels)

        out_iob_fname = join(out_iob_dir, basename(in_iob_fname))

        with open(out_iob_fname, 'w') as outf:
            print('writing ' + out_iob_fname)
            json.dump(text_iob, outf, indent=4, sort_keys=True, ensure_ascii=False)


in_iob_dir = join(EXPS_DIR, 'wordfeats/_train/iob')
out_iob_dir = '_train/iob'

all_token_labels(in_iob_dir, out_iob_dir)

# Step 6: Convert predicted IOB tags to predicted Brat annotations
txt_dir = join(DATA_DIR, 'train')
brat_dir = '_train/brat'

iob_to_brat(out_iob_dir, txt_dir, brat_dir)

# Step 7: Evaluate
calculateMeasures(txt_dir, brat_dir, 'rel')