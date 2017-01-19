import json
from os import makedirs
from os.path import join, basename

from sie import ENTITIES
from sie.utils import sorted_glob



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


def postproc_labels(in_iob_dir, out_iob_dir):
    makedirs(out_iob_dir, exist_ok=True)

    for in_iob_fname in sorted_glob(join(in_iob_dir, '*.json')):
        print('reading ' + in_iob_fname)
        text_iob = json.load(open(in_iob_fname))
        tokens2labels = get_token_labels(text_iob)
        resolve_labels(tokens2labels)
        relabel(text_iob, tokens2labels)

        out_iob_fname = join(out_iob_dir, basename(in_iob_fname))

        with open(out_iob_fname, 'w') as outf:
            print('writing ' + out_iob_fname)
            json.dump(text_iob, outf, indent=4, sort_keys=True, ensure_ascii=False)