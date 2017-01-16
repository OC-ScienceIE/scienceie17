"""
producing Brat annotation format
"""

# TODO add doc strings

from glob import glob
from os.path import join, splitext, basename
from os import makedirs
import json

from collections import namedtuple

from sie import ENTITIES
from sie.utils import sorted_glob

Span = namedtuple('Span', ('label', 'begin', 'end'))


def iob_to_brat(iob_dir, txt_dir, brat_dir):
    for iob_fname in sorted_glob(join(iob_dir, '*.json')):
        spans = get_text_spans(iob_fname)
        # need text file for correct whitespace
        txt_fname = join(txt_dir,
                         splitext(basename(iob_fname))[0] + '.txt')
        text = open(txt_fname).read()
        makedirs(brat_dir, exist_ok=True)
        brat_fname = join(brat_dir,
                          splitext(basename(iob_fname))[0] + '.ann')
        write_brat_file(brat_fname, spans, text)


def get_text_spans(iob_fname):
    print('reading ' + iob_fname)
    text_iob = json.load(open(iob_fname))
    spans = []

    for label in ENTITIES:
        for sent_iob in text_iob:
            begin = None

            for i, token_iob in enumerate(sent_iob):
                tag = token_iob[label]

                if begin is not None and tag in 'OB':
                    # close open span
                    end = sent_iob[i - 1]['end']
                    span = Span(label, begin, end)
                    spans.append(span)
                    begin = None

                if tag == 'B':
                    # open new span
                    begin = token_iob['begin']

            if begin is not None:
                # there is still an open span when last tag is B
                end = sent_iob[-1]['end']
                span = Span(label, begin, end)
                spans.append(span)

    # sort on begin char offsets
    spans.sort(key=lambda span: span.begin)
    return spans


def write_brat_file(brat_fname, spans, text):
    print('writing ' + brat_fname)
    with open(brat_fname, 'w') as outf:
        for i, span in enumerate(spans):
            outf.write('T{}\t{} {} {}\t{}\n'.format(
                i + 1,
                span.label,
                span.begin,
                span.end,
                text[span.begin:span.end]
            ))
