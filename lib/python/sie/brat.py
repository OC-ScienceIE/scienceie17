from glob import glob
from os.path import join, splitext, basename
from os import makedirs

from collections import namedtuple

from sie import ENTITIES

Span = namedtuple('Span', ('label', 'begin', 'end'))


def iob_to_brat(iob_dir, txt_dir, brat_dir):
    for iob_fname in glob(join(iob_dir, '*.tsv')):
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
    columns = read_iob_file(iob_fname)
    spans = []

    for label in ENTITIES:
        begin = None
        iob_tags = columns[label]

        for i, tag in enumerate(iob_tags):
            if begin and (tag.startswith('B-') or tag.startswith('O-')):
                # close open span
                end = int(columns['end'][i - 1])
                span = Span(label, begin, end)
                spans.append(span)
                begin = None

            if tag.startswith('B-'):
                # open new span
                begin = int(columns['begin'][i])

        if False and begin:
            # there is still an open span when last tag is B-*
            end = int(columns['end'][-1])
            span = Span(label, begin, end)
            spans.append(span)

    # sort on begin char offsets
    spans.sort(key=lambda span: span.begin)
    return spans


def read_iob_file(iob_fname):
    # Assumes a tab-separated file where the last five columns contain
    #   -5: begin character offset
    #   -4: end character offset
    #   -3: IOB tags for Material
    #   -2: IOB tags for Process
    #   -1: IOB tags for Task
    # All preceding columns are ignored.
    keys = ('begin', 'end') + ENTITIES
    print('reading ' + iob_fname)
    records = (line.rstrip('\n').split('\t')[-5:] for line in open(iob_fname))
    columns = zip(*records)
    return dict(pair for pair in zip(keys, columns))


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
