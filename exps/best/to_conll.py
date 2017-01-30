from glob import glob
from os import makedirs
from os.path import join, basename
import json

from sie import ENTITIES


def to_conll(iob_dir, conll_dir):
    for label in ENTITIES:
        makedirs(join(conll_dir, label), exist_ok=True)

    for iob_fname in glob(join(iob_dir, '*.json')):
        doc_iob = json.load(open(iob_fname))
        txt_fname = basename(iob_fname).replace('.json', '.txt')

        out_files = [open(join(conll_dir, label, txt_fname), 'w')
                     for label in ENTITIES]

        for sent_iob in doc_iob:
            for token_iob in sent_iob:
                for label, f in zip(ENTITIES, out_files):
                    line = '{}\t_\t{}\n'.format(token_iob['token'], token_iob[label])
                    f.write(line)

            for f in out_files:
                f.write('\n')

        for f in out_files:
            f.close()


to_conll(
    '_dev/iob_pp',
    '_dev/conll'
)

to_conll(
    '_test/iob_pp',
    '_test/conll'
)
