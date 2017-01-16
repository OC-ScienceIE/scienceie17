from os.path import join, basename
from glob import glob
import json

from sie import LOCAL_DIR
from sie.utils import sorted_glob

true_iob_dir = join(LOCAL_DIR, 'train', 'iob')
synvec_feats_dir = join('_train', 'synvec_feats')

for iob_fname in sorted_glob(join(true_iob_dir, '*'))[10:12]:
    doc_iob = json.load(open(iob_fname))
    synvec_fname = join(synvec_feats_dir, basename(iob_fname))
    doc_feats = json.load(open(synvec_fname))

    for sent_iob, sent_feats in zip(doc_iob, doc_feats):
        for token_iob, token_feats in zip(sent_iob, sent_feats):
            if 'Pred' in token_feats['synvec']:
                print('{:20} {:10} {:10} {:10} {:10}'.format(
                    token_iob['token'],
                    'Material' if token_iob['Material'] != 'O' else '-',
                    'Process' if token_iob['Process'] != 'O' else '-',
                    'Task' if token_iob['Task'] != 'O' else '-',
                    token_feats['synvec']['Pred']))
            else:
                print('{:20} {:10} {:10} {:10} Material={:.2f}        Process={:.2f}        Task={:.2f}        Other={:.2f}'.format(
                    token_iob['token'],
                    'Material' if token_iob['Material'] != 'O' else '-',
                    'Process' if token_iob['Process'] != 'O' else '-',
                    'Task' if token_iob['Task'] != 'O' else '-',
                    token_feats['synvec']['Material'],
                    token_feats['synvec']['Process'],
                    token_feats['synvec']['Task'],
                    token_feats['synvec']['Other'],
                ))

