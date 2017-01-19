#!/usr/bin/env python

"""
evaluate CRF++ output on ScienceIE keyword prediction task
"""

import json
from glob import glob
from os import makedirs
from os.path import join, splitext, basename

from sie import LOCAL_DIR, ENTITIES, DATA_DIR
from sie.brat import iob_to_brat
from sie.utils import sorted_glob

from eval import calculateMeasures


def convert(crfplus_dirs, true_iob_dir, pred_iob_dir):
    makedirs(pred_iob_dir, exist_ok=True)

    for iob_fname in sorted_glob(join(true_iob_dir, '*.json')):
        try:
            doc_iob = json.load(open(iob_fname))
            base_name = basename(iob_fname)

            for label in ENTITIES:
                crfplus_fname = join(
                    crfplus_dirs[label],
                    base_name.replace('.json', '.txt'))
                f = open(crfplus_fname)

                for sent_iob in doc_iob:
                    for tok_iob in sent_iob:
                        line = next(f)
                        pred_tag = line.split('\t')[2].strip()
                        tok_iob[label] = pred_tag
                    next(f)

            pred_iob_fname = join(pred_iob_dir, base_name)
            json.dump(doc_iob, open(pred_iob_fname, 'w'),
                      indent=4, sort_keys=True, ensure_ascii=False)
        except Exception as err:
            print('*** ERRROR **', err)
            print(crfplus_fname)
            print(line)
            print()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='evaluate CRF++ output on ScienceIE keyword prediction task')

    parser.add_argument('true_iob_dir',
                        help='directory containing json files with true IOB tags')
    parser.add_argument('true_brat_dir',
                        help='directory containing true Brat annotation files')
    parser.add_argument('material_dir',
                        help='directory containing tab-delimited files with predicted IOB tags for label "Material" in 3rd column')
    parser.add_argument('process_dir',
                        help='directory containing tab-delimited files with predicted IOB tags for label "Process" in 3rd column')
    parser.add_argument('task_dir',
                        help='directory containing tab-delimited files with predicted IOB tags for label "Task" in 3rd column')
    parser.add_argument('pred_iob_dir',
                        help='directory for writing json files with predicted IOB tags')
    parser.add_argument('pred_brat_dir',
                        help='directory for writing predicted Brat annotation files')

    args = parser.parse_args()

    # Step 1: Convert CFR++ output to IOB tags in Json format
    crfplus_dirs = {
        'Material': args.material_dir,
        'Process': args.process_dir,
        'Task': args.task_dir
    }

    convert(crfplus_dirs, args.true_iob_dir, args.pred_iob_dir)

    # Step 2: Convert predicted IOB tags to predicted Brat annotations
    iob_to_brat(args.pred_iob_dir, args.true_brat_dir, args.pred_brat_dir)

    # Step 3: Evaluate
    calculateMeasures(args.true_brat_dir, args.pred_brat_dir, 'rel')



