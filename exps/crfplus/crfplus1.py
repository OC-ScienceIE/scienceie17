"""
convert output from CRF++ to Brat annotation,
so we can use eval.py to compute official scores

Expects input files in
entityOp_Utpal/materialOp
entityOp_Utpal/processOp
entityOp_Utpal/taskOp

Writes output to
entityOp_Utpal/iob
entityOp_Utpal/brat
"""

import json
from glob import glob
from os import makedirs
from os.path import join, splitext, basename

from sie import LOCAL_DIR, ENTITIES, DATA_DIR
from sie.brat import iob_to_brat

from eval import calculateMeasures


def convert(crfplus_dirs, true_iob_dir, pred_iob_dir):
    makedirs(pred_iob_dir, exist_ok=True)

    for iob_fname in glob(join(true_iob_dir, '*.json')):
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


# Step 1: Convert CFR++ output to IOB tags in Json format
true_iob_dir = join(LOCAL_DIR, 'train/iob')
pred_iob_dir = '_entityOp_Utpal/iob'

crfplus_dirs = {
    'Material': '_entityOp_Utpal/materialOp',
    'Process': '_entityOp_Utpal/processOp',
    'Task': '_entityOp_Utpal/taskOp'
}

convert(
    crfplus_dirs,
    true_iob_dir,
    pred_iob_dir
)


# Step 2: Convert predicted IOB tags to predicted Brat annotations
true_brat_dir = join(DATA_DIR, 'train')
pred_brat_dir = '_entityOp_Utpal/brat'

iob_to_brat(pred_iob_dir, true_brat_dir, pred_brat_dir)

# Step 3: Evaluate
calculateMeasures(true_brat_dir, pred_brat_dir, 'rel')
