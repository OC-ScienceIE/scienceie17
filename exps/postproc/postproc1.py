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

from os.path import join

from eval import calculateMeasures
from sie import EXPS_DIR, DATA_DIR
from sie.brat import iob_to_brat
from sie.postproc import postproc_labels

in_iob_dir = join(EXPS_DIR, 'best/_train/iob')
#in_iob_dir = join(EXPS_DIR, 'prune/_train/iob')
out_iob_dir = '_train/iob'

postproc_labels(in_iob_dir, out_iob_dir)

# Step 6: Convert predicted IOB tags to predicted Brat annotations
txt_dir = join(DATA_DIR, 'train')
brat_dir = '_train/brat'

iob_to_brat(out_iob_dir, txt_dir, brat_dir)

# Step 7: Evaluate
calculateMeasures(txt_dir, brat_dir, 'rel')