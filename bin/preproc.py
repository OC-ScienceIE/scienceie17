#!/usr/bin/env python

"""
Preprocess annotated data (train/dev/test):
- NLP by Spacy
- generation of IOB tags for each entity type

Writes file output to dir _local.
"""

from os.path import join
from os import makedirs

from sie import DATA_DIR, LOCAL_DIR
from sie.spacynlp import run_nlp, generate_iob_tags

# TODO: preproc test data once available
data_parts = 'train', 'dev'  # , 'test'

for part in data_parts:
    part_dir = join(DATA_DIR, part)
    makedirs(part_dir, exist_ok=True)

    spacy_dir = join(LOCAL_DIR, part, 'spacy')
    makedirs(spacy_dir, exist_ok=True)

    run_nlp(part_dir, spacy_dir)

    iob_dir = join(LOCAL_DIR, part, 'iob')
    makedirs(iob_dir, exist_ok=True)

    generate_iob_tags(part_dir, spacy_dir, iob_dir)
