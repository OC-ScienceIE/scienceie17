"""
Test of performance loss due to use of IOB scheme

Converts the derived IOB tags for train/dev data back to Brat annotation format
(.ann files) and then use the evaluation script to compare it to the original
Brat annotation files.
If the conversion was perfect, scores would be perfect.
However, scores are lower because some annotation spans cannot be aligned
to the tokens resulting from Spacy.
Also, entities embedded in entities of the same type (e.g. a Material text span
containing another Material text span) can not be represented in an IOB schema.
"""

from os.path import join

from sie import DATA_DIR, LOCAL_DIR
from sie.brat import iob_to_brat

from eval import calculateMeasures

for part in 'train', 'dev':
    iob_dir = join(LOCAL_DIR, part, 'iob')
    txt_dir = join(DATA_DIR, part)
    brat_dir = join('_brat', part)
    iob_to_brat(iob_dir, txt_dir, brat_dir)
    print('\nScores for {} part:\n'.format(part))
    calculateMeasures(txt_dir, brat_dir, 'rel')
