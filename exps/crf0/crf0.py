from sie.spacynlp import run_nlp, generate_iob_tags
from sie.brat import iob_to_brat
from eval import calculateMeasures

run_nlp('../../data/train', '_train/spacy')

generate_iob_tags('../../data/train', '_train/spacy', 'train/iob')

iob_to_brat('train/iob', '../../data/train', '_train/brat')

calculateMeasures('../../data/train', '_train/brat', 'rel')
