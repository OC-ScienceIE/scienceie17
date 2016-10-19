import json
from glob import glob
from os.path import join, basename, splitext

from sie import ENTITIES
from sie.feats import Features


def collect_crf_data(iob_dir, *feat_dirs):
    """
    Collect the data to train/eval CRF classifier.
    Labels for entities are derived from IOB tags in the files in the iob_dir.
    Features are collected from the json files in one or more feat_dir.
    Group names are derived from the filenames.
    """
    data = dict((label, list()) for label in ENTITIES)
    data['feats'] = []
    data['groups'] = []

    for iob_fname in glob(join(iob_dir, '*.json')):
        text_iob = json.load(open(iob_fname))

        feat_basename = basename(iob_fname)
        feat_filenames = [join(dir, feat_basename)  for dir in feat_dirs]
        text_feat = Features.from_file(*feat_filenames)
        assert len(text_iob) == len(text_feat)
        data['feats'] += text_feat

        for label in ENTITIES:
            data[label] += _text_iob_tags(text_iob, label)

        data['groups'] += len(text_iob) * [splitext(basename(iob_fname))[0]]

    return data


def _text_iob_tags(text_iob, label):
    return [_sent_iob_tags(sent_iob, label) for sent_iob in text_iob]


def _sent_iob_tags(sent_iob, label):
    return [token_iob[label] for token_iob in sent_iob]