import json
from glob import glob
from os.path import join, basename
from os import makedirs

from sie import ENTITIES
from sie.feats import Features


def collect_crf_data(iob_dir, *feat_dirs):
    """
    Collect the data to train/eval CRF classifier.
    Labels for entities are derived from IOB tags in the files in the iob_dir.
    Features are collected from the json files in one or more feat_dir.
    Filenames are the basenames of the iob files.
    """
    data = dict((label, list()) for label in ENTITIES)
    data['feats'] = []
    data['filenames'] = []

    for iob_fname in glob(join(iob_dir, '*.json')):
        text_iob = json.load(open(iob_fname))

        filename = basename(iob_fname)
        feat_filenames = [join(dir, filename) for dir in feat_dirs]
        text_feat = Features.from_file(*feat_filenames)
        assert len(text_iob) == len(text_feat)
        data['feats'] += text_feat

        for label in ENTITIES:
            data[label] += _text_iob_tags(text_iob, label)

        data['filenames'] += len(text_iob) * [filename]

    return data


def _text_iob_tags(text_iob, label):
    return [_sent_iob_tags(sent_iob, label) for sent_iob in text_iob]


def _sent_iob_tags(sent_iob, label):
    return [token_iob[label] for token_iob in sent_iob]


def pred_to_iob(pred, filenames, true_iob_dir, pred_iob_dir):
    """
    Convert predictions from CRF classifier to IOB tags

    Parameters
    ----------
    pred: prediction from CRF classifier
    filenames: filename origin for each sentence
    true_iob_dir: directory for annotated IOB tags
    pred_iob_dir: directory for predicted IOB tags
    """

    def write_pred_iob():
        pred_iob_fname = join(pred_iob_dir, prev_iob_fname)
        with open(pred_iob_fname, 'w') as outf:
            print('writing ' + pred_iob_fname)
            json.dump(true_iob, outf, indent=4, sort_keys=True, ensure_ascii=False)

    makedirs(pred_iob_dir, exist_ok=True)
    prev_iob_fname = None

    for sent_count, iob_fname in enumerate(filenames):
        if iob_fname != prev_iob_fname:
            if prev_iob_fname:
                write_pred_iob()
            true_iob_fname = join(true_iob_dir, iob_fname)
            true_iob = json.load(open(true_iob_fname))
            true_iob_iter = iter(true_iob)
            prev_iob_fname = iob_fname

        sent_iob = next(true_iob_iter)

        for token_count, token_iob in enumerate(sent_iob):
            for ent in ENTITIES:
                token_iob[ent] = pred[ent][sent_count][token_count]

    write_pred_iob()
