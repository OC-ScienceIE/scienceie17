import json
import pickle
from glob import glob
from os.path import join, basename
from os import makedirs

from sklearn_crfsuite import CRF
from sklearn.model_selection import GroupKFold

from sie import ENTITIES
from sie.feats import Features
from sie.utils import sorted_glob


def generate_labels(iob_dir, labels_fname):
    """
    Generate labels to train/eval CRF classifier.
    Labels for entities are derived from IOB tags in the files in the iob_dir.
    Filenames are the basenames of the iob files (used for creating folds and
    converting back CRF predictions to IOB files).
    Saved as a pickled dict with keys for all entity labels plus the special key __filenames__.
    """
    labels = dict((label, list()) for label in ENTITIES)
    labels['__filenames__'] = []

    for iob_fname in sorted_glob(join(iob_dir, '*.json')):
        text_iob = json.load(open(iob_fname))
        filename = basename(iob_fname)

        for label in ENTITIES:
            labels[label] += _text_iob_tags(text_iob, label)

        labels['__filenames__'] += len(text_iob) * [filename]

    print('writing labels to file ' + labels_fname)
    pickle.dump(labels, open(labels_fname, 'wb'))


def read_labels(labels_fname):
    print('reading labels from ' + labels_fname)
    return pickle.load(open(labels_fname, 'rb'))


def generate_folds(labels_fname, folds_fname, max_n_folds=10):
    """
    Generate folds for CV exps with n = 2, ..., max_n_folds.
    Save as pickled dict with n as key.
    """
    filenames = read_labels(labels_fname)['__filenames__']
    folds = {}

    for n in range(2, max_n_folds + 1):
        # Create folds from complete texts only
        # (i.e. instances/sentences of the same text are never in different folds).
        # There is no random seed, because the partitioning algorithm is deterministic.
        group_k_fold = GroupKFold(n_splits=n)
        # Don't bother to pass real X and Y, because they are not really used.
        folds[n] = list(group_k_fold.split(filenames, filenames, filenames))

    print('writing folds to ' + folds_fname)
    pickle.dump(folds, open(folds_fname, 'wb'))


def read_folds(folds_fname, n_folds):
    print('reading folds from ' + folds_fname)
    folds = pickle.load(open(folds_fname, 'rb'))
    return folds[n_folds]


def collect_features(iob_dir, *feat_dirs):
    """
    Collect the features to train/eval CRF classifier from the json files in one or more feat_dirs.
    """
    feats = []

    for iob_fname in sorted_glob(join(iob_dir, '*.json')):
        filename = basename(iob_fname)
        feat_filenames = [join(dir, filename) for dir in feat_dirs]
        text_feat = Features.from_file(*feat_filenames)
        feats += text_feat

    return feats


def collect_crf_data(iob_dir, *feat_dirs):
    # *** DEPRECATED *** Do not use in new experiments!
    """
    Collect the data to train/eval CRF classifier.
    Labels for entities are derived from IOB tags in the files in the iob_dir.
    Features are collected from the json files in one or more feat_dir.
    Filenames are the basenames of the iob files.
    """
    data = dict((label, list()) for label in ENTITIES)
    data['feats'] = []
    data['filenames'] = []

    for iob_fname in sorted_glob(join(iob_dir, '*.json')):
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
                try:
                     iob_tag = pred[ent][sent_count][token_count]
                except KeyError:
                    # assume no predictions for this entity type
                    iob_tag = 'O'

                token_iob[ent] = iob_tag

    write_pred_iob()


class PruneCRF(CRF):
    """
    CRF that prunes training sentences without entities (that is, no I or B labels)

    Would be natural to implement this in a Pipeline, but sklearn currently does not support changing the size of y.
    Cf. https://github.com/scikit-learn/scikit-learn/issues/3855
    """

    def fit(self, X, y):
        Xp, yp = self.prune(X, y)
        return CRF.fit(self, Xp, yp)

    def prune(self, X, y):
        Xp, yp = [], []

        for xs, ys in zip(X, y):
            if any(l != 'O' for l in ys):
                Xp.append(xs)
                yp.append(ys)

        print('pruned from {} to {} sentences'.format(len(y), len(yp)))
        return Xp, yp
