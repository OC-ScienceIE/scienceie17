"""
setups for running experiments
"""

from os.path import join
from subprocess import call

from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

from eval import calculateMeasures
from sie import LOCAL_DIR, DATA_DIR
from sie.brat import iob_to_brat
from sie.crf import collect_features, read_labels, read_folds, pred_to_iob


def run_exp_train(crf, feat_dirs, target_label):
    """
    Run an experiment with both training and testing on the train data
    """
    # Collect data for running CRF classifier
    train_dir = join(LOCAL_DIR, 'train')
    true_iob_dir = join(train_dir, 'iob')
    X = collect_features(true_iob_dir, *feat_dirs)
    labels_fname = join(train_dir, 'train_labels.pkl')
    labels = read_labels(labels_fname)
    y_true = labels[target_label]

    # Predict
    crf.fit(X, y_true)
    y_pred = crf.predict(X)
    print(flat_classification_report(y_true, y_pred, digits=3, labels=('B', 'I')))
    return y_pred


def run_exp_dev(crf, train_feat_dirs, dev_feat_dirs, target_label):
    """
    Run an experiment with training on the train data and testing on the dev data
    """
    # Collect data for running CRF classifier
    train_dir = join(LOCAL_DIR, 'train')
    true_iob_dir = join(train_dir, 'iob')
    X_train = collect_features(true_iob_dir, *train_feat_dirs)
    train_labels_fname = join(train_dir, 'train_labels.pkl')
    train_labels = read_labels(train_labels_fname)
    y_train_true = train_labels[target_label]

    dev_dir = join(LOCAL_DIR, 'dev')
    true_iob_dir = join(dev_dir, 'iob')
    X_dev = collect_features(true_iob_dir, *dev_feat_dirs)
    dev_labels_fname = join(dev_dir, 'dev_labels.pkl')
    dev_labels = read_labels(dev_labels_fname)
    y_dev_true = dev_labels[target_label]

    # Predict
    crf.fit(X_train, y_train_true)
    y_dev_pred = crf.predict(X_dev)
    print(flat_classification_report(y_dev_true, y_dev_pred, digits=3, labels=('B', 'I')))
    return y_dev_pred


def run_exp_test(crf, train_feat_dirs, dev_feat_dirs, test_feat_dirs, target_label):
    """
    Run an experiment with training on the train and dev data combined and testing on the test data
    """
    # Collect data for running CRF classifier

    # train
    train_dir = join(LOCAL_DIR, 'train')
    true_iob_dir = join(train_dir, 'iob')
    X_train = collect_features(true_iob_dir, *train_feat_dirs)
    train_labels_fname = join(train_dir, 'train_labels.pkl')
    train_labels = read_labels(train_labels_fname)
    y_train_true = train_labels[target_label]

    # dev
    dev_dir = join(LOCAL_DIR, 'dev')
    true_iob_dir = join(dev_dir, 'iob')
    X_dev = collect_features(true_iob_dir, *dev_feat_dirs)
    dev_labels_fname = join(dev_dir, 'dev_labels.pkl')
    dev_labels = read_labels(dev_labels_fname)
    y_dev_true = dev_labels[target_label]

    # now combine train and dev data
    X_combined = X_train + X_dev
    y_combined_true = y_train_true + y_dev_true

    # test
    test_dir = join(LOCAL_DIR, 'test')
    true_iob_dir = join(test_dir, 'iob')
    X_test = collect_features(true_iob_dir, *test_feat_dirs)
    test_labels_fname = join(test_dir, 'test_labels.pkl')
    test_labels = read_labels(test_labels_fname)
    y_test_true = test_labels[target_label]

    # Predict
    crf.fit(X_combined, y_combined_true)
    y_test_pred = crf.predict(X_test)
    try:
        print(flat_classification_report(y_test_true, y_test_pred, digits=3, labels=('B', 'I')))
    except ZeroDivisionError:
        print('WARNING: no true annotation')
    return y_test_pred


def run_exp_train_cv(crf, feat_dirs, target_label, n_folds=5, n_jobs=-1):
    """
    Run cross-validated experiment on training data
    """
    # Collect data for running CRF classifier
    train_dir = join(LOCAL_DIR, 'train')
    true_iob_dir = join(train_dir, 'iob')
    X = collect_features(true_iob_dir, *feat_dirs)
    labels_fname = join(train_dir, 'train_labels.pkl')
    labels = read_labels(labels_fname)
    y_true = labels[target_label]
    folds_fname = join(train_dir, 'folds.pkl')
    folds = read_folds(folds_fname, n_folds)

    # Predict]
    y_pred = cross_val_predict(crf, X, y_true, cv=folds, verbose=2, n_jobs=n_jobs)
    print(flat_classification_report(y_true, y_pred, digits=3, labels=('B', 'I')))
    return y_pred


def eval_exp_train(preds, part='train', postproc=None, zip_fname=None):
    """
    Evaluate predictions from experiment

    Converts IOB tags predicted by CRF to Brat format and then calls the official scoring function.
    """
    part_dir = join(LOCAL_DIR, part)
    true_iob_dir = join(part_dir, 'iob')

    labels_fname = join(part_dir, part + '_labels.pkl')
    labels = read_labels(labels_fname)
    filenames = labels['__filenames__']

    # Convert CRF prediction to IOB tags
    pred_iob_dir = '_' + part + '/iob'
    pred_to_iob(preds, filenames, true_iob_dir, pred_iob_dir)

    if postproc:
        postproc_dir = '_' + part + '/iob_pp'
        postproc(pred_iob_dir, postproc_dir)
        pred_iob_dir = postproc_dir

    # Convert predicted IOB tags to predicted Brat annotations
    txt_dir = join(DATA_DIR, part)
    brat_dir = '_' + part + '/brat'
    iob_to_brat(pred_iob_dir, txt_dir, brat_dir)

    # Evaluate
    calculateMeasures(txt_dir, brat_dir, 'rel')

    if zip_fname:
        package(brat_dir, part, zip_fname)

    return brat_dir


def package(brat_dir, part, zip_fname):
    txt_dir = join(DATA_DIR, part)
    args = 'zip -j {} {} {}'.format(zip_fname, join(txt_dir, '*.txt'), join(brat_dir, '*.ann'))
    print(args)
    call(args, shell=True)
