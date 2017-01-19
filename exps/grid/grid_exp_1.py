"""
Attempt at grid search for best params.

Unfortunately, this doesn't work...
Optimising the F score on BI tags does not correspond very well with the official scores,
so the optimized params score lower than the CRF's default params.
"""

from os.path import join
import pickle
from pprint import pprint

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_predict
from sklearn_crfsuite.metrics import flat_classification_report

from sklearn_crfsuite import CRF

from sklearn_crfsuite.scorers import make_scorer
from sklearn_crfsuite.metrics import flat_f1_score

from sie import ENTITIES, LOCAL_DIR, EXPS_DIR
from sie.crf import collect_features, read_labels, read_folds, PruneCRF
from sie.exp import eval_exp_train

# Collect data for running CRF classifier
base_feats_dir = join(EXPS_DIR, 'crf1/_train/features1')
word_feats_dir = join(EXPS_DIR, 'wordfeats/_train/wordfeats1')
feat_dirs = [
    base_feats_dir,
    word_feats_dir
]

train_dir = join(LOCAL_DIR, 'train')
true_iob_dir = join(train_dir, 'iob')
X = collect_features(true_iob_dir, *feat_dirs)

labels_fname = join(train_dir, 'train_labels.pkl')
labels = read_labels(labels_fname)

n_folds = 5
folds_fname = join(train_dir, 'folds.pkl')
folds = read_folds(folds_fname, n_folds)

params_space = {
    'c1': [0.0, 0.001, 0.01, 0.1, 1.0, 10],
    'c2': [0.0, 0.001, 0.01, 0.1, 1.0, 10],
    'min_freq': [0, 1, 2, 3, 5, 10],
    'all_possible_states': [True, False],
    'all_possible_transitions': [True, False]
}

# use the same metric for evaluation
f1_scorer = make_scorer(flat_f1_score, average='weighted', labels=('B', 'I'))

preds = {}
n_jobs = 24

for target_label in ENTITIES:
    crf = PruneCRF(algorithm='lbfgs',
                   max_iterations=100
                   )

    y_true = labels[target_label]

    gs = GridSearchCV(crf,
                      params_space,
                      cv=folds,
                      verbose=1,
                      n_jobs=n_jobs,
                      scoring=f1_scorer,
                      refit=False)
    gs.fit(X, y_true)

    pprint(gs.cv_results_)

    pkl_fname = 'grid_search_{}.pkl'.format(target_label)
    pickle.dump(gs, open(pkl_fname, 'wb'))

    best_crf = PruneCRF()
    best_crf.set_params(**gs.best_params_)
    print('\nBest CRF:\n')
    pprint(best_crf)

    y_pred = cross_val_predict(best_crf, X, y_true, cv=folds, verbose=2, n_jobs=n_jobs)
    print(flat_classification_report(y_true, y_pred, digits=3, labels=('B', 'I')))

    preds[target_label] = y_pred

eval_exp_train(preds)
