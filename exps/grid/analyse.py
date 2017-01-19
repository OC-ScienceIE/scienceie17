import pickle

import pandas as pd


from sie import ENTITIES

for target_label in ENTITIES:
    pkl_fname = 'grid_search_{}.pkl'.format(target_label)
    gs = pickle.load(open(pkl_fname, 'rb'))
    df = pd.DataFrame(gs.cv_results_)
    df.sort_values('mean_test_score', ascending=False, inplace=True)
    print('\n' + 80 * '=')
    print(target_label)
    print(80 * '=' + '\n')
    print(df[:25])

