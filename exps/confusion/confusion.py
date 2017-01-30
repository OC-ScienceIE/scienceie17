from os.path import join

from sie import ENTITIES, DATA_DIR, EXPS_DIR
from sie.utils import sorted_glob



def read_true_brat_annots(brat_fname):
    annots = {}

    for line in open(brat_fname):
        if line.startswith('T'):
            part = line.split('\t')[1]
            try:
                label, start, end = part.split()
            except ValueError:
                print('ERROR: {} : {}'.format(brat_fname, line))
            else:
                annots[(start,end)] = label

    return annots



def confusion(true_brat_dir, pred_brat_dir):
    cm = {}

    for pred_label in ENTITIES:
        cm[pred_label] = {}
        for true_label in ENTITIES + ('None',):
            cm[pred_label][true_label]  = 0

    for true_fname, pred_fname in zip(sorted_glob(true_brat_dir + '/*.ann'),
                                      sorted_glob(pred_brat_dir + '/*.ann')):
        true_annots = read_true_brat_annots(true_fname)

        for line in open(pred_fname):
            if line.startswith('T'):
                part = line.split('\t')[1]
                pred_label, start, end = part.split()
                true_label = true_annots.get((start, end), 'None')
                cm[pred_label][true_label] += 1

    l = 16 * ' '
    for true_label in ENTITIES + ('None',):
        l += '{:>16}'.format(true_label)
    print(l)

    for pred_label in ENTITIES:
        l = '{:16}'.format(pred_label)

        for true_label in ENTITIES + ('None',):
            l += '{:16}'.format(cm[pred_label][true_label])
        print(l)


confusion(
    join(DATA_DIR, 'train'),
    join(EXPS_DIR, 'best/_train/brat')
)




