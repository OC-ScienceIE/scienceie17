"""
Postprocessing on Brat files

Another idea for post-processing on the Brat annotaion.
Unfortunately, it does not improve the scores.

Original scores on dev data

           precision   recall f1-score  support

   Material     0.48     0.47     0.47      562
    Process     0.38     0.34     0.36      455
       Task     0.17     0.12     0.14      137

avg / total     0.41     0.38     0.39     1154x`


Post-processed scores on dev data

           precision   recall f1-score  support

   Material     0.43     0.50     0.46      562
    Process     0.36     0.35     0.35      455
       Task     0.17     0.12     0.14      137

avg / total     0.38     0.39     0.39     1154



"""

from os.path import join, basename
from os import makedirs
from collections import defaultdict, Counter
import re

from eval import calculateMeasures
from sie import EXPS_DIR, DATA_DIR
from sie.brat import Span, read_brat_file, write_brat_file
from sie.utils import sorted_glob





def postproc_brat(in_brat_dir, txt_dir, out_brat_dir):
    makedirs(out_brat_dir, exist_ok=True)

    for in_brat_fname in sorted_glob(join(in_brat_dir, '*.ann')):
        spans = read_brat_file(in_brat_fname)
        txt_fname = join(txt_dir, basename(in_brat_fname).replace('.ann', '.txt'))
        print('reading ' + txt_fname)
        text = open(txt_fname).read()
        phrase2annots = get_phrase_annots(spans, text)

        for phrase, annots in phrase2annots.items():
            counts = Counter(span.label for span in annots)
            most_common = counts.most_common()
            if len(most_common) > 1:
                if most_common[0][1] > most_common[1][1]:
                    print('--> found majority label for phrase',  repr(phrase), ':', counts)
                    majority_label = most_common[0][0]
                    for span in annots:
                        if span.label != majority_label:
                            print('==> removing', span)
                            annots.remove(span)
                            spans.remove(span)
                else:
                    best_label = 'Material' if 'Material' in counts else 'Process'
                    print('--> found best label for phrase',  repr(phrase), ':', counts)
                    for span in annots:
                        if span.label != best_label:
                            print('==> removing', span)
                            annots.remove(span)
                            spans.remove(span)

            # now all labels in annots are the same
            unique_label = annots[0].label
            for m in re.finditer(re.escape(phrase), text):
                try:
                    if text[m.start()-1].isalpha():
                        continue
                except IndexError:
                    pass

                try:
                    if text[m.end()].isalpha():
                        continue
                except IndexError:
                    pass

                span = Span(unique_label, m.start(), m.end())
                if span not in annots:
                    print(annots)
                    print('==> adding span',span, 'for phrase', repr(phrase))
                    spans.append(span)

        out_brat_fname = join(out_brat_dir, basename(in_brat_fname))
        write_brat_file(out_brat_fname, spans, text)



def get_phrase_annots(spans, text):
    phrase2annots = defaultdict(list)
    for span in spans:
        phrase = text[span.begin:span.end]
        phrase2annots[phrase].append(span)
    return phrase2annots


in_brat_dir = join(EXPS_DIR, 'best/_dev/brat')
txt_dir = join(DATA_DIR, 'dev')
out_brat_dir = '_dev/brat'


postproc_brat(in_brat_dir, txt_dir, out_brat_dir)


calculateMeasures(txt_dir, in_brat_dir, 'rel')
calculateMeasures(txt_dir, out_brat_dir, 'rel')
