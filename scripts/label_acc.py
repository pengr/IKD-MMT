# -*- coding: utf-8 -*-
import re
import sys

from ntlk_metrics import precision, recall, f_measure

SPACE_NORMALIZER = re.compile(r"\s+")
def tokenize_line(line):
    line = SPACE_NORMALIZER.sub(" ", line)
    line = line.strip()
    return line.split()


def label_acc(argv):
    label_acc = 0
    label_prec = 0
    label_recl = 0
    label_fmea = 0
    num_sents = 0

    with open(argv[0], encoding='utf8') as ref_f, \
         open(argv[1], encoding='utf8') as pred_f:

        for pred_l, ref_l in zip(pred_f, ref_f):
            pred_l, ref_l = tokenize_line(pred_l), tokenize_line(ref_l)
            # label_acc += accuracy(ref_l, pred_l)
            reference_set = set(ref_l)
            test_set = set(pred_l)
            label_prec += precision(reference_set, test_set)
            label_recl += recall(reference_set, test_set)
            label_fmea += f_measure(reference_set, test_set)
            num_sents += 1

    # label_acc /= num_sents
    label_prec /= num_sents
    label_recl /= num_sents
    label_fmea /= num_sents
    # print('label accuracy: %.2f %%' % (label_acc * 100))
    print('label precision: %.2f %%' % (label_prec * 100))
    # print('label recall: %.2f %%' % (label_recl * 100))
    # print('label f-measure: %.2f %%' % (label_fmea * 100))


if __name__ == "__main__":
    label_acc(sys.argv[1:])