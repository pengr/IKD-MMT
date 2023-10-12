import sys
import random


def main(argv):
    seed = argv[0]
    random.seed(seed)  # alloc a rand seed

    lines = []
    with open(argv[1], 'r', encoding='utf8') as inp_f:
        for l in inp_f:
            lines.append(l)

    random.shuffle(lines)
    with open(argv[2], 'w', encoding='utf8') as out_f:
        for line in lines:
            out_f.write(line)


if "__main__" == __name__:
    main(sys.argv[1:])