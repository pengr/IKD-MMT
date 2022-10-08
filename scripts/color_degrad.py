import webcolors
import sys

colors = webcolors.CSS3_NAMES_TO_HEX

def main(argv):
    lines = []
    with open(argv[0], 'r', encoding='utf8') as inp_f:
        for l in inp_f:
            mask_l = ['<unk>' if w in colors else w for w in l.strip().split()]
            lines.append(' '.join(mask_l))

    with open(argv[1], 'w', encoding='utf8') as out_f:
        for line in lines:
            out_f.write(line+'\n')


if "__main__" == __name__:
    main(sys.argv[1:])