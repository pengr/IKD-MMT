#!/usr/bin/env python3 -u
# Copyright (c) RuPeng, Inc. and its affiliates.
import argparse, time


def make_dataset(input_prefix, output_suffix, src, offsets=0):
    output_file = input_prefix + f'.{output_suffix}'

    with open(input_prefix+'.'+src, 'r', encoding='utf8') as src_f, \
         open(output_file, 'w', encoding='utf8') as out_f:

        num_sents = len(src_f.readlines())

        # Write image index to file
        for index in range(offsets, num_sents+offsets):
            out_f.write(str(index) +'\n')

        # update current image index after finish the file
        offsets += num_sents

        return offsets

def main(args):
    if args.trainpref:
        offsets = make_dataset(args.trainpref, 'image', args.src, offsets=1)
    if args.validpref:
        for k, validpref in enumerate(args.validpref.split(",")):
            offsets = make_dataset(validpref, 'image', args.src, offsets=offsets)
    if args.testpref:
        for k, testpref in enumerate(args.testpref.split(",")):
            offsets = make_dataset(testpref, 'image', args.src, offsets=offsets)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="create image indexs corresponding to the image features")
    parser.add_argument("--src", "--source-lang", default=None, metavar="SRC",
                       help="source language")

    # the prefixes of train, valid and test files
    parser.add_argument("--trainpref", metavar="FP", default=None,
                        help="train file prefix")
    parser.add_argument("--validpref", metavar="FP", default=None,
                        help="comma separated, valid file prefixes")
    parser.add_argument("--testpref", metavar="FP", default=None,
                        help="comma separated, test file prefixes")

    args = parser.parse_args()

    start_time = time.time()
    main(args)
    print('Finished! Cost {}s'.format(time.time() - start_time))