#!/usr/bin/env python3 -u
# Copyright (c) RuPeng, Inc. and its affiliates.

import argparse
import gc
from typing import List

import stanza
import toma
from stanza.models.common.doc import Document
from stanza_batch import batch


# toma requires the first argument of the method to be the batch size
def run_batch(batch_size: int, stanza_nlp: stanza.Pipeline, data: List[str]) -> List[Document]:
    # So that we can see what the batch size changes to.
    return [doc for doc in batch(data, stanza_nlp, batch_size=batch_size)]


# POS tagging (default: universal POS (UPOS) tags, instead of treebank-specific POS (XPOS) tags)
# https://universaldependencies.org/u/pos/
def pos(docs):
    sents = []
    for i, doc in enumerate(docs):
        for sent in doc.sentences:
            mask_sent = ['<unk>' if word.pos in ['NOUN'] else word.text for word in sent.words]
            sents.append(mask_sent)
    return sents


def make_dataset(nlp, file):
    with open(file, 'r', encoding='utf8') as f:
        sents = f.readlines()

        # POS Tagging
        src_docs: List[Document] = toma.simple.batch(run_batch, args.batch_size, nlp, sents) # Create the document batch
        mask_sents = pos(src_docs)
        del src_docs
        gc.collect()

    with open(file, 'w', encoding='utf8') as f:
        for sent in mask_sents:
            f.write(' '.join(sent) + '\n')


def main(args):
    # load English pos model
    # stanza.download('en')
    en_nlp = stanza.Pipeline('en', r"../stanza_resources", processors='tokenize,mwt,pos',
                              tokenize_pretokenized=True, verbose=False)
    make_dataset(en_nlp, args.train)
    for k, valid in enumerate(args.valid.split(",")):
        make_dataset(en_nlp, valid)
    for k, test in enumerate(args.test.split(",")):
        make_dataset(en_nlp, test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="entity masking")

    # source files of train, valid and test
    parser.add_argument("--train", metavar="FP", default=None,
                        help="train file")
    parser.add_argument("--valid", metavar="FP", default=None,
                        help="valid file")
    parser.add_argument("--test", metavar="FP", default=None,
                        help="test file")

    # the settings of pos tagging and stanza
    parser.add_argument('--no-pos', default=False, action="store_true", help='part-of-speeching')
    parser.add_argument('--batch-size', type=int, metavar='D', default=1024,
                        help='Batch size for text processing in stanza (sentences/per batch)')

    args = parser.parse_args()
    main(args)