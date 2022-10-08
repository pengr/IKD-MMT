#!/usr/bin/env python3 -u
# Copyright (c) RuPeng, Inc. and its affiliates.

import argparse
import gc
import html
import time
from itertools import islice
from typing import List

import stanza
import toma
from stanza.models.common.doc import Document

from stanza_batch import batch


# toma requires the first argument of the method to be the batch size
def run_batch(batch_size: int, stanza_nlp: stanza.Pipeline, data: List[str]
              ) -> List[Document]:
    # So that we can see what the batch size changes to.
    print(batch_size)
    return [doc for doc in batch(data, stanza_nlp, batch_size=batch_size)]


# no aggressive hyphen splitting, no-escape and segmentation
mapper = {'"': '``'}
def tokenize(sent):
    sent = sent.replace("@-@", "-")
    tokens = html.unescape(sent.strip()).split()
    tokens = list(map(lambda t: mapper.get(t, t), tokens))
    return tokens


def reconst_orig(sent, separator='@@'):
    tokens = tokenize(sent)
    word, words = [], []
    for tok in tokens:
        if tok.endswith(separator):
            tok = tok.strip(separator)
            word.append(tok)
        else:
            word.append(tok)
            words.append(''.join(word))
            word = []
    sentence = ' '.join(words)
    return sentence


# Split the corpus into multiple shards
def segment_corpus(sents, shard_size):
    if shard_size <= 0:   # didn't divided
        yield sents
    else:
        sents = iter(sents)
        while True:      # Divide dataset once through the loop, and return each dataset that has been divided
            shard = list(islice(sents, shard_size))
            if not shard:
                return
            yield shard


# POS tagging (default: universal POS (UPOS) tags, instead of treebank-specific POS (XPOS) tags)
# https://universaldependencies.org/u/pos/
def pos(docs):
    sents_labels = []
    for i, doc in enumerate(docs):
        for sent in doc.sentences:
            labels = [word.text for word in sent.words if word.pos in ['NOUN', 'PROPN']]
            sents_labels.append(labels)

    return sents_labels


def make_dataset(src_nlp, tgt_nlp, input_prefix, output_suffix, lang):
    src, tgt = lang
    output_file = input_prefix + f'.{output_suffix}'

    src_labels = []
    tgt_labels = []

    with open(input_prefix+'.'+src, 'r', encoding='utf8') as src_f, \
         open(input_prefix+'.'+tgt, 'r', encoding='utf8') as tgt_f, \
         open(output_file, 'w', encoding='utf8') as out_f:

        src_sents = src_f.readlines()
        tgt_sents = tgt_f.readlines()

        # Prevent too many sentences
        src_shards = segment_corpus(src_sents, args.shard_size)
        tgt_shards = segment_corpus(tgt_sents, args.shard_size)

        # Stanza process  bacth by batch
        for src_shard, tgt_shard in zip(src_shards, tgt_shards):

            # Step1. no aggressive hyphen splitting, no-escape and segmentation, remove bpe
            if not args.no_reconst_orig:
                src_shard = list(map(reconst_orig, src_shard))
                tgt_shard = list(map(reconst_orig, tgt_shard))
                print('Reconst_orig Done')

            # step2. POS Tagging
            if not args.no_pos:
                # Create the document batch
                src_docs: List[Document] = toma.simple.batch(run_batch, args.batch_size, src_nlp, src_shard)
                src_shard = pos(src_docs)
                del src_docs
                gc.collect()

                # Create the document batch
                tgt_docs: List[Document] = toma.simple.batch(run_batch, args.batch_size, tgt_nlp, tgt_shard)
                tgt_shard = pos(tgt_docs)
                del tgt_docs
                gc.collect()
                print('POS Done')

            # step3. save the src and tgt labels
            src_labels += src_shard
            tgt_labels += tgt_shard

        # Step 11. Write Gloss Sentences to file
        for src_label, tgt_label in zip(src_labels, tgt_labels):
            out_f.write(' '.join(src_label + tgt_label) +'\n')


def main(args):
    src, tgt = args.lang

    # load source and target pos model
    # stanza.download(src)
    src_nlp = stanza.Pipeline(src, r"../stanza_resources", processors='tokenize,mwt,pos',
                              tokenize_pretokenized=True, verbose=False)
    tgt_nlp = stanza.Pipeline(tgt, r"../stanza_resources", processors='tokenize,mwt,pos',
                              tokenize_pretokenized=True, verbose=False)

    if args.trainpref:
        make_dataset(src_nlp, tgt_nlp, args.trainpref, 'label', args.lang)
    if args.validpref:
        for k, validpref in enumerate(args.validpref.split(",")):
            make_dataset(src_nlp, tgt_nlp, validpref, 'label', args.lang)
    if args.testpref:
        for k, testpref in enumerate(args.testpref.split(",")):
            make_dataset(src_nlp, tgt_nlp, testpref, 'label', args.lang)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="extract both source and target langugae labels \
                                                  corresponding to the image")
    parser.add_argument('--lang', "--language",type=str, nargs='+',
                        help="source and target languages")

    # the prefixes of train, valid and test files
    parser.add_argument("--trainpref", metavar="FP", default=None,
                        help="train file prefix")
    parser.add_argument("--validpref", metavar="FP", default=None,
                        help="comma separated, valid file prefixes")
    parser.add_argument("--testpref", metavar="FP", default=None,
                        help="comma separated, test file prefixes")

    # the settings of pos tagging and stanza
    parser.add_argument('--shard-size', type=int, metavar='D', default=1000000,
                        help="Divide corpus into smaller multiple corpus files,"
                             "shard_size>0 means segment dataset into multiple shards")
    parser.add_argument('--no-reconst-orig', default=False, action="store_true",
                        help='Attempt reconstructing original data'
                             '(no aggressive hyphen splitting, no-escape and segmentation, remove bpe)')
    parser.add_argument('--no-pos', default=False, action="store_true", help='part-of-speeching')
    parser.add_argument('--batch-size', type=int, metavar='D', default=1024,
                        help='Batch size for text processing in stanza (sentences/per batch)')

    args = parser.parse_args()

    start_time = time.time()
    main(args)
    print('Finished! Cost {}s'.format(time.time() - start_time))