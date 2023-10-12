#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Data pre-processing: build vocabularies and binarize training data.
"""

from collections import Counter
from itertools import zip_longest
import logging
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
import os
import shutil
import sys

from fairseq import options, tasks, utils
from fairseq.data import indexed_dataset
from fairseq.binarizer import Binarizer
import multiprocessing

# define a logger
logging.basicConfig(
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO,
    stream=sys.stdout,
)
logger = logging.getLogger('fairseq_cli.preprocess')


def main(args):
    utils.import_user_module(args)  # load additional custom modules if "--user-dir" isn't None

    os.makedirs(args.destdir, exist_ok=True)  # confirm that the output directory exists

    logger.addHandler(logging.FileHandler(
        filename=os.path.join(args.destdir, 'preprocess.log'),
    ))                 # store all arguments into destdir/preprocess.log
    logger.info(args)  # print all arguments

    task = tasks.get_task(args.task)  # take out the python file and class by args.task

    def train_path(lang):
        return "{}{}".format(args.trainpref, ("." + lang) if lang else "")

    def file_name(prefix, lang):
        fname = prefix
        if lang is not None:
            fname += ".{lang}".format(lang=lang)
        return fname

    def dest_path(prefix, lang):
        return os.path.join(args.destdir, file_name(prefix, lang))

    def dict_path(lang):
        return dest_path("dict", lang) + ".txt"

    # filenames is the train.src/tgt path, and src/tgt=True determines if the src-side or tgt-side is processed
    def build_dictionary(filenames, src=False, tgt=False):
        assert src ^ tgt  # OR operator
        return task.build_dictionary(
            filenames,
            workers=args.workers,
            threshold=args.thresholdsrc if src else args.thresholdtgt,
            nwords=args.nwordssrc if src else args.nwordstgt,
            padding_factor=args.padding_factor,
        )

    target = not args.only_source  # not only process the source language

    # If the src/tgt dictionary path is not set, and the src/tgt dictionary text dict.lang.txt exists, report FileExistsError
    if not args.srcdict and os.path.exists(dict_path(args.source_lang)):
        raise FileExistsError(dict_path(args.source_lang))
    if target and not args.tgtdict and os.path.exists(dict_path(args.target_lang)):
        raise FileExistsError(dict_path(args.target_lang))

    # Generate joined dictionary
    if args.joined_dictionary:
        assert not args.srcdict or not args.tgtdict, \
            "cannot use both --srcdict and --tgtdict with --joined-dictionary"

        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        elif args.tgtdict:
            src_dict = task.load_dictionary(args.tgtdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary(
                {train_path(lang) for lang in [args.source_lang, args.target_lang]}, src=True
            )
        tgt_dict = src_dict
    else:
        # Generate source and target dictionary
        # if source and target dictionary are given, choose it
        # else build_dictionary from train.src and train.tgt
        if args.srcdict:
            src_dict = task.load_dictionary(args.srcdict)
        else:
            assert args.trainpref, "--trainpref must be set if --srcdict is not specified"
            src_dict = build_dictionary([train_path(args.source_lang)], src=True)

        if target:
            if args.tgtdict:
                tgt_dict = task.load_dictionary(args.tgtdict)
            else:
                assert args.trainpref, "--trainpref must be set if --tgtdict is not specified"
                tgt_dict = build_dictionary([train_path(args.target_lang)], tgt=True)
        else:
            tgt_dict = None

    src_dict.save(dict_path(args.source_lang))
    if target and tgt_dict is not None:
        tgt_dict.save(dict_path(args.target_lang))

    # <<fix>>
    def make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers):
        logger.info("[{}] Dictionary: {} types".format(lang, len(vocab) - 1))
        n_seq_tok = [0, 0]
        replaced = Counter()

        def merge_result(worker_result):
            replaced.update(worker_result["replaced"])
            n_seq_tok[0] += worker_result["nseq"]
            n_seq_tok[1] += worker_result["ntok"]

        input_file = "{}{}".format(
            input_prefix, ("." + lang) if lang is not None else ""
        )
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize,
                    (
                        args,
                        input_file,
                        vocab,
                        prefix,
                        lang,
                        offsets[worker_id],
                        offsets[worker_id + 1],
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                          impl=args.dataset_impl, vocab_size=len(vocab))
        merge_result(
            Binarizer.binarize(
                input_file, vocab, lambda t: ds.add_item(t),
                offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, lang)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))

        logger.info(
            "[{}] {}: {} sents, {} tokens, {:.3}% replaced by {}".format(
                lang,
                input_file,
                n_seq_tok[0],
                n_seq_tok[1],
                100 * sum(replaced.values()) / n_seq_tok[1],
                vocab.unk_word,
            )
        )

    # <<fix>>, only one thread
    def make_binary_image_dataset(images_fnames, images_path, output_prefix, num_workers):
        print("preprocesing image info......")
        num_workers = 1
        nseq = [0]

        # build the specified pre-trained cnn
        from PretrainedCNNModels import PretrainedDisCNN
        def build_pretrained_cnn(pretrained_cnn_name):
            """ Uses pytorch/cadene to load pre-trained CNN. """
            cnn = PretrainedDisCNN(pretrained_cnn_name)
            cnn.model = cnn.model.cuda()
            return cnn
        pretrained_cnn = build_pretrained_cnn(args.pretrained_cnn)

        def merge_result(worker_result):
            nseq[0] += worker_result['nseq']

        offsets = Binarizer.find_offsets(images_fnames, num_workers)
        pool = None
        if num_workers > 1:
            pool = ThreadPool(processes=num_workers - 1)  # <<fix>>, pool
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize_images,
                    (
                        args,
                        images_fnames,
                        images_path,
                        pretrained_cnn.load_image_from_path,
                        # utils.parse_image,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()


        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"),
                                          impl=args.dataset_impl, image=True)
        merge_result(
            Binarizer.binarize_images(
                images_fnames, images_path, pretrained_cnn.load_image_from_path, lambda t: ds.add_item(t),
                offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, None)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))

        logger.info(
            "[imgae] {}: parsed {} images".format(
                images_fnames,
                nseq[0]
            )
        )

    def make_binary_alignment_dataset(input_prefix, output_prefix, num_workers):
        nseq = [0]

        def merge_result(worker_result):
            nseq[0] += worker_result['nseq']

        input_file = input_prefix
        offsets = Binarizer.find_offsets(input_file, num_workers)
        pool = None
        if num_workers > 1:
            pool = Pool(processes=num_workers - 1)
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                pool.apply_async(
                    binarize_alignments,
                    (
                        args,
                        input_file,
                        utils.parse_alignment,
                        prefix,
                        offsets[worker_id],
                        offsets[worker_id + 1]
                    ),
                    callback=merge_result
                )
            pool.close()

        ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"),
                                          impl=args.dataset_impl)

        merge_result(
            Binarizer.binarize_alignments(
                input_file, utils.parse_alignment, lambda t: ds.add_item(t),
                offset=0, end=offsets[1]
            )
        )
        if num_workers > 1:
            pool.join()
            for worker_id in range(1, num_workers):
                prefix = "{}{}".format(output_prefix, worker_id)
                temp_file_path = dataset_dest_prefix(args, prefix, None)
                ds.merge_file_(temp_file_path)
                os.remove(indexed_dataset.data_file_path(temp_file_path))
                os.remove(indexed_dataset.index_file_path(temp_file_path))

        ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))

        logger.info(
            "[alignments] {}: parsed {} alignments".format(
                input_file,
                nseq[0]
            )
        )

    # <<fix>>
    def make_dataset(vocab, input_prefix, output_prefix, lang, num_workers=1):
        if args.dataset_impl == "raw":
            # Copy original text file to destination folder
            output_text_file = dest_path(
                output_prefix + ".{}-{}".format(args.source_lang, args.target_lang),
                lang,
            )
            shutil.copyfile(file_name(input_prefix, lang), output_text_file)
        else:
            make_binary_dataset(vocab, input_prefix, output_prefix, lang, num_workers)

    # <<fix>>
    def make_all(lang, vocab):
        if args.trainpref:
            make_dataset(vocab, args.trainpref, "train", lang, num_workers=args.workers)
        if args.validpref:
            for k, validpref in enumerate(args.validpref.split(",")):
                outprefix = "valid{}".format(k) if k > 0 else "valid"
                make_dataset(vocab, validpref, outprefix, lang, num_workers=args.workers)
        if args.testpref:
            for k, testpref in enumerate(args.testpref.split(",")):
                outprefix = "test{}".format(k) if k > 0 else "test"
                make_dataset(vocab, testpref, outprefix, lang, num_workers=args.workers)

    # <<fix>>
    def make_all_images(image_suffix):
        if args.train_fnames:
            make_binary_image_dataset(args.train_fnames, args.images_path, f"train.{image_suffix}", num_workers=args.workers)
        if args.valid_fnames:
            for k, valid_fnames in enumerate(args.valid_fnames.split(",")):
                make_binary_image_dataset(valid_fnames, args.images_path, f"valid{k}.{image_suffix}" if k > 0 else f"valid.{image_suffix}", num_workers=args.workers)
        if args.test_fnames:
            for k, test_fnames in enumerate(args.test_fnames.split(",")):
                make_binary_image_dataset(test_fnames, args.images_path, f"test{k}.{image_suffix}" if k > 0 else f"test.{image_suffix}", num_workers=args.workers)

    def make_all_alignments():
        if args.trainpref and os.path.exists(args.trainpref + "." + args.align_suffix):
            make_binary_alignment_dataset(args.trainpref + "." + args.align_suffix, "train.align", num_workers=args.workers)
        if args.validpref and os.path.exists(args.validpref + "." + args.align_suffix):
            make_binary_alignment_dataset(args.validpref + "." + args.align_suffix, "valid.align", num_workers=args.workers)
        if args.testpref and os.path.exists(args.testpref + "." + args.align_suffix):
            make_binary_alignment_dataset(args.testpref + "." + args.align_suffix, "test.align", num_workers=args.workers)

    make_all(args.source_lang, src_dict)
    if target:
        make_all(args.target_lang, tgt_dict)

    # <<fix>>
    assert os.path.exists(args.images_path)
    make_all_images(args.image_suffix)  # <<fix>>

    if args.align_suffix:
        make_all_alignments()

    logger.info("Wrote preprocessed data to {}".format(args.destdir))

    if args.alignfile:
        assert args.trainpref, "--trainpref must be set if --alignfile is specified"
        src_file_name = train_path(args.source_lang)
        tgt_file_name = train_path(args.target_lang)
        freq_map = {}
        with open(args.alignfile, "r", encoding='utf-8') as align_file:
            with open(src_file_name, "r", encoding='utf-8') as src_file:
                with open(tgt_file_name, "r", encoding='utf-8') as tgt_file:
                    for a, s, t in zip_longest(align_file, src_file, tgt_file):
                        si = src_dict.encode_line(s, add_if_not_exist=False)
                        ti = tgt_dict.encode_line(t, add_if_not_exist=False)
                        ai = list(map(lambda x: tuple(x.split("-")), a.split()))
                        for sai, tai in ai:
                            srcidx = si[int(sai)]
                            tgtidx = ti[int(tai)]
                            if srcidx != src_dict.unk() and tgtidx != tgt_dict.unk():
                                assert srcidx != src_dict.pad()
                                assert srcidx != src_dict.eos()
                                assert tgtidx != tgt_dict.pad()
                                assert tgtidx != tgt_dict.eos()

                                if srcidx not in freq_map:
                                    freq_map[srcidx] = {}
                                if tgtidx not in freq_map[srcidx]:
                                    freq_map[srcidx][tgtidx] = 1
                                else:
                                    freq_map[srcidx][tgtidx] += 1

        align_dict = {}
        for srcidx in freq_map.keys():
            align_dict[srcidx] = max(freq_map[srcidx], key=freq_map[srcidx].get)

        with open(
                os.path.join(
                    args.destdir,
                    "alignment.{}-{}.txt".format(args.source_lang, args.target_lang),
                ),
                "w", encoding='utf-8'
        ) as f:
            for k, v in align_dict.items():
                print("{} {}".format(src_dict[k], tgt_dict[v]), file=f)


def binarize(args, filename, vocab, output_prefix, lang, offset, end, append_eos=True):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, lang, "bin"),
                                      impl=args.dataset_impl, vocab_size=len(vocab))

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize(filename, vocab, consumer, append_eos=append_eos,
                             offset=offset, end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, lang, "idx"))
    return res


# <<fix>>
def binarize_images(args, filename, path, parse_image, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"),
                                      impl=args.dataset_impl, image=True)

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_images(filename, path, parse_image, consumer, offset=offset,
                                     end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
    return res


def binarize_alignments(args, filename, parse_alignment, output_prefix, offset, end):
    ds = indexed_dataset.make_builder(dataset_dest_file(args, output_prefix, None, "bin"),
                                      impl=args.dataset_impl, vocab_size=None)

    def consumer(tensor):
        ds.add_item(tensor)

    res = Binarizer.binarize_alignments(filename, parse_alignment, consumer, offset=offset,
                                        end=end)
    ds.finalize(dataset_dest_file(args, output_prefix, None, "idx"))
    return res


def dataset_dest_prefix(args, output_prefix, lang):
    base = "{}/{}".format(args.destdir, output_prefix)
    if lang is not None:
        lang_part = ".{}-{}.{}".format(args.source_lang, args.target_lang, lang)
    elif args.only_source:
        lang_part = ""
    else:
        lang_part = ".{}-{}".format(args.source_lang, args.target_lang)

    return "{}{}".format(base, lang_part)


def dataset_dest_file(args, output_prefix, lang, extension):
    base = dataset_dest_prefix(args, output_prefix, lang)
    return "{}.{}".format(base, extension)


def get_offsets(input_file, num_workers):
    return Binarizer.find_offsets(input_file, num_workers)


def cli_main():
    parser = options.get_preprocessing_parser()  # get preprocess configuration according to specified task
    args = parser.parse_args()
    main(args)


if __name__ == "__main__":
    multiprocessing.set_start_method('spawn', force=True)
    cli_main()
