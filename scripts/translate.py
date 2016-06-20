#!/usr/bin/env python

import os
import logging

from chainer import cuda

from ecnn import util
from ecnn import procedure
from ecnn.vocabulary import Vocab


def main(args):
    # load model
    encdec = util.load_model(args.model)

    if args.gpu is not None:
        cuda.get_device(args.gpu).use()
        encdec.to_gpu()

    # load data
    model_base_path = os.path.dirname(args.model)

    if os.path.exists(os.path.join(model_base_path, 'vocab_src')):
        vocab_src_path = os.path.join(model_base_path, 'vocab_src')
        vocab_trg_path = os.path.join(model_base_path, 'vocab_trg')
        vocab_src = Vocab.load(vocab_src_path)
        vocab_trg = Vocab.load(vocab_trg_path)
    else:
        vocab_path = os.path.join(model_base_path, 'vocab')
        vocab = Vocab.load(vocab_path)
        vocab_src = vocab
        vocab_trg = vocab
    data = util.load_sentences(args.data, vocab_src)

    # create batches
    batches = util.create_batches_src(data, args.batch, args.bucket_step)

    # generate
    res = {}
    for idx_lst, xs_data in batches:
        if args.gpu is not None:
            xs_data = cuda.to_gpu(xs_data)
        xs = procedure.create_variables(xs_data)

        ids_batch, ys, ws = encdec.generate(xs, max_len=args.max_len, sample=args.sample, temp=args.temp)
        assert len(idx_lst) == len(ids_batch)
        for idx, ids in zip(idx_lst, ids_batch):
            words = map(vocab_trg.get_word, ids)
            assert idx not in res
            res[idx] = words

    for idx, (idx2, words) in enumerate(sorted(res.items(), key=lambda k_v: k_v[0])):
        assert idx == idx2
        print(' '.join(words))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model.')

    parser.add_argument('model', help='model path')
    parser.add_argument('data', help='data path')

    parser.add_argument('--max-len', type=int, default=50, help='maximum length of generation')
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: use CPU)')
    parser.add_argument('--bucket-step', type=int, default=4, help='step size for padding of source')

    parser.add_argument('--sample', action='store_true', help='sampling-based generation')
    parser.add_argument('--temp', type=float, default=1., help='temperature of softmax for sampling')

    main(parser.parse_args())
