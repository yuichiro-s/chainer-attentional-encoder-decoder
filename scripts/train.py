#!/usr/bin/env python

import os
import logging

from chainer import cuda

from ecnn.nn.attentional_encoder_decoder import AttentionalEncoderDecoder
from ecnn import util
from ecnn import vocabulary
from ecnn.procedure import train_model, evaluate_model


def main(args):
    if args.gpu is not None:
        cuda.get_device(args.gpu).use()

    os.makedirs(args.model)

    # set up logger
    logger = logging.getLogger()
    logging.basicConfig(level=logging.INFO)
    log_path = os.path.join(args.model, 'log')
    file_handler = logging.FileHandler(log_path)
    fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    # set up optimizer
    optimizer = util.list2optimizer(args.optim)

    # load data
    logger.info('Loading vocabulary...')
    vocab_trg = vocabulary.Vocab.load(args.vocab, args.vocab_size)
    vocab_src = vocabulary.Vocab.load(args.src_vocab, args.src_vocab_size) if args.src_vocab else vocab_trg
    logger.info('Loading training data...')
    train_data = util.load_bitext(args.train, vocab_src, vocab_trg)

    # save vocabulary
    if args.src_vocab:
        vocab_src.save(os.path.join(args.model, 'vocab_src'))
        vocab_trg.save(os.path.join(args.model, 'vocab_trg'))
    else:
        vocab_trg.save(os.path.join(args.model, 'vocab'))

    # save hyperparameters
    with open(os.path.join(args.model, 'params'), 'w') as f:
        for k, v in vars(args).items():
            print('{}\t{}'.format(k, v), file=f)

    # create batches
    logger.info('Creating batches...')
    train_batches = util.create_batches(train_data, args.batch, args.src_bucket_step, args.trg_bucket_step)

    # create model
    vocab_size_src = vocab_src.size()
    vocab_size_trg = vocab_trg.size()
    layer_num = args.layer
    hidden_dim = args.hidden
    bidirectional = not args.bidirectional
    pyramidal = not args.pyramidal
    encdec = AttentionalEncoderDecoder(vocab_size_trg, hidden_dim, layer_num, vocab_size_trg, bidirectional, pyramidal,
                                       src_vocab_size=vocab_size_src)

    if args.dev:
        dev_data = util.load_bitext(args.dev, vocab_src, vocab_trg)
        dev_batches = util.create_batches(dev_data, args.batch, args.src_bucket_step, args.trg_bucket_step)

        def epoch_end_func():
            evaluate_model(encdec, dev_batches, args.gpu)
    else:
        epoch_end_func = None

    train_model(encdec, train_batches, optimizer, args.model,
                max_epoch=args.epoch, gpu=args.gpu, save_every=args.save_every,
                epoch_end_func=epoch_end_func)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train a model.')

    parser.add_argument('model', help='model path')
    parser.add_argument('train', help='training data path')
    parser.add_argument('vocab', help='vocabulary path')

    # data
    parser.add_argument('--vocab-size', type=int, default=-1, help='vocabulary size (default: use all words)')
    parser.add_argument('--src-vocab', help='source vocabulary path (use different vocabulary for source)')
    parser.add_argument('--src-vocab-size', type=int, default=-1, help='source vocabulary size (use different vocabulary for source)')
    parser.add_argument('--dev', help='development data path')

    # NN architecture
    parser.add_argument('--hidden', type=int, default=128, help='size of hidden layer')
    parser.add_argument('--layer', type=int, default=-1, help='vocabulary size (default: use all words)')
    parser.add_argument('--gru', action='store_true', help='use GRU instead of LSTM')
    parser.add_argument('--bidirectional', action='store_true', help='use bidirectional RNNs for encoder')
    parser.add_argument('--pyramidal', action='store_true', help='use pyramidal structure for encoder')

    # training options
    parser.add_argument('--batch', type=int, default=32, help='batch size')
    parser.add_argument('--src-bucket-step', type=int, default=4, help='step size for padding of source')
    parser.add_argument('--trg-bucket-step', type=int, default=8, help='step size for padding of target')
    parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--optim', nargs='+', default=['Adam'], help='optimization method')
    parser.add_argument('--gpu', type=int, default=None, help='GPU ID (default: use CPU)')
    parser.add_argument('--save-every', type=int, default=1, help='save model every this number of epochs')

    main(parser.parse_args())
