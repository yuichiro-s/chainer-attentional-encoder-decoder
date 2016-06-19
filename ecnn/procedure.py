import time
import logging
from collections import OrderedDict
import random

from chainer import cuda
from chainer import Variable
import chainer.functions as F

from ecnn.vocabulary import IGNORE_ID, UNK_ID
from ecnn import util


def train_model(model, batches, optimizer, dest_dir, max_epoch=None, gpu=None, save_every=1, epoch_end_func=None):
    """Common training procedure.
    :param model: model to train
    :param batches: training data
    :param optimizer: chainer optimizer
    :param dest_dir: destination directory
    :param max_epoch: maximum number of epochs to train (None to train indefinitely)
    :param gpu: ID of GPU (None to use CPU)
    :param save_every: save every this number of epochs (first epoch and last epoch are always saved)
    """
    if gpu is not None:
        # set up GPU
        model.to_gpu()

    logger = logging.getLogger()
    n_batches = len(batches)

    util.save_model_def(model, dest_dir)

    # set up optimizer
    optimizer.setup(model)

    # training loop
    epoch = 0
    while max_epoch is None or epoch < max_epoch:
        time_start = time.time()

        ok = 0
        tot = 0
        loss_tot = 0
        sample_num = 0
        random.shuffle(batches)
        for i, batch in enumerate(batches):
            try:
                xs_data, ts_data = batch
                x_len, batch_size = xs_data.shape
                t_len = ts_data.shape[0]
                sample_num += batch_size

                # copy data to GPU
                t1 = time.time()
                if gpu is not None:
                                       xs_data = cuda.to_gpu(xs_data)
                                       ts_data = cuda.to_gpu(ts_data)

                # create variable
                xs = _create_variables(xs_data)
                ts = _create_variables(ts_data)

                t2 = time.time()
                optimizer.zero_grads()
                loss, ys, ws = model(xs, ts)
                loss_tot += loss.data * batch_size

                t3 = time.time()
                loss.backward()

                t4 = time.time()
                optimizer.update()

                t5 = time.time()

                assert len(ys) == t_len
                acc = sum(map(lambda y_t: F.accuracy(y_t[0], y_t[1], ignore_label=IGNORE_ID).data, zip(ys, ts))) / t_len
                tot += batch_size * t_len
                ok += acc * batch_size * t_len

                # report training status
                status = OrderedDict()
                status['type'] = 'TRAIN'
                status['epoch'] = epoch
                status['batch'] = i
                status['prog'] = '{:.1%}'.format(float(i + 1) / n_batches)
                status['loss'] = '{:.4}'.format(float(loss.data))  # training loss
                status['loss_avg'] = '{:.4}'.format(float(loss.data / t_len))
                status['acc'] = '{:.1%}'.format(float(acc))  # training accuracy
                status['size'] = batch_size
                status['x_len'] = x_len
                status['t_len'] = t_len
                status['unk_src'] = '{:.1%}'.format(_unk_ratio(xs_data))
                status['unk_trg'] = '{:.1%}'.format(_unk_ratio(ts_data))
                status['time'] = int((t5 - t1) * 1000)
                #status['time_m'] = int((t2 - t1) * 1000)
                status['time_f'] = int((t3 - t2) * 1000)
                status['time_b'] = int((t4 - t3) * 1000)
                #status['time_u'] = int((t5 - t4) * 1000)
                logger.info(_status_str(status))

            except Exception as e:
                logger.warn(e)
                time.sleep(1)

        time_end = time.time()
        acc = ok / tot
        # report results
        report = OrderedDict()
        report['type'] = 'TRAIN_DONE'
        status['acc'] = '{:.1%}'.format(float(acc))
        report['correct'] = int(ok)
        report['total'] = tot
        report['loss'] = '{:.4}'.format(float(loss_tot))
        report['samples'] = sample_num
        report['time'] = int(time_end - time_start)
        logger.info(_status_str(report))

        if epoch_end_func is not None:
            epoch_end_func()

        # save model
        if epoch % save_every == 0 or (max_epoch is not None and epoch == max_epoch - 1):
            util.save_model(model, dest_dir, epoch)

        epoch += 1


def evaluate_model(model, batches, gpu=None):
    if gpu is not None:
        # set up GPU
        model.to_gpu()

    logger = logging.getLogger()
    n_batches = len(batches)

    time_start = time.time()

    ok = 0
    tot = 0
    loss_tot = 0
    sample_num = 0
    for i, batch in enumerate(batches):
        xs_data, ts_data = batch
        x_len, batch_size = xs_data.shape
        t_len = ts_data.shape[0]
        sample_num += batch_size

        # copy data to GPU
        t1 = time.time()
        if gpu is not None:
            xs_data = cuda.to_gpu(xs_data)
            ts_data = cuda.to_gpu(ts_data)

        # create variable
        xs = _create_variables(xs_data)
        ts = _create_variables(ts_data)

        t2 = time.time()
        loss, ys, ws = model(xs, ts)
        loss_tot += loss.data * batch_size

        t3 = time.time()

        assert len(ys) == t_len
        acc = sum(map(lambda y_t: F.accuracy(y_t[0], y_t[1], ignore_label=IGNORE_ID).data, zip(ys, ts))) / t_len
        tot += batch_size * t_len
        ok += acc * batch_size * t_len

        # report training status
        status = OrderedDict()
        status['type'] = 'EVAL'
        status['batch'] = i
        status['prog'] = '{:.1%}'.format(float(i + 1) / n_batches)
        status['loss'] = '{:.4}'.format(float(loss.data))
        status['loss_avg'] = '{:.4}'.format(float(loss.data / t_len))
        status['acc'] = '{:.1%}'.format(float(acc))
        status['size'] = batch_size
        status['x_len'] = x_len
        status['t_len'] = t_len
        status['unk_src'] = '{:.1%}'.format(_unk_ratio(xs_data))
        status['unk_trg'] = '{:.1%}'.format(_unk_ratio(ts_data))
        status['time'] = int((t3 - t1) * 1000)
        #status['time_m'] = int((t2 - t1) * 1000)
        #status['time_f'] = int((t3 - t2) * 1000)
        logger.info(_status_str(status))

    time_end = time.time()
    acc = ok / tot

    # report results
    report = OrderedDict()
    report['type'] = 'EVAL_DONE'
    status['acc'] = '{:.1%}'.format(float(acc))
    report['correct'] = ok
    report['total'] = tot
    report['loss'] = '{:.4}'.format(float(loss_tot))
    report['samples'] = sample_num
    report['time'] = int(time_end - time_start)
    logger.info(_status_str(report))


def _unk_ratio(data):
    return float((data == UNK_ID).sum() / (data != IGNORE_ID).sum())


def _create_variables(xs):
    return list(map(lambda x: Variable(x), xs))


def _status_str(status):
    lst = []
    for k, v in status.items():
        lst.append(k + ':')
        lst.append(str(v))
    return '\t'.join(lst)