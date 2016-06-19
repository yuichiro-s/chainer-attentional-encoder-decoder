import copy
import pickle
import os

import numpy as np
import chainer.optimizers as O
from chainer.serializers import save_hdf5, load_hdf5
from collections import defaultdict

from ecnn.vocabulary import EOS_ID, IGNORE_ID


MODEL_DEF_NAME = 'model_def.pickle'


def list2optimizer(lst):
    """Create chainer optimizer object from list of strings, such as ['SGD', '0.01']"""
    optim_name = lst[0]
    optim_args = map(float, lst[1:])
    optimizer = getattr(O, optim_name)(*optim_args)
    return optimizer


def load_bitext(path, vocab_src, vocab_trg):
    data = []
    with open(path) as f:
        for line in f:
            es = line.strip().split('\t')
            if len(es) == 2:
                src, trg = line.strip().split('\t')
                src_ids = list(map(vocab_src.get_id, src.split()))
                trg_ids = list(map(vocab_trg.get_id, trg.split()))
                data.append((src_ids, trg_ids))
    return data


def create_batches(data, batch_size, src_bucket_step, trg_bucket_step):
    # NOTE: this function modifies `data`
    batches = []
    buckets = defaultdict(list)
    for src_ids, trg_ids in data:
        while len(src_ids) % src_bucket_step > 0:
            src_ids.append(IGNORE_ID)
        trg_ids.append(EOS_ID)
        while len(trg_ids) % trg_bucket_step > 0:
            trg_ids.append(IGNORE_ID)
        buckets[len(src_ids), len(trg_ids)].append((src_ids, trg_ids))
    for samples in buckets.values():
        for i in range(0, len(samples), batch_size):
            src_ids_lst, trg_ids_lst = zip(*samples[i:i+batch_size])
            src_ids_arr = np.asarray(src_ids_lst, dtype=np.int32).T
            trg_ids_arr = np.asarray(trg_ids_lst, dtype=np.int32).T
            batch = src_ids_arr, trg_ids_arr
            batches.append(batch)
    return batches


def save_model_def(model, model_base_path):
    obj = copy.deepcopy(model)
    for p in obj.params():
        p.data = None
    path = os.path.join(model_base_path, MODEL_DEF_NAME)
    with open(path, 'wb') as f:
        pickle.dump(model, f)


def save_model(self, model_base_path, epoch):
    assert os.path.exists(
        os.path.join(model_base_path, MODEL_DEF_NAME)), 'Must call save_model_def() first'
    model_path = os.path.join(model_base_path, 'epoch{}'.format(epoch))
    save_hdf5(model_path, self)


def load_model(path):
    model_base_path = os.path.dirname(path)
    model_def_path = os.path.join(model_base_path, MODEL_DEF_NAME)
    with open(model_def_path, 'rb') as f:
        model = pickle.load(f)  # load model definition
        load_hdf5(path, model)  # load parameters
    return model
