from chainer import Chain
from chainer import Variable
from chainer import ChainList
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np

from ecnn.vocabulary import BOS_ID, EOS_ID, IGNORE_ID
from ecnn.nn.rnn import LstmRnn
from ecnn.nn.bi_rnn import BiRnn
from ecnn.nn.multi_rnn import MultiLayerRnn


class Encoder(Chain):

    def __init__(self, word_emb, multi_rnn):
        super(Encoder, self).__init__(
            multi_rnn=multi_rnn,
        )
        self.word_emb = word_emb

    def __call__(self, xs):
        hs = []
        for x in xs:
            h = self.word_emb(x)
            hs.append(h)
        hs = self.multi_rnn(hs)
        return hs


class AttentionalDecoder(Chain):

    def __init__(self, word_emb, hidden_dim, layer_num, out_vocab_size):
        super(AttentionalDecoder, self).__init__(
            softmax_linear=L.Linear(hidden_dim * 2, out_vocab_size),
            lstms=ChainList(),
        )
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.out_vocab_size = out_vocab_size
        for i in range(layer_num):
            lstm = L.StatelessLSTM(hidden_dim, hidden_dim)
            self.lstms.add_link(lstm)
        self.word_emb = word_emb

    def __call__(self, xs, encoder_states):
        ys = []
        ws = []
        cs = [None] * self.layer_num
        hs = [None] * self.layer_num
        bos = _create_var(encoder_states.data, BOS_ID, dtype=np.int32)
        for x in [bos] + xs:
            y, w, hs, cs = self.step(x, hs, cs, encoder_states)
            ys.append(y)
            ws.append(w)
        return ys, ws

    def generate(self, encoder_states, max_len):
        ids = []
        ys = []
        ws = []
        cs = [None] * self.layer_num
        hs = [None] * self.layer_num
        x = _create_var(encoder_states.data, BOS_ID, dtype=np.int32)

        batch_size = encoder_states.data.shape[0]
        done = [False] * batch_size
        for i in range(max_len):
            y, w, hs, cs = self.step(x, hs, cs, encoder_states)
            ys.append(y)
            ws.append(w)

            next_id = y.data.argmax(axis=1)
            ids.append(next_id)
            done |= next_id == EOS_ID
            if all(done):
                # all samples have reached EOS
                break

        return ids, ys, ws

    def step(self, x, hs, cs, encoder_states):
        new_hs = []
        new_cs = []
        h_in = self.word_emb(x)
        for lstm, h, c in zip(self.lstms, hs, cs):
            c, h_in = lstm(c, h, h_in)
            new_hs.append(h_in)
            new_cs.append(c)

        # TODO: linear + tanh for encoder_states and h_in
        batch_size, input_length, hidden_dim = encoder_states.data.shape
        unnormalized_weights = F.reshape(F.batch_matmul(encoder_states, h_in), (batch_size, input_length))   # (batch, input_length)
        normalized_weights = F.softmax(unnormalized_weights)   # (batch, input_length)
        encoder_context = F.reshape(F.batch_matmul(encoder_states, normalized_weights, transa=True), (batch_size, hidden_dim))  # (batch, hidden_dim)
        encoder_context_h_in = F.concat([encoder_context, h_in], axis=1)   # (batch, hidden_dim * 2)
        y = self.softmax_linear(encoder_context_h_in)   # TODO: according to the paper, ReLU is used after this linear

        return y, normalized_weights, new_hs, new_cs


class AttentionalEncoderDecoder(Chain):

    def __init__(self, in_vocab_size, hidden_dim, layer_num, out_vocab_size, bidirectional, pyramidal, src_vocab_size=None):
        super(AttentionalEncoderDecoder, self).__init__()

        if src_vocab_size is None:
            # use same vocabulary for source/target
            word_emb = L.EmbedID(in_vocab_size, hidden_dim, ignore_label=IGNORE_ID)
            self.add_link('word_emb', word_emb)
            self.word_emb_src = word_emb
            self.word_emb_trg = word_emb
        else:
            word_emb_src = L.EmbedID(src_vocab_size, hidden_dim, ignore_label=IGNORE_ID)
            word_emb_trg = L.EmbedID(in_vocab_size, hidden_dim, ignore_label=IGNORE_ID)
            self.add_link('word_emb_src', word_emb_src)
            self.add_link('word_emb_trg', word_emb_trg)

        rnns = ChainList()
        for i in range(layer_num):
            if bidirectional:
                rnn_f = LstmRnn(hidden_dim, hidden_dim)
                rnn_b = LstmRnn(hidden_dim, hidden_dim)
                rnn = BiRnn(rnn_f, rnn_b)
            else:
                rnn = LstmRnn(hidden_dim, hidden_dim)
            rnns.add_link(rnn)
        multi_rnn = MultiLayerRnn(rnns, [hidden_dim] * layer_num, pyramidal)
        self.add_link('encoder', Encoder(self.word_emb_src, multi_rnn))
        self.add_link('decoder', AttentionalDecoder(self.word_emb_trg, hidden_dim, layer_num, out_vocab_size))

        self.in_vocab_size = in_vocab_size
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.out_vocab_size = out_vocab_size
        self.bidirectional = bidirectional
        self.pyramidal = pyramidal

    def __call__(self, xs, ts):
        # NOTE: users must prepend <EOS> and pad with -1 by themselves
        hs = self.encoder(xs)
        encoder_states = _create_encoder_states_matrix(hs)
        ys, ws = self.decoder(ts[:-1], encoder_states)  # last element of ts only plays a role of target

        loss = 0
        assert len(ys) == len(ts)
        for y, t in zip(ys, ts):
            # TODO: map ID to UNK if in_vocab_size != out_vocab_size
            loss += F.softmax_cross_entropy(y, t)

        return loss, ys, ws

    def generate(self, xs, **kwargs):
        hs = self.encoder(xs)
        encoder_states = _create_encoder_states_matrix(hs)
        ids, ys, ws = self.decoder.generate(encoder_states, **kwargs)
        return ids, ys, ws


def _create_var(arr, val, dtype):
    batch_size = arr.shape[0]
    xp = cuda.get_array_module(arr)
    return Variable(xp.full((batch_size,), val, dtype))


def _create_encoder_states_matrix(hs):
    hs_3d = list(map(lambda h: F.expand_dims(h, 1), hs))  # (batch_size, 1, dim)
    return F.concat(hs_3d, axis=1)    # (batch_size, input_length, dim)
