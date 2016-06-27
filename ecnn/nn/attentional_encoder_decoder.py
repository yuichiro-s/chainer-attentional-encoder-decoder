from chainer import Chain
from chainer import Variable
from chainer import ChainList
from chainer import cuda
import chainer.functions as F
import chainer.links as L
import numpy as np

from ecnn.vocabulary import BOS_ID, EOS_ID, IGNORE_ID
from ecnn.nn.rnn import LstmRnn, GruRnn
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

    def __init__(self, word_emb, hidden_dim, layer_num, out_vocab_size, gru):
        super(AttentionalDecoder, self).__init__(
            softmax_linear=L.Linear(hidden_dim * 2, out_vocab_size),
            rnns=ChainList(),
        )
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.out_vocab_size = out_vocab_size
        Rnn = L.GRU if gru else L.StatelessLSTM
        for i in range(layer_num):
            rnn = Rnn(hidden_dim, hidden_dim)
            self.rnns.add_link(rnn)
        self.word_emb = word_emb
        self.gru = gru

    def __call__(self, xs, encoder_states):
        ys = []
        ws = []
        xp = cuda.get_array_module(encoder_states.data)
        batch_size = encoder_states.data.shape[0]
        rnn_states = self.init_states(xp, batch_size)
        bos = _create_var(encoder_states.data, BOS_ID, dtype=np.int32)
        for x in [bos] + xs:
            y, w, rnn_states = self.step(x, rnn_states, encoder_states)
            ys.append(y)
            ws.append(w)
        return ys, ws

    def generate(self, encoder_states, max_len, sample=False, temp=1.):
        ids = []
        ys = []
        ws = []
        xp = cuda.get_array_module(encoder_states.data)
        batch_size = encoder_states.data.shape[0]
        rnn_states = self.init_states(xp, batch_size)
        x = _create_var(encoder_states.data, BOS_ID, dtype=np.int32)

        done = xp.asarray([False] * batch_size)
        for i in range(max_len):
            y, w, rnn_states = self.step(x, rnn_states, encoder_states)
            ys.append(y)
            ws.append(w)

            if sample:
                unnormalized_probs = xp.exp(y.data * temp)
                probs = unnormalized_probs / unnormalized_probs.sum(axis=1, keepdims=True)
                probs = cuda.to_cpu(probs)
                next_id = xp.asarray(list(map(lambda ps: np.random.choice(len(ps), p=ps), probs)), dtype=np.int32)
            else:
                # deterministic
                next_id = y.data.argmax(axis=1).astype(np.int32)
            ids.append(next_id)
            done |= next_id == EOS_ID
            if all(done):
                # all samples have reached EOS
                break

            # create next ID
            x = Variable(next_id, volatile='on')

        # remove trailing IGNORE_ID's
        ids = map(cuda.to_cpu, ids)
        ids_lst = []
        for id_lst in zip(*ids):
            if EOS_ID in id_lst:
                eos_idx = id_lst.index(EOS_ID)
                id_lst = id_lst[:eos_idx+1]
            ids_lst.append(id_lst)

        return ids_lst, ys, ws

    def generate_beam(self, encoder_states, beam_size, max_len):
        assert encoder_states.data.shape[0] == 1, "batch size must be 1"

        xp = cuda.get_array_module(encoder_states.data)

        # grow encoder_states
        encoder_states = Variable(encoder_states.data.repeat(beam_size, axis=0), volatile='on')    # (batch_size x input_length)

        # initialize hypotheses
        hypotheses = []     # [(score, is_done, ids, length)]
        init_hypothesis = (
            0.,
            False,
            [BOS_ID],
            0,
        )
        hypotheses.append(init_hypothesis)

        step = 0
        rnn_states = self.init_states(xp, beam_size)
        while step < max_len and not all(map(lambda hypothesis: hypothesis[1], hypotheses)):
            # maximum steps have not been reached and some hypotheses have not reached <EOS>

            # generate next hypotheses
            next_hypotheses = []

            # run RNN simultaneously for all hypotheses
            assert len(hypotheses) <= beam_size
            x_d = xp.full((beam_size,), IGNORE_ID, np.int32)
            for i, (_, _, ids, _) in enumerate(hypotheses):
                x_d[i] = ids[-1]
            x = Variable(x_d, volatile='on')
            y, _, rnn_states = self.step(x, rnn_states, encoder_states)

            scores = xp.exp(y.data)
            scores /= scores.sum(axis=1, keepdims=True)
            for i, (hypothesis, y_i) in enumerate(zip(hypotheses, scores)):
                score, is_done, ids, length = hypothesis

                if is_done:
                    # leave as-is
                    next_hypotheses.append((i, hypothesis))
                else:
                    for next_id, next_score in sorted(enumerate(y_i), key=lambda i_d: -i_d[1])[:beam_size]:
                        next_h = (
                            score + next_score,
                            next_id == EOS_ID,
                            ids + [next_id],
                            length + 1,
                        )
                        next_hypotheses.append((i, next_h))

            # sort & filter hypotheses
            next_hypotheses.sort(key=lambda i_h: -i_h[1][0] / i_h[1][3])   # sort by average score
            #next_hypotheses.sort(key=lambda i_h: -i_h[1][0])   # sort by average score

            # prepare data for next step
            hypotheses = []

            # create next state
            new_states_data = []
            for state in rnn_states:
                if self.gru:
                    h = state
                    new_state_data = xp.zeros_like(h.data)
                else:
                    c, h = state
                    new_state_data = (xp.zeros_like(c.data), xp.zeros_like(h.data))
                new_states_data.append(new_state_data)
            for i, (j, hypothesis) in enumerate(next_hypotheses[:beam_size]):
                hypotheses.append(hypothesis)
                for l in range(self.layer_num):
                    if self.gru:
                        new_h_data = new_states_data[l]
                        old_h_data = rnn_states[l].data
                        new_h_data[i] = old_h_data[j]
                    else:
                        new_c_data, new_h_data = new_states_data[l]
                        old_c_data, old_h_data = rnn_states[l][0].data, rnn_states[l][1].data
                        new_c_data[i] = old_c_data[j]
                        new_h_data[i] = old_h_data[j]
            rnn_states = []
            for new_state_data in new_states_data:
                if self.gru:
                    new_state = Variable(new_state_data, volatile='on')
                else:
                    c_data, h_data = new_state_data
                    new_state = (Variable(c_data, volatile='on'), Variable(h_data, volatile='on'))
                rnn_states.append(new_state)

            step += 1

        res = []
        for _, _, ids, _ in hypotheses:
            # exclude BOS_ID
            res.append(ids[1:])

        return res

    def step(self, x, rnn_states, encoder_states):
        new_states = []
        h_in = self.word_emb(x)
        for rnn, state in zip(self.rnns, rnn_states):
            if self.gru:
                h = state
                state = rnn(h, h_in)
                h_in = state
            else:
                c, h = state
                state = rnn(c, h, h_in)
                _, h_in = state
            new_states.append(state)

        # TODO: linear + tanh for encoder_states and h_in
        batch_size, input_length, hidden_dim = encoder_states.data.shape
        unnormalized_weights = F.reshape(F.batch_matmul(encoder_states, h_in), (batch_size, input_length))   # (batch, input_length)
        normalized_weights = F.softmax(unnormalized_weights)   # (batch, input_length)
        encoder_context = F.reshape(F.batch_matmul(encoder_states, normalized_weights, transa=True), (batch_size, hidden_dim))  # (batch, hidden_dim)
        encoder_context_h_in = F.concat([encoder_context, h_in], axis=1)   # (batch, hidden_dim * 2)
        y = self.softmax_linear(encoder_context_h_in)   # TODO: according to the paper, ReLU is used after this linear

        return y, normalized_weights, new_states

    def init_states(self, xp, batch_size):
        if self.gru:
            vs = []
            for _ in range(self.layer_num):
                d = xp.zeros((batch_size, self.hidden_dim), dtype=np.float32)
                v = Variable(d, volatile='auto')
                vs.append(v)
            return vs
        else:
            return [(None, None)] * self.layer_num


class AttentionalEncoderDecoder(Chain):

    def __init__(self, in_vocab_size, hidden_dim, layer_num, out_vocab_size, gru, bidirectional, pyramidal, src_vocab_size=None):
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
        Rnn = GruRnn if gru else LstmRnn

        for i in range(layer_num):
            if bidirectional:
                rnn_f = Rnn(hidden_dim)
                rnn_b = Rnn(hidden_dim)
                rnn = BiRnn(rnn_f, rnn_b)
            else:
                rnn = Rnn(hidden_dim)
            rnns.add_link(rnn)
        multi_rnn = MultiLayerRnn(rnns, [hidden_dim] * layer_num, pyramidal)
        self.add_link('encoder', Encoder(self.word_emb_src, multi_rnn))
        self.add_link('decoder', AttentionalDecoder(self.word_emb_trg, hidden_dim, layer_num, out_vocab_size, gru))

        self.in_vocab_size = in_vocab_size
        self.hidden_dim = hidden_dim
        self.layer_num = layer_num
        self.out_vocab_size = out_vocab_size
        self.gru = gru
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

    def generate_beam(self, xs, **kwargs):
        hs = self.encoder(xs)
        encoder_states = _create_encoder_states_matrix(hs)
        return self.decoder.generate_beam(encoder_states, **kwargs)


def _create_var(arr, val, dtype):
    batch_size = arr.shape[0]
    xp = cuda.get_array_module(arr)
    return Variable(xp.full((batch_size,), val, dtype), volatile='auto')


def _create_encoder_states_matrix(hs):
    hs_3d = list(map(lambda h: F.expand_dims(h, 1), hs))  # (batch_size, 1, dim)
    return F.concat(hs_3d, axis=1)    # (batch_size, input_length, dim)
