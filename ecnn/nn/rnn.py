from chainer import Chain, Variable, cuda
import chainer.links as L
import numpy as np


class LstmRnn(Chain):

    def __init__(self, hidden_dim):
        super(LstmRnn, self).__init__(
            lstm=L.StatelessLSTM(hidden_dim, hidden_dim),
        )

    def __call__(self, xs, h0=None):
        h = h0
        c = None
        hs = []
        for x in xs:
            c, h = self.lstm(c, h, x)
            hs.append(h)
        return hs


class GruRnn(Chain):

    def __init__(self, hidden_dim):
        super(GruRnn, self).__init__(
            gru=L.GRU(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def __call__(self, xs, h0=None):
        if h0 is None:
            xp = cuda.get_array_module(xs[0].data)
            batch_size = xs[0].data.shape[0]
            h = self.init_hidden(xp, batch_size)
        else:
            h = h0
        hs = []
        for x in xs:
            h = self.gru(h, x)
            hs.append(h)
        return hs

    def init_hidden(self, xp, batch_size):
        d = xp.zeros((batch_size, self.hidden_dim), dtype=np.float32)
        v = Variable(d, volatile='auto')
        return v
