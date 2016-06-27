from chainer import Chain
import chainer.links as L


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
            gru=L.StatefulGRU(hidden_dim, hidden_dim),
        )

    def __call__(self, xs):
        hs = []
        for x in xs:
            h = self.gru(x)
            hs.append(h)
        return hs
