from chainer import Chain
import chainer.links as L


class LstmRnn(Chain):

    def __init__(self, in_dim, out_dim):
        super(LstmRnn, self).__init__(
            lstm=L.StatelessLSTM(in_dim, out_dim),
        )

    def __call__(self, xs, h0=None):
        h = h0
        c = None
        hs = []
        for x in xs:
            c, h = self.lstm(c, h, x)
            hs.append(h)
        return hs
