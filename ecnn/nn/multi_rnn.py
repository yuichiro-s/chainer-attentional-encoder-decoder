from chainer import Chain
from chainer import ChainList
from chainer import Variable
from chainer import cuda
import chainer.functions as F


class MultiLayerRnn(Chain):

    def __init__(self, rnns, dims, pyramidal, dropout_ratio):
        super(MultiLayerRnn, self).__init__(
            rnns=rnns,
        )
        self.pyramidal = pyramidal
        if self.pyramidal:
            self.add_link('combine_twos', ChainList())
            for in_dim, out_dim in zip(dims, dims[1:]):
                combine_two = CombineTwo(in_dim, out_dim)
                self.combine_twos.add_link(combine_two)
            assert len(self.rnns) == len(self.combine_twos) + 1
        self.dropout_ratio = dropout_ratio

    def __call__(self, xs, train):
        hs = xs
        for i, rnn in enumerate(self.rnns):
            hs = rnn(hs)
            if i < len(self.rnns) - 1:
                if self.dropout_ratio > 0:
                    hs = list(map(lambda h: F.dropout(h, self.dropout_ratio, train), hs))
                if self.pyramidal:
                    hs = self.combine_twos[i](hs)
        return hs


class CombineTwo(Chain):

    def __init__(self, in_dim, out_dim, f=F.tanh):
        super(CombineTwo, self).__init__(
            l=F.Linear(in_dim * 2, out_dim),
        )
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.f = f

    def __call__(self, xs):
        if len(xs) % 2 == 1:
            # length is odd
            d = xs[0].data
            xp = cuda.get_array_module(d)
            pad = Variable(xp.zeros_like(d), volatile='auto')    # TODO: is zero padding OK?
            xs = xs + [pad]
        assert len(xs) % 2 == 0

        hs = []
        for i in range(len(xs) // 2):
            x1 = xs[i*2]
            x2 = xs[i*2+1]
            x_in = F.concat([x1, x2], axis=1)
            h = self.f(self.l(x_in))
            hs.append(h)
        assert len(hs) == len(xs) // 2

        return hs
