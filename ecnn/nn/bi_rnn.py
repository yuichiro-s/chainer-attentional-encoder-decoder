from chainer import Chain


class BiRnn(Chain):

    def __init__(self, forward_rnn, backward_rnn):
        """
        Bidirectional RNN.
        """
        super(BiRnn, self).__init__(
            forward_rnn=forward_rnn,
            backward_rnn=backward_rnn,
        )

    def __call__(self, xs):
        hs_f = self.forward_rnn(xs)
        hs_b = reversed(self.backward_rnn(list(reversed(xs))))
        hs = list(map(lambda h_fb: h_fb[0] + h_fb[1], zip(hs_f, hs_b)))
        return hs
