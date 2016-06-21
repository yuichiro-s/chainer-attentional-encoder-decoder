import unittest
import numpy as np
from chainer import Variable

from ecnn.nn.attentional_encoder_decoder import AttentionalEncoderDecoder


class TestAttentionalEncoderDecoder(unittest.TestCase):

    def setUp(self):
        in_vocab_size = 10
        out_vocab_size = 10
        hidden_dim = 5
        layer_num = 3
        self.encdecs = [
            AttentionalEncoderDecoder(in_vocab_size, hidden_dim, layer_num, out_vocab_size, bidirectional, pyramidal, src_vocab_size)
            for bidirectional in [True, False]
            for pyramidal in [True, False]
            for src_vocab_size in [in_vocab_size, None]
            ]

    def test_forward(self):
        for encdec in self.encdecs:
            batch_size = 6
            x_len = 11
            t_len = 4
            xs = _create_xs(encdec, batch_size, x_len)
            ts = _create_xs(encdec, batch_size, t_len)
            loss, ys, ws = encdec(xs, ts)

            h_len = x_len
            if encdec.pyramidal:
                for i in range(encdec.layer_num-1):
                    h_len = (h_len + 1) // 2

            self.assertEqual(len(ys), t_len)
            self.assertEqual(len(ws), t_len)
            self.assertEqual(ys[0].data.shape, (batch_size, encdec.out_vocab_size))
            self.assertEqual(ws[0].data.shape, (batch_size, h_len))

    def test_generate(self):
        for encdec in self.encdecs:
            batch_size = 6
            x_len = 11
            max_len = 10
            xs = _create_xs(encdec, batch_size, x_len, volatile='on')
            ids, ys, ws = encdec.generate(xs, max_len=max_len)

            max_len = max(map(len, ids))
            self.assertEqual(max_len, len(ys))
            self.assertEqual(max_len, len(ws))
            self.assertEqual(len(ids), batch_size)
            self.assertTrue(all(map(lambda id: id < encdec.out_vocab_size, ids[0])))

    def test_generate_beam(self):
        for encdec in self.encdecs:
            batch_size = 1
            x_len = 11
            max_len = 10
            xs = _create_xs(encdec, batch_size, x_len, volatile='on')

            beam_size = 4
            #for ids, ys, ws in encdec.generate_beam(xs, beam_size=beam_size, max_len=max_len):
            for ids in encdec.generate_beam(xs, beam_size=beam_size, max_len=max_len):
                #self.assertEqual(len(ids), len(ys))
                #self.assertEqual(len(ids), len(ws))
                self.assertTrue(all(map(lambda id: id < encdec.out_vocab_size, ids)))


def _create_x(encdec, batch_size, volatile):
    data = np.random.randint(0, encdec.in_vocab_size, (batch_size,), dtype=np.int32)
    return Variable(data, volatile=volatile)


def _create_xs(encdec, batch_size, x_len, volatile='off'):
    xs = list(map(lambda _: _create_x(encdec, batch_size, volatile=volatile), range(x_len)))
    return xs

