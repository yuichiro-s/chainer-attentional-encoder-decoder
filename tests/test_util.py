import unittest

from ecnn.util import create_batches
from ecnn.vocabulary import EOS_ID


class TestUtil(unittest.TestCase):

    def test_create_batches(self):
        batch_size = 3
        src_bucket_step = 2
        trg_bucket_step = 2
        data = [([9, 6, 3, 6, 4, 8, 9],          [9, 5, 8, 8, 7]),
                ([5, 3, 3, 8, 8, 9, 8, 7, 5, 7], [5, 7, 7, 4, 5, 3, 4, 6, 7]),
                ([5, 3, 6, 8, 3, 4, 6, 8],       [9, 7, 8, 9, 6, 9, 4]),
                ([8, 9, 6, 4, 3, 4],             [4, 4, 4, 4, 6]),
                ([6, 6, 5, 6, 4, 5, 7, 7, 4],    [5, 7, 4, 3, 7, 3]),
                ([9, 5, 8, 6, 5, 4, 8, 6, 9],    [8, 5, 4, 8, 7, 3, 5]),
                ([6, 8, 5, 9, 6, 5, 3, 3, 9, 4], [3, 3, 4, 5, 4, 4, 4, 5, 5]),
                ([7, 5, 7, 6, 7],                [6, 3, 5, 6, 4, 3, 9, 8, 5, 8]),
                ([7, 7, 7, 4, 8, 5, 7, 9],       [3, 5, 5, 3, 4]),
                ([4, 6, 8, 8, 9, 6, 8, 7, 6],    [6, 3, 8, 5, 9])]
        batches = create_batches(data, batch_size, src_bucket_step, trg_bucket_step)

        for xs, ts in batches:
            xs0, xs1 = xs.shape
            ts0, ts1 = ts.shape
            self.assertEqual(xs0 % src_bucket_step, 0)
            self.assertLessEqual(xs1, batch_size)
            self.assertEqual(ts0 % trg_bucket_step, 0)
            self.assertLessEqual(ts1, batch_size)
            self.assertIn(EOS_ID, ts)

