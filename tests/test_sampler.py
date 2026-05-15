import unittest

import torch

from sparsevllm.layers.sampler import Sampler


class SamplerTest(unittest.TestCase):
    def test_all_greedy_skips_sampling_path(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 3.0, 2.0], [5.0, 4.0, 6.0]])

        out = sampler(logits, temperatures=None, all_greedy=True)

        self.assertEqual(out.tolist(), [1, 2])

    def test_non_greedy_requires_temperatures(self):
        sampler = Sampler()
        logits = torch.tensor([[1.0, 3.0, 2.0]])

        with self.assertRaises(ValueError):
            sampler(logits, temperatures=None, all_greedy=False)


if __name__ == "__main__":
    unittest.main()
