import unittest

import numpy as np
import pandas as pd
import torch

import nbsr.distributions as dist

class TestDistributions(unittest.TestCase):

    def test_log_normal(self):
        beta = torch.tensor([[1.1, 0.8, 0.7], [0.9, 1.0, 1.2]])
        sd = torch.tensor([0.1, 0.2, 0.3])
        log_normal_realized = torch.sum(dist.log_normal(beta, torch.zeros_like(sd), sd))
        log_normal_expected = -127.5039
        print(log_normal_expected)
        print(log_normal_realized)
        self.assertTrue(np.allclose(log_normal_expected, log_normal_realized))

if __name__ == '__main__':
    unittest.main()