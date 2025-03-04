import sys
import unittest
from parameterized import parameterized
import torch
import numpy as np

sys.path.append("../")
from modules.irm import _estimate_residual_rec_and_var, _scale_thetas


class TestRen(unittest.TestCase):

    @parameterized.expand(
        [
            (10, 1e-3, 100),
            (13, 5e-3, 100),
            (100, 1e-4, 100),
        ]
    )
    def test_var_estimation(self, gamma1, mu1, num_tests):
        exp_range = [1 / torch.exp(torch.tensor(gamma1)), 1 / mu1]
        print(f"checking if var is in range: {exp_range}")

        for _ in range(num_tests):
            # init. a dummy prediction and compute the associated variance
            Y_hat = 1000 * torch.randn((1, 2, 512, 512))
            _, var = _estimate_residual_rec_and_var(Y_hat, gamma1, mu1)

            # check that the total variance is in correct range
            assert var.min() >= exp_range[0]
            assert var.max() <= exp_range[1]

    @parameterized.expand(
        [
            (10, 1e-3, 100),
            (13, 5e-3, 100),
        ]
    )
    def test_scaled_var_estimation(self, gamma1, mu1, num_tests):
        # compute the expected range in which the total variance should be defined
        exp_range = [1 / torch.exp(torch.tensor(gamma1)), 1 / mu1]
        print(f"checking if var is in range: {exp_range}")

        for _ in range(num_tests):
            # compute the scaled gamma and mu
            num_iterations = np.random.choice(range(1, 10))
            gamma, mu = _scale_thetas(num_iterations, gamma1, mu1)
            var = sum(
                [
                    _estimate_residual_rec_and_var(
                        100 * torch.randn((1, 2, 512, 512)), gamma, mu
                    )[1]
                    for _ in range(num_iterations)
                ]
            )

            # check that the total variance is in correct range
            assert torch.abs(var.min() - exp_range[0]) < 1e-3, var.min()
            assert torch.abs(var.max() - exp_range[1]) < 1e-3, var.max()


if __name__ == "__main__":
    unittest.main()
