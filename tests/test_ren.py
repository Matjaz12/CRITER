import sys
import unittest
from parameterized import parameterized
import torch

sys.path.append("../")
from modules.ren import ResidualEstimationNetwork


class TestRen(unittest.TestCase):
    """Test the Residual Estimation Network"""

    @parameterized.expand(
        [
            (8, 3, [32, 64, 128], 256, 256),
            (8, 3, [32, 64], 144, 180),
        ]
    )
    def test_pred_shape(
        self,
        batch,
        time_win,
        channels,
        height,
        width,
    ):
        """Test the output shape of the refinement module"""
        observation = torch.randn((batch, time_win + 1, height, width))
        rm = ResidualEstimationNetwork(observation.shape[1], channels=channels)
        Y_hat = rm(observation)
        assert Y_hat.shape == (batch, 2, height, width)


if __name__ == "__main__":
    unittest.main()
