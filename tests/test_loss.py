import sys
import unittest
from parameterized import parameterized

import torch
import numpy as np

sys.path.append("../")
from loss import (
    mean_squared_error,
    neg_log_likelihood_gaussian,
    _masked_squared_error,
)
from modules.irm import _estimate_residual_rec_and_var


def _get_dummy_sample(B, C, H, W):
    observation = torch.rand((B, C, H, W))
    missing_mask = torch.ones((B, H, W))
    missing_mask[:, int(0.8 * H) :, int(0.8 * W) :] = 0
    land_mask = torch.ones((H, W))
    land_mask[: int(0.5 * H), : int(0.5 * W)] = 0
    sampled_mask = torch.ones_like(land_mask)
    sampled_mask[int(0.3 * H) :, int(0.3 * W) :] = 0
    return observation, missing_mask, land_mask, sampled_mask


class TestLoss(unittest.TestCase):
    """Test the loss implementation"""

    @parameterized.expand([(256, 256), (144, 180)])
    def test_loss_trend(self, height, width):
        """Test that loss monotonically decreases as reconstruction gets closer to the ground truth measurement"""
        observation, missing_mask, land_mask, sampled_mask = _get_dummy_sample(
            1, 1, height, width
        )

        losses = {"mse_all": [], "mse_vis": [], "mse_hid": []}
        for eps in [10, 5, 0.5, 0.1, 0.001, 0]:
            # fetch the expected sst and add noise to it
            target = observation[0, 0, :, :]
            pred = target + eps * torch.randn_like(target)

            # compute L2 loss
            ld = mean_squared_error(
                target, pred, missing_mask[0], land_mask, sampled_mask
            )
            losses["mse_all"] += [ld["mse_all"].item()]
            losses["mse_vis"] += [ld["mse_vis"].item()]
            losses["mse_hid"] += [ld["mse_hid"].item()]

        for name, l in losses.items():
            is_decreasing = np.all(np.diff(l) <= 0)
            assert is_decreasing, f"{name}: {l} is not monotonically decreasing!"

    @parameterized.expand([(256, 256), (144, 180)])
    def test_mse_vectorization(self, height, width):
        """Compare the vectorized implementation of the `mse` function with a sequential implementation"""
        observation, missing_mask, land_mask, sampled_mask = _get_dummy_sample(
            4, 1, height, width
        )

        # generate a dummy prediction and compute the corresponding loss
        target = observation[:, 0, :, :]
        pred = torch.randn(target.shape)
        ld = mean_squared_error(target, pred, missing_mask, land_mask, sampled_mask)

        def mse_sisd(y_hat, y, M):
            batch, height, width = y_hat.shape
            loss, n_pixels = 0, 0
            for b in range(batch):
                for i in range(height):
                    for j in range(width):
                        if M[b, i, j] == 1:
                            loss += (y_hat[b, i, j] - y[b, i, j]) ** 2
                            n_pixels += 1
            return loss / n_pixels

        # test if the vectorized loss matches with the expected (SISD) loss
        loss = mse_sisd(target, pred, ld["M_all"])
        assert int(loss) == int(
            ld["mse_all"]
        ), f"vectorized loss_all: {ld['mse_all']} doesn't equal to the expected loss_all: {loss}"

        loss = mse_sisd(target, pred, ld["M_vis"])
        assert int(loss) == int(
            ld["mse_vis"]
        ), f"vectorized loss_vis: {ld['mse_vis']} doesn't equal to the expected loss_vis: {loss}"

        loss = mse_sisd(target, pred, ld["M_hid"])
        assert int(loss) == int(
            ld["mse_hid"]
        ), f"vectorized loss_hid: {ld['mse_hid']} doesn't equal to the expected loss_hid: {loss}"

    @parameterized.expand([(256, 256), (144, 180)])
    def test_compute_mse_masks(self, height, width):
        """Test if mask_all, mask_vis and mask_hid obey mask_all = mask_vis + mask_hid"""
        observation, missing_mask, land_mask, sampled_mask = _get_dummy_sample(
            4, 1, height, width
        )
        target = observation[:, 0, :, :] * (land_mask * missing_mask)
        pred = torch.randn(target.shape)

        ld = mean_squared_error(target, pred, missing_mask, land_mask, sampled_mask)
        assert torch.equal(ld["M_all"], ld["M_vis"] + ld["M_hid"])

    @parameterized.expand([(256, 256), (144, 180)])
    def test_masked_squared_error(self, height, width):
        """Test that the sum of the squared error computed over masked regions is zero"""
        observation, missing_mask, land_mask, sampled_mask = _get_dummy_sample(
            4, 1, height, width
        )
        target = observation[:, 0, :, :] * (
            land_mask * missing_mask
        )  # simulate missing values over the target
        pred = torch.randn(target.shape)

        ld = mean_squared_error(target, pred, missing_mask, land_mask, sampled_mask)
        # compute the sum of the error over the masked out region
        se = _masked_squared_error(pred, target, ld["M_all"])
        assert se[ld["M_all"] == 0].sum() == 0
        se = _masked_squared_error(pred, target, ld["M_vis"])
        assert se[ld["M_vis"] == 0].sum() == 0
        se = _masked_squared_error(pred, target, ld["M_hid"])
        assert se[ld["M_hid"] == 0].sum() == 0

    @parameterized.expand([(256, 256), (144, 180)])
    def test_neg_log_likelihood_gaussian_vectorization(self, height, width):
        """Compare the vectorized implementation of the `neg_log_likelihood_gaussian` function with a sequential implementation"""
        observation, missing_mask, land_mask, _ = _get_dummy_sample(4, 1, height, width)

        # generate a dummy prediction and compute the corresponding loss
        target = observation[:, 0, :, :]
        Y_hat = torch.randn((target.shape[0], 2, target.shape[1], target.shape[2]))
        x_hat, var_hat = _estimate_residual_rec_and_var(Y_hat)
        ld = neg_log_likelihood_gaussian(
            target, x_hat, var_hat, missing_mask, land_mask
        )

        def neg_log_likelihood_gaussian_sisd(y, y_hat, var_hat, M):
            batch, height, width = y_hat.shape
            loss, n_pixels = 0, 0
            for b in range(batch):
                for i in range(height):
                    for j in range(width):
                        if M[b, i, j] == 1:
                            loss += (y_hat[b, i, j] - y[b, i, j]) ** 2 / var_hat[
                                b, i, j
                            ] + torch.log(var_hat[b, i, j])
                            n_pixels += 1

            return loss / n_pixels

        loss = neg_log_likelihood_gaussian_sisd(target, x_hat, var_hat, ld["M_all"])
        assert int(loss) == int(
            ld["loss_all"]
        ), f"vectorized loss: {ld['loss_all']} doesn't equal to the expected loss_all: {loss}"


if __name__ == "__main__":
    unittest.main()
