import os
import sys
import unittest
from unittest import TestLoader
from parameterized import parameterized

import torch
import numpy as np

sys.path.append("../")
from datasets.sst import (
    MaskSampler,
    get_dataloaders,
    _compute_missing_val_fraction,
)


class Test_SST_Dataset(unittest.TestCase):
    """Test SST Dataset"""

    @classmethod
    def setUpClass(cls):
        # load the dataset
        data_path = os.path.join(
            os.environ["DATA_PATH"], "SST_L3_CMEMS_2006-2021_Mediterranean.nc"
        )
        n_samples = (
            int(os.environ["N_SAMPLES"]) if "N_SAMPLES" in os.environ.keys() else None
        )

        cls.dataloaders, cls.utils = get_dataloaders(
            batch_size=8,
            time_win=3,
            auxiliary_feat=True,
            data_path=data_path,
            shuffle=False,
            n_samples=n_samples,
            cloud_coverage_threshold=1.1,
        )

    def test_if_missing_masks_correspond_to_measurements(self):
        """
        Test if measurements are assigned correct missing masks, i.e.,
        check if x and x_hat * (missing_mask * land_mask) have unknown values at the same set of locations.
        """

        def _get_visible_pixel_idxs(observation):
            non_zero_idxs = np.argwhere(
                observation != 0
            )  # matrix of shape (2 x n), with rows [i, j]
            return non_zero_idxs[0, :] * observation.shape[1] + non_zero_idxs[1, :]

        for mode in self.dataloaders.keys():
            for x_batch, missing_mask_batch in self.dataloaders[mode]:
                for x, missing_mask in zip(x_batch, missing_mask_batch):

                    # fetch the sst at time step t
                    sst_t = x[
                        self.utils["time_win"] // 2,
                        self.utils["feat_to_idx"]["measurement"],
                        :,
                        :,
                    ]
                    # mask a dummy sst with the corresponding missing and land mask
                    sst_const_t = torch.ones_like(sst_t) * (
                        missing_mask * self.utils["land_mask"]
                    )

                    # check if the same set of pixels is
                    if not torch.equal(
                        _get_visible_pixel_idxs(sst_t),
                        _get_visible_pixel_idxs(sst_const_t),
                    ):
                        raise Exception(f"Masks do not match!")

    def test_measurement_temporal_consistency(self):
        """
        Test if **un-shuffled** dataset is temporally consistent, i.e.,
        check if the central measurement of current samples is equal to the previous measurement
        of next sample, precisely: x^{(i)}_{t} == x^{(i+1)}_{t-1}
        """
        x_prev = None
        for mode in self.dataloaders.keys():
            for x in self.dataloaders[mode].dataset.data["observation"]:
                x = x[:, self.utils["feat_to_idx"]["measurement"]].unsqueeze(dim=1)

                if x_prev is not None:
                    # x_prev(T//2) == x_curr(T//2-1)
                    if not torch.equal(
                        x_prev[
                            self.utils["time_win"] // 2,
                            self.utils["feat_to_idx"]["measurement"],
                        ],
                        x[
                            self.utils["time_win"] // 2 - 1,
                            self.utils["feat_to_idx"]["measurement"],
                        ],
                    ):

                        raise Exception(
                            f"x_prev(T//2) and x_curr(T//2-1) do not match!"
                        )

                x_prev = x.clone()

    def test_duplicated_observations(self):
        """
        Test if there are exactly time_win unqiue observations [x_t, a_t] within each sample,
        with the expection of the first and the last sample in the dataset
        """
        for mode in self.dataloaders.keys():
            for idx, observation in enumerate(
                self.dataloaders[mode].dataset.data["observation"]
            ):

                # ignore the two duplicates which are there by design.
                # The first and the last sample have duplicated frames
                if mode == "train" and idx == 0:
                    continue
                if (
                    mode == "test"
                    and idx
                    == len(self.dataloaders[mode].dataset.data["observation"]) - 1
                ):
                    continue

                # number of unique tensor along the temporal dim
                num_unique = len(torch.unique(observation, dim=0))
                if num_unique != self.utils["time_win"]:
                    raise Exception(
                        f"Number of unique frames: {num_unique} < time_win: {self.utils['time_win']}!"
                    )

    def test_missing_masks(self, n_iters=100):
        """Make sure that the missing mask is NOT placed over the pixels corresponding to the land."""
        for _ in range(n_iters):
            _, Mm = next(iter(self.dataloaders["train"]))
            assert torch.unique(Mm[:, self.utils["land_mask"] == 0]) == 1  # not hidden

    def test_compute_missing_fract(self):
        """
        Test the computation of the missing fraction, i.e., the fraction
        of missing (unknown, undefined) measurements
        """
        x = np.ones((256, 256))
        x[:, : 256 // 2] = 0
        assert _compute_missing_val_fraction(x) == 0.5

        x = np.ones((256, 256))
        x[0, 0] = 0
        assert _compute_missing_val_fraction(x) == (1 / 256**2)

        x = np.ones((256, 256))
        assert _compute_missing_val_fraction(x) == 0.0

        x = np.zeros((256, 256))
        assert _compute_missing_val_fraction(x) == 1.0

    @parameterized.expand(
        [
            (0.0, 100),
            (0.1, 100),
            (0.2, 100),
            (0.3, 100),
            (0.4, 100),
        ]
    )
    def test_mask_sampler(self, mask_size_threshold, num_samples):
        """Test mask-sampler"""
        mask_sampler = MaskSampler(
            self.dataloaders["train"].dataset, mask_size_threshold
        )

        for _ in range(num_samples):
            mask = mask_sampler()
            mask_size = _compute_missing_val_fraction(mask.numpy())
            assert (
                mask_size > mask_size_threshold
            ), f"Size:{mask_size} of a sampled mask is bellow the threshold:{mask_sampler.mask_size_threshold}"


if __name__ == "__main__":
    unittest.main()
