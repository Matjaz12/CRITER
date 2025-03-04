import os
import sys
import unittest
from parameterized import parameterized

import torch
import numpy as np

sys.path.append("../")
from datasets.sst import get_dataloaders


class TestDataloader(unittest.TestCase):
    """Test Dataloader"""

    @parameterized.expand(
        [
            ("SST_L3_CMEMS_2006-2021_Mediterranean.nc", False, 3, 256, 256),
            ("SST_L3_CMEMS_2006-2021_Mediterranean.nc", True, 3, 256, 256),
            ("SST_L3_CMEMS_2006-2021_Mediterranean.nc", False, 5, 256, 256),
            ("SST_L3_CMEMS_2006-2021_Mediterranean.nc", True, 5, 256, 256),
            ("CHEMS_L3_SST_Adriatic.nc", False, 3, 144, 180),
            ("CHEMS_L3_SST_Adriatic.nc", True, 3, 144, 180),
            ("CHEMS_L3_SST_Adriatic.nc", False, 5, 144, 180),
            ("CHEMS_L3_SST_Adriatic.nc", True, 5, 144, 180),
            ("CHEMS_SST_Atlantic.nc", False, 3, 256, 256),
            ("CHEMS_SST_Atlantic.nc", True, 3, 256, 256),
            ("CHEMS_SST_Atlantic.nc", False, 5, 256, 256),
            ("CHEMS_SST_Atlantic.nc", True, 5, 256, 256),
        ]
    )
    def test_shapes(self, filename, auxiliary_feat, time_win, height, width):
        """Test torch dataset shapes at different configurations"""
        batch_size = 1
        n_samples = 50
        data_path = os.path.join(os.environ["DATA_PATH"], filename)

        # load data
        dataloaders, _ = get_dataloaders(
            batch_size=batch_size,
            time_win=time_win,
            auxiliary_feat=auxiliary_feat,
            data_path=data_path,
            shuffle=False,
            n_samples=n_samples,
        )

        # iterate over dataloader modes and check shapes
        for mode in dataloaders.keys():
            observation, missing_mask = next(iter(dataloaders[mode]))

            expected_shape = (
                batch_size,
                time_win,
                3 if auxiliary_feat else 1,
                height,
                width,
            )
            assert (
                observation.shape == expected_shape
            ), f"Unexpected shape for observation! Got {observation.shape}, expected {expected_shape}"

            expected_shape = (batch_size, height, width)
            assert (
                missing_mask.shape == expected_shape
            ), f"Unexpected shape for missing_mask! Got {missing_mask.shape}, expected {expected_shape}"

    @parameterized.expand(
        [
            ("SST_L3_CMEMS_2006-2021_Mediterranean.nc", True, 3),
            ("SST_L3_CMEMS_2006-2021_Mediterranean.nc", False, 3),
            ("CHEMS_L3_SST_Adriatic.nc", True, 3),
            ("CHEMS_L3_SST_Adriatic.nc", False, 3),
            ("CHEMS_SST_Atlantic.nc", True, 3),
            ("CHEMS_SST_Atlantic.nc", False, 3),
        ]
    )
    def test_data_leak(self, filename, auxiliary_feat, time_win):
        """Test that no samples from train appear in the validation and test set"""
        # grab a subset of the data
        data_path = os.path.join(os.environ["DATA_PATH"], filename)
        dataloaders, _ = get_dataloaders(
            batch_size=1,
            time_win=time_win,
            auxiliary_feat=auxiliary_feat,
            data_path=data_path,
            shuffle=False,
            n_samples=500,
        )

        # make sure that there is not overlap between the training & test sets
        for x_train in dataloaders["train"].dataset.data["observation"]:
            for mode in ["val", "test"]:
                for x in dataloaders[mode].dataset.data["observation"]:
                    assert not (
                        x_train == x
                    ).all(), f"Data point x_train inside {mode} set!"


if __name__ == "__main__":
    unittest.main()
