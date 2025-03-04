import sys
import unittest
from parameterized import parameterized
import torch

sys.path.append("../")
from modules.criter import load_model


class TestCRITER(unittest.TestCase):
    @parameterized.expand(
        [
            (8, 3, 3, [32, 64, 128], 256, 256, 32, 1),
            (8, 3, 3, [32, 64, 128], 256, 256, 16, 1),
            (8, 3, 3, [32, 64], 144, 180, 12, 1),
            (8, 3, 3, [32, 64], 144, 180, 6, 1),
        ]
    )
    def test_pred_shape(
        self,
        batch,
        time_win,
        channels,
        rm_layers,
        height,
        width,
        s_patch_size,
        t_patch_size,
    ):
        """Test if output shape of CRITER is as expected"""

        model = load_model(
            (height, width),
            channels,
            time_win,
            s_patch_size,
            t_patch_size,
            rm_layers=rm_layers,
            extraction_layer=11,
        )

        # initialize dummy data
        observation = torch.randn((batch, time_win, channels, height, width))
        mask = torch.zeros((height, width))
        metadata = {"time_win": time_win, "feat_to_idx": {"measurement": 0}}

        missing_mask = torch.ones((height, width))
        missing_mask[0:10, 0:10] = 0
        land_mask = torch.ones((height, width))
        land_mask[20:40, 20:40] = 0

        # inference
        rec, var = model(
            observation, missing_mask, land_mask, mask, metadata, inference=True
        )

        assert rec.shape == (batch, height, width)
        assert var.shape == (batch, height, width)


if __name__ == "__main__":
    unittest.main()
