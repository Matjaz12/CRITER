import sys
import torch
import unittest
from parameterized import parameterized

sys.path.append("../")
from modules.ffm import FeatureFusionModule, _reshape_2d_to_3d


class TestFFM(unittest.TestCase):
    """Test Feature Fusion Module"""

    @parameterized.expand(
        [
            (128, 32, 32, 3 // 1, 256 // 16, 256 // 16, 192),
            (64, 36, 45, 3 // 1, 144 // 16, 180 // 16, 192),
        ]
    )
    def test_output_shape(
        self,
        feat_channels,
        feat_height,
        feat_width,
        num_patches_t,
        num_patches_h,
        num_patches_w,
        embed_dim,
    ):
        """Check if the output shape of the feature fusion module matches the input shape"""
        # construct dummy input and dummy features
        feat = torch.randn((1, feat_channels, feat_height, feat_width))
        toks = torch.randn(1, num_patches_t * num_patches_h * num_patches_w, embed_dim)

        ffm = FeatureFusionModule(
            feat_channels,
            embed_dim,
            num_patches_w,
            num_patches_h,
            num_patches_t,
        )

        # check that the output is of the same shape as the input
        feat_tilde = ffm(feat, toks)
        assert feat_tilde.shape == feat.shape

    @parameterized.expand(
        [
            (3, 256, 256, 16, 1, 192),
            (3, 144, 180, 12, 1, 192),
        ]
    )
    def test_reshape_2d_to_3d(
        self, time_win, height, width, s_patch_size, t_patch_size, embed_dim
    ):
        """Test the feature 2d to 3d mapping, i.e. test if tokens are correctly mapped from a 2D space to the 3D space."""
        # init a dummy features tensor
        Nh = height // s_patch_size
        Nw = width // s_patch_size
        Nt = time_win // t_patch_size

        # generate a random 2D token set
        toks = torch.randn((1, Nt * Nh * Nw, embed_dim))

        # reshape to a 3D tensor
        temp_ranges = [(t * Nh * Nw, (t + 1) * Nh * Nw) for t in range(Nt)]
        toks_3d = _reshape_2d_to_3d(toks, Nh, Nw, temp_ranges)
        toks, toks_3d = toks.squeeze(dim=0), toks_3d.squeeze(dim=0)

        # check the shape of feat_3d
        expected_shape = (Nt * embed_dim, Nh, Nw)
        assert (
            toks_3d.shape == expected_shape
        ), f"Unexpected shape for feat_3d! Got {toks_3d.shape}, expected {expected_shape}"

        # check if features match after reshaping
        for t in range(Nt):
            for i in range(Nh):
                for j in range(Nw):
                    channel_idxs = torch.arange(t * embed_dim, (t + 1) * embed_dim)
                    # the following two feature vectors should be the same!
                    if not torch.equal(
                        toks_3d[channel_idxs, i, j], toks[(i * Nw + j) + t * Nh * Nw, :]
                    ):
                        raise Exception("feat_3d doesn't correspond to feat!")

        # fetch the central frame range and reshape the central frame features to a 3D tensor
        temp_ranges = [temp_ranges[len(temp_ranges) // 2]]
        toks = torch.randn((1, Nt * Nh * Nw, embed_dim))
        toks_3d = _reshape_2d_to_3d(toks, Nh, Nw, temp_ranges)
        toks, toks_3d = toks.squeeze(dim=0), toks_3d.squeeze(dim=0)

        # check the shape of feat_3d
        expected_shape = (embed_dim, Nh, Nw)
        assert (
            toks_3d.shape == expected_shape
        ), f"Unexpected shape for feat_3d! Got {toks_3d.shape}, expected {expected_shape}"

        # check if features from time step t match after reshaping
        t = time_win // 2
        for i in range(Nh):
            for j in range(Nw):
                if not torch.equal(
                    toks_3d[:, i, j], toks[(i * Nw + j) + t * Nh * Nw, :]
                ):
                    raise Exception("feat_3d doesn't correspond to feat!")


if __name__ == "__main__":
    unittest.main()
