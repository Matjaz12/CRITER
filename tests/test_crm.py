import sys
import unittest
from parameterized import parameterized

import torch
import torch.nn as nn
from functools import partial
import numpy as np

sys.path.append("../")
from modules.crm import (
    CoarseReconstructionModule,
    Encoder,
    Decoder,
    _get_rec_indices,
)
from modules.util.video_vit import PatchEmbed, TransformerBlock
from modules.util.pos_embed import get_3d_sincos_pos_embed


# set random seed for reproducibility
SEED = 42
np.random.seed(SEED)


def generate_perlin_noise(height, width, resolutions=[(2, 4), (4, 8), (8, 8), (8, 16)]):
    """Generates a perlin noise image of size (height, width)"""

    for res in resolutions:
        assert (
            height % res[0] == 0 and width % res[1] == 0
        ), """
        Height and width should be divisible by Perlin noise resolution!
        Resolutions for different regions:
        - Mediterranean, Atlantic: [(2, 4), (4, 8), (8, 8), (8, 16)]
        - Adriatic: [(2, 4), (4, 6), (6, 6), (6, 12)]
        """

    idx = np.random.randint(0, len(resolutions))
    noise = _generate_perlin_noise_2d((height, width), resolutions[idx])
    return torch.tensor((noise > 0).astype(float)[:height, :width])


def _generate_perlin_noise_2d(shape, res):
    """Perlin Noise Generator (See: https://github.com/VitjanZ/DRAEM/blob/main/perlin.py)"""

    def f(t):
        return 6 * t**5 - 15 * t**4 + 10 * t**3

    delta = (res[0] / shape[0], res[1] / shape[1])
    d = (shape[0] // res[0], shape[1] // res[1])
    grid = np.mgrid[0 : res[0] : delta[0], 0 : res[1] : delta[1]].transpose(1, 2, 0) % 1

    # Gradients
    angles = 2 * np.pi * np.random.rand(res[0] + 1, res[1] + 1)
    gradients = np.dstack((np.cos(angles), np.sin(angles)))
    g00 = gradients[0:-1, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g10 = gradients[1:, 0:-1].repeat(d[0], 0).repeat(d[1], 1)
    g01 = gradients[0:-1, 1:].repeat(d[0], 0).repeat(d[1], 1)
    g11 = gradients[1:, 1:].repeat(d[0], 0).repeat(d[1], 1)

    # Ramps
    n00 = np.sum(grid * g00, 2)
    n10 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1])) * g10, 2)
    n01 = np.sum(np.dstack((grid[:, :, 0], grid[:, :, 1] - 1)) * g01, 2)
    n11 = np.sum(np.dstack((grid[:, :, 0] - 1, grid[:, :, 1] - 1)) * g11, 2)

    # Interpolation
    t = f(grid)
    n0 = n00 * (1 - t[:, :, 0]) + t[:, :, 0] * n10
    n1 = n01 * (1 - t[:, :, 0]) + t[:, :, 0] * n11
    return np.sqrt(2) * ((1 - t[:, :, 1]) * n0 + t[:, :, 1] * n1)


class TestCRM(unittest.TestCase):
    """Test the Coarse Reconstruction Module"""

    @parameterized.expand(
        [
            ((256, 256), 1, 3, 16, 1, 192),
            ((256, 256), 3, 3, 16, 1, 192),
            ((144, 180), 1, 3, 12, 1, 192),
            ((144, 180), 3, 3, 12, 1, 192),
        ]
    )
    def test_patch_embedding(
        self, img_size, in_channels, num_frames, s_patch_size, t_patch_size, embed_dim
    ):
        """
        Check if patch embedding is correct, i.e., check if output of patch embedding is of correct shape,
        check computation of patch embeddings.
        """
        patch_embd = PatchEmbed(
            img_size, s_patch_size, in_channels, embed_dim, num_frames, t_patch_size
        )
        x = torch.randn(1, in_channels, num_frames, img_size[0], img_size[1])
        emb = patch_embd(x)

        # check if embedding is of expected size
        num_patches_temporal = num_frames // t_patch_size
        num_patches_spatial = (img_size[0] // s_patch_size) * (
            img_size[1] // s_patch_size
        )
        assert emb.shape == (1, num_patches_temporal, num_patches_spatial, embed_dim)

        # check that there is exactly `num_patches_spatial` unqiue embeddings for each time step
        unique_embs = torch.unique(emb, dim=2)
        for t in range(num_patches_temporal):
            assert unique_embs[:, t, :, :].shape[1] == num_patches_spatial

        # for each frame generate a unqiue tensor, defined as a constant over space
        x = torch.zeros((1, in_channels, num_frames, img_size[0], img_size[1]))
        emb = patch_embd(x)
        for t in range(num_patches_temporal):
            x[:, :, t, :, :] = t

        unique_embs = torch.unique(emb, dim=2)
        for t in range(num_patches_temporal):
            # we expect one unique embedding for each time-step
            assert unique_embs[:, t, :, :].shape[1] == 1

    @parameterized.expand([(192, 3, 768), (192, 6, 768), (192, 3, 540), (192, 6, 540)])
    def test_transformer_block(self, embed_dim, num_heads, num_tokens):
        """Test transformer block. Checks shapes, and normalization"""
        # init a transformer block
        transformer_block = TransformerBlock(
            embed_dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )

        x = torch.randn((1, num_tokens, embed_dim))
        x_hat = transformer_block(x)

        # check if shape is maintained
        assert x_hat.shape == x.shape

        # check that the mean of each patch is 0, and variance of each patch is 1
        # norm along feature dimension
        x_hat = transformer_block.norm1(x)
        patches_mean = x_hat.mean(dim=2)
        patches_var = x_hat.var(dim=2)
        assert torch.allclose(patches_mean, torch.zeros_like(patches_mean), atol=1e-5)
        assert torch.allclose(patches_var, torch.ones_like(patches_var), atol=1e-2)

        # check that attention preserves shape
        x_hat = transformer_block.attn(x)
        assert x_hat.shape == x.shape

        # check mlp preserves the shapes
        x_hat = transformer_block.mlp(x)
        assert x_hat.shape == x.shape

    @parameterized.expand(
        [
            (192, 256 // 16, 256 // 16, 3 // 1),
            (192, 144 // 12, 180 // 12, 3 // 1),
        ]
    )
    def test_positional_embedding(
        self, embed_dim, num_patches_h, num_patches_w, num_patches_t
    ):
        """
        Check if positional embedding is correct.
        Checks shape, spatial consistency of temporal embedding, temporal consistency of spatial embedding
        """
        num_patches_all = num_patches_h * num_patches_w * num_patches_t
        pos_emb = get_3d_sincos_pos_embed(
            embed_dim, num_patches_h, num_patches_w, num_patches_t
        )
        # check if pos embedding is of expected shape
        assert pos_emb.shape == (num_patches_all, embed_dim)

        # 3d positional embedding is defined as a concatenation of the temporal and spatial embedding
        # pos_emb = [pos_emb_temp; pos_emb_spatial]
        # pos_emb.shape (N, D), pos_emb_temp.shape (N, D//4), pos_emb_spatial.shape (N, 3D//4)

        # check that temporal embedding is a constant over space
        num_patches_spatial = num_patches_w * num_patches_h
        num_patches_all = num_patches_t * num_patches_spatial
        for t in range(num_patches_t):
            start_idx = t * num_patches_spatial
            end_idx = min((t + 1) * num_patches_spatial, num_patches_all)
            pos_emb_temp = pos_emb[start_idx:end_idx, : embed_dim // 4]
            # number of unique temporal embeddings within the current frame should be one
            assert np.unique(pos_emb_temp, axis=0).shape[0] == 1

        # check that positional embedding is a constant over time
        for i in range(num_patches_spatial):
            pos_emb_spatial_i = pos_emb[i, embed_dim // 4 :]
            for t in range(1, num_patches_t):
                assert np.all(
                    pos_emb_spatial_i
                    == pos_emb[i + t * num_patches_spatial, embed_dim // 4 :]
                )

        # check that the combine positional embedding is unique for each patch
        out = np.unique(pos_emb, axis=1)
        assert out.shape[0] == num_patches_all

    @parameterized.expand(
        [
            (8, (256, 256), 1, 3, 16, 1),
            (8, (256, 256), 3, 3, 16, 1),
            (8, (144, 180), 1, 3, 12, 1),
            (8, (144, 180), 3, 3, 12, 1),
        ]
    )
    def test_encoder_output_shapes(
        self, batch_size, img_size, in_channels, num_frames, s_patch_size, t_patch_size
    ):
        """Test the shape of encoder output tokens"""
        num_patches_spatial = (img_size[0] // s_patch_size) * (
            img_size[1] // s_patch_size
        )
        num_patches_all = (num_frames // t_patch_size) * num_patches_spatial

        encoder = Encoder(
            img_size,
            in_channels,
            num_frames,
            s_patch_size,
            t_patch_size,
            depth=3,
            num_heads=1,
            embed_dim=192,
        )

        # mask the entire central frame
        observation = torch.randn(
            (batch_size, in_channels, num_frames, img_size[0], img_size[1])
        )
        mask = torch.zeros((img_size[0], img_size[1]))
        toks_vis, toks_hid, _ = encoder.forward(observation, mask)

        assert toks_vis.shape == (
            batch_size,
            num_patches_all - num_patches_spatial,
            192,
        )

        assert toks_hid.shape == (
            batch_size,
            num_patches_spatial,
            192,
        )

        # mask the first img_size[0]//2 rows of the mask
        mask = torch.ones((img_size[0], img_size[1]))
        mask[: img_size[0] // 2, :] = 0
        toks_vis, toks_hid, _ = encoder.forward(observation, mask)

        assert toks_vis.shape == (
            batch_size,
            num_patches_all - num_patches_spatial // 2,
            192,
        )

        assert toks_hid.shape == (
            batch_size,
            num_patches_spatial // 2,
            192,
        )

        # mask the first patch only
        mask = torch.ones((img_size[0], img_size[1]))
        mask[0, 0] = 0
        toks_vis, toks_hid, _ = encoder.forward(observation, mask)

        assert toks_vis.shape == (
            batch_size,
            num_patches_all - 1,
            192,
        )

        assert toks_hid.shape == (
            batch_size,
            1,
            192,
        )

    @parameterized.expand(
        [
            (8, (256, 256), 1, 3, 16, 1),
            (8, (256, 256), 3, 3, 16, 1),
            (8, (144, 180), 1, 3, 12, 1),
            (8, (144, 180), 3, 3, 12, 1),
        ]
    )
    def test_decoder_output_shapes(
        self, batch_size, img_size, in_channels, num_frames, s_patch_size, t_patch_size
    ):
        """Test encoder output shapes"""
        num_patches_spatial = (img_size[0] // s_patch_size) * (
            img_size[1] // s_patch_size
        )
        num_patches_all = (num_frames // t_patch_size) * num_patches_spatial
        num_pixels_per_patch = in_channels * s_patch_size * s_patch_size

        encoder = Encoder(
            img_size,
            in_channels,
            num_frames,
            s_patch_size,
            t_patch_size,
            depth=3,
            num_heads=1,
            embed_dim=192,
        )

        decoder = Decoder(
            in_channels,
            s_patch_size,
            t_patch_size,
            encoder.num_patches,
            depth=3,
            num_heads=1,
            embed_dim=192,
            extraction_layer=2,
        )

        # mask the entire central frame
        observation = torch.randn(
            (batch_size, in_channels, num_frames, img_size[0], img_size[1])
        )
        mask = torch.zeros((img_size[0], img_size[1]))

        toks_vis, toks_hid, ids_restore = encoder(observation, mask)
        patches_hat, toks_extracted = decoder(toks_vis, toks_hid, ids_restore)

        assert patches_hat.shape == (batch_size, num_patches_all, num_pixels_per_patch)
        assert toks_extracted.shape == (batch_size, num_patches_all, 192)

    @parameterized.expand(
        [
            (3, 256, 256, 16, 1),
            (3, 256, 256, 8, 1),
        ]
    )
    def test_index_restore_complex(
        self,
        time_win,
        height,
        width,
        s_patch_size,
        t_patch_size,
        embed_dim=192,
        n_iters=100,
    ):
        """
        Test the ids_restore procedure with sampled perlin masks.
        Generate a binary mask, compute the ids_remove and ids_keep and
        make sure that ids_restore puts masked tokens to indices specified by the ids_remove set.
        """
        # compute the total number of patches
        num_patches_s = (height // s_patch_size) * (width // s_patch_size)
        num_patches_t = time_win // t_patch_size
        num_patches = num_patches_t * num_patches_s

        for _ in range(n_iters):
            # generate a perlin noise mask
            mask = generate_perlin_noise(height, width)

            # generate a dummy set of tokens
            toks_all = torch.randn((1, num_patches, embed_dim))

            # set mask tokens equal to vectors with infinitely large numbers
            toks_mask = torch.inf * torch.ones((1, num_patches_s, embed_dim))

            # add mask tokens to the corresponding hidden tokens
            ids_hid_mask = _get_rec_indices(mask, s_patch_size)
            ids_hid = ids_hid_mask + (num_patches_t // 2) * num_patches_s
            toks_all[:, ids_hid, :] += toks_mask[:, ids_hid_mask, :]

            # split tokens
            ids_all = torch.arange(0, num_patches, device=toks_all.device)
            ids_vis = torch.masked_select(
                ids_all, torch.logical_not(torch.isin(ids_all, ids_hid))
            )
            toks_vis = toks_all[:, ids_vis, :]
            toks_hid = toks_all[:, ids_hid, :]

            # construct ids_restore
            ids_shuffle = -1 * torch.ones(
                (num_patches,), dtype=torch.int64, device=toks_all.device
            )
            ids_shuffle[: len(ids_vis)] = ids_vis
            ids_shuffle[len(ids_vis) :] = ids_hid
            ids_restore = torch.argsort(ids_shuffle)

            # check that utilizing `ids_restore` puts masked tokens back to the place specified by ids_remove
            toks_all = torch.cat([toks_vis, toks_hid], dim=1)
            index = ids_restore.unsqueeze(-1).repeat(1, 1, toks_all.shape[2])
            toks_all = torch.gather(toks_all, dim=1, index=index)

            hidden_token_indices_out = torch.where(
                torch.all(toks_all == torch.inf, dim=-1)
            )[1]
            assert torch.equal(hidden_token_indices_out, ids_hid)

    @parameterized.expand(
        [
            (4, 3, 3, 256, 256, 16, 1),
            (4, 3, 1, 256, 256, 16, 1),
            (4, 3, 3, 144, 180, 12, 1),
            (4, 3, 1, 144, 180, 12, 1),
        ]
    )
    def test_pred_shapes(
        self,
        batch,
        time_win,
        channels,
        height,
        width,
        s_patch_size,
        t_patch_size,
    ):
        """Test if the output of CRM is of correct shape."""
        # initialize dummy data
        observation = torch.randn((batch, time_win, channels, height, width))
        mask = torch.zeros((height, width))
        mask[: height // 2, : width // 2] = 0
        metadata = {"time_win": time_win, "feat_to_idx": {"measurement": 0}}

        # init the model and predict
        model = CoarseReconstructionModule(
            (height, width), channels, time_win, s_patch_size, t_patch_size
        )
        rec, _ = model(observation, mask, metadata)

        # check if the prediction shape is as expected
        expected_shape = (batch, height, width)
        assert (
            rec.shape == expected_shape
        ), f"Unexpected shape for pred_img! Got {rec.shape}, expected {expected_shape}"

    @parameterized.expand(
        [
            (
                torch.tensor(
                    [[0, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 1, 1]],
                    dtype=torch.int32,
                ),
                2,
                torch.tensor([0, 3]),
            ),
            (
                torch.tensor(
                    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]],
                    dtype=torch.int32,
                ),
                2,
                torch.tensor([2]),
            ),
            (
                torch.tensor(
                    [[1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1], [1, 1, 1, 1]],
                    dtype=torch.int32,
                ),
                2,
                torch.tensor([]),
            ),
            (
                torch.tensor(
                    [
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 0, 1],
                        [1, 1, 1, 1, 1, 1],
                        [1, 1, 0, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1],
                    ],
                    dtype=torch.int32,
                ),
                2,
                torch.tensor([1, 5, 7, 8]),
            ),
            (
                torch.tensor(
                    [
                        [1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 0, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 0, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 0],
                    ],
                    dtype=torch.int32,
                ),
                3,
                torch.tensor([1, 5, 8]),
            ),
            (
                torch.tensor(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 0, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    ],
                    dtype=torch.int32,
                ),
                3,
                torch.tensor([0, 4, 6]),
            ),
            (
                torch.tensor(
                    [
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 0, 1, 1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 0, 0, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 1, 1, 1, 0, 1, 1, 1],
                        [1, 1, 1, 1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 1, 1, 1, 1],
                    ],
                    dtype=torch.int32,
                ),
                8,
                torch.tensor([0]),
            ),
        ]
    )
    def test_get_hidden_indices(self, mask, patch_size, expected_masked_patch_idxs):
        """Test `test_get_hidden_indices` implementation"""
        masked_patch_idxs = _get_rec_indices(mask, patch_size)
        assert torch.equal(
            masked_patch_idxs, expected_masked_patch_idxs
        ), f"masked_patch_idxs:{masked_patch_idxs} does not equal of expected_masked_patch_idxs:{expected_masked_patch_idxs}"


if __name__ == "__main__":
    unittest.main()
