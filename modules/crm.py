from functools import partial
import torch
import torch.nn as nn
from typing import List, Tuple
from modules.util.pos_embed import get_3d_sincos_pos_embed
from modules.util.video_vit import PatchEmbed, TransformerBlock
import sys

sys.path.append("../")


class CoarseReconstructionModule(nn.Module):
    """
    Coarse Reconstruction Module (CRM).
    (See: https://arxiv.org/abs/2205.09113, https://github.com/facebookresearch/mae_st)
    """

    CHECKPOINT_NAME = "CRM"

    def __init__(
        self,
        img_size: Tuple[int, int] = (256, 256),
        channels: int = 3,
        time_win: int = 3,
        s_patch_size: int = 16,
        t_patch_size: int = 1,
        encoder_depth: int = 12,
        encoder_num_heads: int = 3,
        decoder_depth: int = 12,
        decoder_num_heads: int = 3,
        embed_dim: int = 192,
        extraction_layer: int = -1,
    ) -> None:
        """
        :param img_size: (height, width) of the image
        :param channels: number of channels / number of features
        :param time_win: number of time steps
        :param s_patch_size: height and width of the patch
        :param t_patch_size: temporal extent of the patch
        :param encoder_depth: number of encoder transformer blocks
        :param encoder_num_heads: number of encoder heads in the multi-head attention
        :param decoder_depth: number of decoder transformer blocks
        :param decoder_num_heads: number of decoder heads in the multi-head attention
        :param embed_dim: dimension of each token
        :param extraction_layer: layer at which decoder tokens are extracted
        """
        super(CoarseReconstructionModule, self).__init__()
        self.img_size = img_size
        self.channels = channels
        self.time_win = time_win
        self.s_patch_size = s_patch_size
        self.t_patch_size = t_patch_size

        # initialize encoder
        self.encoder = Encoder(
            img_size,
            channels,
            time_win,
            s_patch_size,
            t_patch_size,
            encoder_depth,
            encoder_num_heads,
            embed_dim,
        )

        # initialize decoder
        self.decoder = Decoder(
            channels,
            s_patch_size,
            t_patch_size,
            self.encoder.num_patches,
            decoder_depth,
            decoder_num_heads,
            embed_dim,
            extraction_layer,
        )

        # store number of patches along time, height and width
        self.num_patches_t, self.num_patches_h, self.num_patches_w = (
            self.encoder.get_number_of_patches()
        )
        print(
            f"num_patches_t: {self.num_patches_t} \t num_patches_h: {self.num_patches_h} \t num_patches_w: {self.num_patches_w}",
            flush=True,
        )

        self.initialize_weights()

    def initialize_weights(self):
        # initialize encoder positional embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.encoder.pos_embed.shape[-1],
            self.num_patches_h,
            self.num_patches_w,
            self.num_patches_t,
            cls_token=False,
        )

        self.encoder.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # initialize decoder positional embedding
        pos_embed = get_3d_sincos_pos_embed(
            self.decoder.pos_embed.shape[-1],
            self.num_patches_h,
            self.num_patches_w,
            self.num_patches_t,
            cls_token=False,
        )

        self.decoder.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0)
        )

        # initialize patch embedding weights
        w = self.encoder.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize linear and layer norm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def unpatchify(self, x):
        """Reshape to original shape"""
        x = x.reshape(
            (
                x.shape[0],
                self.num_patches_t,
                self.num_patches_h,
                self.num_patches_w,
                self.t_patch_size,
                self.s_patch_size,
                self.s_patch_size,
                self.channels,
            )
        )

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        return x.reshape(
            (
                x.shape[0],
                self.channels,
                self.time_win,
                self.img_size[0],
                self.img_size[1],
            )
        )

    def forward(
        self,
        observation: torch.Tensor,
        sampled_mask: torch.Tensor,
        metadata: dict,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate coarse reconstruction and extract decoder tokens
        :param observation: observation tensor, of shape (batch, time_win, channels, height, width)
        :param sampled_mask: sampled missing mask, of shape (height, width)
        :param metadata: metadata dictionary
        :return rec: coarse reconstruction
        :return toks_extracted: extracted tokens
        """

        # conceal the central measurement with the provided cloud mask
        observation__ = observation.clone()
        observation__[
            :, metadata["time_win"] // 2, metadata["feat_to_idx"]["measurement"], :, :
        ] *= sampled_mask

        # apply the encoder
        observation__ = observation__.permute(0, 2, 1, 3, 4)
        toks_context, toks_rec, ids_restore = self.encoder(observation__, sampled_mask)

        # apply the decoder
        rec, toks_extracted = self.decoder(toks_context, toks_rec, ids_restore)
        rec = self.unpatchify(rec)

        # extract coarse reconstruction
        rec = rec.permute(0, 2, 1, 3, 4)
        rec = rec[
            :, metadata["time_win"] // 2, metadata["feat_to_idx"]["measurement"], :, :
        ]
        return (rec, toks_extracted)


class Encoder(nn.Module):
    """Encoder with ViT backbone."""

    def __init__(
        self,
        img_size: Tuple[int, int],
        channels: int,
        time_win: int,
        s_patch_size: int,
        t_patch_size: int,
        depth: int,
        num_heads: int,
        embed_dim: int,
        mlp_ratio: int = 4.0,
    ):
        super(Encoder, self).__init__()

        # initialize patch embedding
        self.s_patch_size = s_patch_size
        self.t_patch_size = t_patch_size
        self.patch_embed = PatchEmbed(
            img_size, s_patch_size, channels, embed_dim, time_win, t_patch_size
        )

        # initialize mask embedding
        self.mask_embed = PatchEmbed(
            img_size, s_patch_size, 1, embed_dim, 1, t_patch_size
        )

        # initialize positional embedding
        self.num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, embed_dim), requires_grad=False
        )

        # initialize transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(depth):
            block = TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            self.transformer_blocks += [block]

        # initialize layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

    def forward(self, observation__, sampled_mask):
        # compute context and reconstruction tokens
        toks_context, toks_rec, ids_restore = self.contextual_mask_injection(
            observation__, sampled_mask
        )

        # apply transformer blocks to context tokens
        for block in self.transformer_blocks:
            toks_context = block(toks_context)

        toks_context = self.norm(toks_context)
        return toks_context, toks_rec, ids_restore

    def contextual_mask_injection(self, observation__, mask):
        """Compute a set of context and reconstruction tokens, given a concealed observation and mask"""

        # apply patch embedding to the concealed observation
        toks_all = self.patch_embed(observation__)
        batch, num_patches_t, num_patches_s, embed_dim = toks_all.shape
        num_patches = num_patches_t * num_patches_s
        toks_all = toks_all.view((batch, num_patches, embed_dim))

        # compute mask tokens
        toks_mask = self.mask_embed(mask.view((1, 1, 1, mask.shape[0], mask.shape[1])))
        toks_mask = toks_mask.squeeze(dim=1)

        # add mask tokens to the corresponding reconstruction tokens
        ids_rec_mask = _get_rec_indices(mask, self.s_patch_size)
        ids_rec = ids_rec_mask + (num_patches_t // 2) * num_patches_s
        toks_all[:, ids_rec, :] += toks_mask[:, ids_rec_mask, :]

        # add positional embedding
        toks_all = toks_all + self.pos_embed

        # split tokens
        ids_all = torch.arange(0, num_patches, device=toks_all.device)
        ids_context = torch.masked_select(
            ids_all, torch.logical_not(torch.isin(ids_all, ids_rec))
        )
        toks_context = toks_all[:, ids_context, :]
        toks_rec = toks_all[:, ids_rec, :]

        # construct ids_restore
        ids_shuffle = -1 * torch.ones(
            (num_patches,), dtype=torch.int64, device=toks_all.device
        )
        ids_shuffle[: len(ids_context)] = ids_context
        ids_shuffle[len(ids_context) :] = ids_rec
        ids_restore = torch.argsort(ids_shuffle)
        ids_restore = ids_restore.expand(batch, -1)

        return toks_context, toks_rec, ids_restore

    def get_number_of_patches(self) -> Tuple[int, int, int]:
        """Get number of patches"""
        num_patches_t = self.patch_embed.t_grid_size
        num_patches_h = self.patch_embed.grid_size_h
        num_patches_w = self.patch_embed.grid_size_w
        return (num_patches_t, num_patches_h, num_patches_w)


class Decoder(nn.Module):
    """Decoder with ViT backbone."""

    def __init__(
        self,
        channels: int,
        s_patch_size: int,
        t_patch_size: int,
        num_patches: int,
        depth: int,
        num_heads: int,
        embed_dim: int,
        extraction_layer: int,
        mlp_ratio: int = 4.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.extraction_layer = extraction_layer
        assert (
            self.extraction_layer < depth
        ), f"Extraction layer out of range! The model has {depth} layers."

        # initialize positional embedding
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, embed_dim), requires_grad=False
        )

        # initialize transformer blocks
        self.transformer_blocks = nn.ModuleList()
        for _ in range(depth):
            block = TransformerBlock(
                embed_dim,
                num_heads,
                mlp_ratio,
                qkv_bias=True,
                qk_scale=None,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
            )
            self.transformer_blocks += [block]

        # initialize layer norm
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # initialize linear projection
        self.lin_proj = nn.Linear(
            embed_dim, t_patch_size * (channels * (s_patch_size**2)), bias=True
        )

    def forward(self, toks_context, toks_rec, ids_restore):
        # combine context and reconstruction tokens
        toks_all = torch.cat([toks_context, toks_rec], dim=1)
        index = ids_restore.unsqueeze(-1).repeat(1, 1, toks_all.shape[2])
        toks_all = torch.gather(toks_all, dim=1, index=index)

        # add positional embedding
        toks_all = toks_all + self.pos_embed

        # apply transformer blocks to all tokens
        toks_extracted = None
        for l, block in enumerate(self.transformer_blocks):
            toks_all = block(toks_all)

            if l == self.extraction_layer:
                # store the tokens at current layer
                toks_extracted = toks_all.clone()

        # apply linear projection
        toks_all = self.norm(toks_all)
        patches_all = self.lin_proj(toks_all)
        return patches_all, toks_extracted


def _get_rec_indices(mask, patch_size):
    """Return a list of 1D indices corresponding to the reconstruction tokens."""
    # calculate the number of patches along the height and width
    num_patches_height, num_patches_width = (
        mask.shape[0] // patch_size,
        mask.shape[1] // patch_size,
    )

    # reshape the mask to a set of non-overlapping patches.
    # each patch is of shape (patch_size, patch_size)
    reshaped_mask = mask.reshape(
        num_patches_height, patch_size, num_patches_width, patch_size
    )

    # compute the 2D index of reconstruction patches
    min_values = reshaped_mask.amin(dim=(1, 3))
    patch_idxs = torch.argwhere(min_values == 0)

    # return 1D index of reconstruction patches
    return patch_idxs[:, 0] * num_patches_width + patch_idxs[:, 1]


def load_model(
    img_size: Tuple[int, int],
    channels: int,
    time_win: int,
    s_patch_size: int,
    t_patch_size: int,
    model_path: str = None,
    extraction_layer: int = -1,
) -> CoarseReconstructionModule:
    """
    :param img_size: (height, width) of the image
    :param channels: number of channels / number of features
    :param time_win: number of time steps
    :param s_patch_size: height and width of the patch
    :param t_patch_size: temporal extent of the patch
    :param model_path: path to the pre-trained model
    :param extraction_layer: layer at which decoder tokens are extracted
    """
    # initialize the model
    model = CoarseReconstructionModule(
        img_size,
        channels,
        time_win,
        s_patch_size,
        t_patch_size,
        extraction_layer=extraction_layer,
    )

    if model_path:
        # load the parameters
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "loss" in checkpoint.keys():
            print(f"loaded model (loss): {checkpoint['loss']}")
    return model
