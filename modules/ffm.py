import torch
import torch.nn as nn
import torchvision.transforms.functional as F


class FeatureFusionModule(torch.nn.Module):
    """
    Feature Fusion Module (FFM)
    (See: https://arxiv.org/abs/1808.00897)
    """

    def __init__(
        self,
        in_channels: int,
        embed_dim: int,
        num_patches_h: int,
        num_patches_w: int,
        num_patches_t: int,
    ) -> None:
        """
        :param in_channels: number of channels, corresponding to the bottleneck features
        :param embed_dim: dimension of each token
        :param num_patches_h: number of patches along height
        :param num_patches_w: number of patches along width
        :param num_patches_t: number of patches along the temporal dimension
        """
        super(FeatureFusionModule, self).__init__()

        self.channels = in_channels
        self.embed_dim = embed_dim

        # compute temporal range
        self.num_patches_h = num_patches_h
        self.num_patches_w = num_patches_w
        self.num_patches_t = num_patches_t
        self.temp_range = [
            (
                (num_patches_t // 2) * num_patches_h * num_patches_w,
                ((num_patches_t // 2) + 1) * num_patches_h * num_patches_w,
            )
        ]

        self.conv_block = _ConvBlock(self.channels + self.embed_dim, self.channels)
        self.conv1 = nn.Conv2d(self.channels, self.channels, kernel_size=(1, 1))
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(self.channels, self.channels, kernel_size=(1, 1))
        self.sigmoid = nn.Sigmoid()
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, feat, toks):
        # extract tokens within the temporal range and resize to a 3D tensor
        toks = _reshape_2d_to_3d(
            toks, self.num_patches_h, self.num_patches_w, self.temp_range
        )
        toks = F.resize(toks, feat.shape[2:], antialias=True)

        # apply the BiSeNet feature fusion
        return self._forward(feat, toks)

    def _forward(self, feat, toks_3d):
        # concatenate the bottleneck features and 3D tokens, follow by the conv. block
        feat = torch.cat([feat, toks_3d], dim=1)
        feat = self.conv_block(feat)

        # compute a weight vector (a value between zero and one for each channel)
        w = self.avg_pool(feat)
        w = self.relu(self.conv1(w))
        w = self.sigmoid(self.conv2(w))

        # multiply-in the weight vector and add original features
        return torch.add(torch.mul(feat, w), feat)


def _reshape_2d_to_3d(feat, H, W, t_ranges):
    """Reshape to a 3D dimensional tensor"""
    B, _, D = feat.shape
    return torch.cat(
        [
            feat[:, start:end, :].view(B, H, W, D).permute(0, 3, 1, 2)
            for (start, end) in t_ranges
        ],
        dim=1,
    )


class _ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, input):
        x = self.conv1(input)
        return self.relu(self.bn(x))
