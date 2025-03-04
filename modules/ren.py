import torch
import torch.nn as nn
from typing import Tuple, List


class ResidualEstimationNetwork(nn.Module):
    """Residual Estimation Network (REN)"""

    CHECKPOINT_NAME = "REN"

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        channels: List[int] = [32, 64, 128],
    ) -> None:
        super(ResidualEstimationNetwork, self).__init__()
        print(f"kernels per layer: {channels}", flush=True)

        # initialize down-sampling layers
        self.down_layers = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        for ch in channels:
            self.down_layers += [_DoubleConv(in_channels, ch)]
            in_channels = ch

        self.bottleneck_channels = channels[-1] * 2
        self.bottleneck = _DoubleConv(channels[-1], self.bottleneck_channels)

        # initialize up-sampling layers
        self.up_layers = nn.ModuleList()
        for ch in channels[::-1]:
            self.up_layers += [
                nn.ConvTranspose2d(ch * 2, ch, kernel_size=(2, 2), stride=(2, 2))
            ]
            self.up_layers += [_DoubleConv(ch * 2, ch)]

        # reconstruction
        self.rec_head = nn.Conv2d(channels[0], out_channels, kernel_size=(1, 1))

    def forward_encoder(self, x):
        skip_conns = []
        for down in self.down_layers:
            x = down(x)
            skip_conns += [x]
            x = self.pool(x)
        x = self.bottleneck(x)
        return x, skip_conns

    def forward_decoder(self, x, skip_conns):
        skip_conns = skip_conns[::-1]
        for idx in range(0, len(self.up_layers), 2):
            t_conv = self.up_layers[idx]
            conv = self.up_layers[idx + 1]
            x = t_conv(x)
            skip_conn = skip_conns[idx // 2]
            concat_skip = torch.cat([skip_conn, x], dim=1)
            x = conv(concat_skip)

        # apply the reconstruction head
        return self.rec_head(x)

    def forward(self, x):
        x, skip_conns = self.forward_encoder(x)
        return self.forward_decoder(x, skip_conns)


class _DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(_DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)
