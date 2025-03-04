import math
import torch
import torch.nn as nn
from typing import Tuple, List
from modules.ffm import FeatureFusionModule
from modules.ren import ResidualEstimationNetwork


class IterativeRefinementModule(nn.Module):
    """Iterative Refinement Module (IRM)"""

    CHECKPOINT_NAME = "IRM"

    def __init__(
        self,
        in_channels: int,
        out_channels: int = 2,
        channels: List[int] = [32, 64, 128],
        num_iterations: int = 1,
        embed_dim: int = 192,
        num_patches_h: int = 32,
        num_patches_w: int = 32,
        num_patches_t: int = 1,
    ) -> None:
        """
        :param in_channels: number of input channels
        :param out_channels: number of output channels
        :param channels: number of kernels per layer
        :param num_iterations: number of refinement iterations
        :param embed_dim: dimension of each token
        :param num_patches_h: number of patches along height
        :param num_patches_w: number of patches along width
        :param num_patches_t: number of patches along the temporal dimension
        """
        super(IterativeRefinementModule, self).__init__()
        self.num_iterations = num_iterations
        self.theta1, self.theta2 = _scale_thetas(self.num_iterations)
        print(f"num of refinement iterations: {self.num_iterations}", flush=True)
        print(f"theta1, theta2 = {self.theta1, self.theta2}", flush=True)

        # initialize individual Refinement and Feature Fusion Modules
        self.rms = nn.ModuleList()
        self.ffms = nn.ModuleList()

        for _ in range(self.num_iterations):
            self.rms += [
                ResidualEstimationNetwork(
                    in_channels + 2,
                    out_channels,
                    channels,
                )
            ]
            self.ffms += [
                FeatureFusionModule(
                    self.rms[0].bottleneck_channels,
                    embed_dim,
                    num_patches_h,
                    num_patches_w,
                    num_patches_t,
                )
            ]

    def forward(
        self,
        rec,
        toks,
        observation,
        land_mask,
        mask,
        metadata,
    ):
        """Compute the refined reconstruction and estimate the corresponding variance."""
        observation__ = observation.clone()

        # prepare input for the refinement stage
        batch_size, _, _, height, width = observation__.shape
        observation__ = observation__[
            :, :, metadata["feat_to_idx"]["measurement"], :, :
        ]
        observation__[:, metadata["time_win"] // 2, :, :] *= mask
        zeros = torch.zeros((batch_size, 2, height, width)).to(observation__.device)
        observation__ = torch.cat([observation__, zeros], dim=1)

        # apply num_iterations of refinement
        var = torch.zeros_like(rec)
        for i in range(self.num_iterations):
            # pass reconstruction and inverse of variance from the previous refinement iteration
            # as the input to the current iteration
            observation___ = observation__.clone()
            observation___[:, -2, :, :] = rec
            observation___[:, -1, :, :] = var

            # estimate the residual reconstruction and residual variance
            x, skip_connections = self.rms[i].forward_encoder(observation___)
            x = self.ffms[i].forward(x, toks)
            Y_hat = self.rms[i].forward_decoder(x, skip_connections)
            residual_rec, residual_var = _estimate_residual_rec_and_var(
                Y_hat, self.theta1, self.theta2
            )

            # update the reconstruction and variance estimate
            rec = rec + residual_rec * land_mask
            var = var + residual_var

        return (rec, var)


def _estimate_residual_rec_and_var(Y_hat, theta1, theta2):
    """
    Estimate the residual reconstruction and variance.
    :param Y_hat: tensor of shape (batch_size, 2, height, width)
    """
    T0, T1 = Y_hat[:, 0, :, :], Y_hat[:, 1, :, :]
    min_vals = torch.min(T1, theta1 * torch.ones_like(T1))
    exp_vals = torch.exp(min_vals)
    var_hat = 1 / torch.max(exp_vals, theta2 * torch.ones_like(min_vals))
    return T0 * var_hat, var_hat


def _scale_thetas(num_iterations, theta1_=10, theta2_=1e-3):
    """Compute the scaled theta1 and theta2, by taking into account the number of refinement iterations."""
    gamma = math.log(num_iterations) + theta1_
    mu = num_iterations * theta2_
    return (gamma, mu)
