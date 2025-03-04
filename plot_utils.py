import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from loss import _masked_squared_error

plt.style.use("seaborn-v0_8-paper")
plt.rcParams["axes.titlesize"] = 18


def plot_reconstruction(
    measurement_min,
    measurement_max,
    observation,
    target,
    mask_all,
    mask_land,
    rec,
    var=None,
    labels=[],
    save_path="",
):
    settings = {"interpolation": "none", "aspect": "auto"}
    cmap = "viridis"

    # compute the squared error
    se = _masked_squared_error(rec, target, mask_all)

    # normalize the plots using measurement_min and measurement_max
    norm = plt.Normalize(vmin=measurement_min, vmax=measurement_max)

    # plot all samples in the batch
    rec = rec * mask_land  # remove locations corresponding to the land
    batch_size, in_channels, _, _ = observation.shape
    for idx in range(batch_size):
        ncols = in_channels + 3 + (var is not None)
        fig, axes = plt.subplots(nrows=1, ncols=ncols, figsize=(20, 3))

        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])

        # display the input
        for i in range(in_channels):
            img = _to_img(observation[idx, i, :, :])
            im = axes[i].imshow(img, cmap=cmap, norm=norm, **settings)
            axes[i].set_title(labels[i])

        # display the target
        img = _to_img(target[idx])
        axes[in_channels].imshow(img, cmap=cmap, norm=norm, **settings)
        axes[in_channels].set_title(labels[in_channels])

        # display the reconstruction
        img = _to_img(rec[idx])
        axes[in_channels + 1].imshow(img, cmap=cmap, norm=norm, **settings)
        axes[in_channels + 1].set_title(labels[in_channels + 1])

        if var is not None:
            # display variance
            img = _to_img(var[idx] * mask_land)
            axes[in_channels + 2].imshow(img, **settings)
            axes[in_channels + 2].set_title(labels[in_channels + 2])

            # display the squared error
            img = _to_img(torch.sqrt(se[idx]))
            im = axes[in_channels + 3].imshow(img, **settings)
            fig.colorbar(im, ax=axes[in_channels + 3], fraction=0.1)
            rmse = torch.sqrt(se[idx].sum() / torch.count_nonzero(mask_all[idx]))
            axes[in_channels + 3].set_title(
                r"RMSE_all" + f"={rmse:.3f}" + r"[$\deg$]"
            )

        else:
            # display the squared error
            img = _to_img(torch.sqrt(se[idx]))
            im = axes[in_channels + 2].imshow(img, **settings)
            fig.colorbar(im, ax=axes[in_channels + 2], fraction=0.1)
            rmse = torch.sqrt(se[idx].sum() / torch.count_nonzero(mask_all[idx]))
            axes[in_channels + 2].set_title(
                r"RMSE_all$" + f"={rmse:.3f}" + r"[$\deg$]"
            )

        plt.tight_layout()
        # plt.show()
        save_path__ = f"{save_path}_{idx}" if batch_size > 1 else save_path
        plt.savefig(save_path__)
        plt.close(fig)


def _to_img(img):
    if isinstance(img, torch.Tensor):
        img = img.cpu().detach().numpy()
    img = np.flipud(img)
    return np.ma.masked_where(img == 0, img)
