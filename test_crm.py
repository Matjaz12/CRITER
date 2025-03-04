import os
import argparse
import torch
import numpy as np

from modules.crm import load_model
from loss import mean_squared_error
from datasets.sst import (
    MaskSampler,
    get_dataloaders,
    get_observation_labels,
)

from plot_utils import plot_reconstruction


@torch.no_grad()
def evaluate(
    model,
    test_dataloader,
    mask_sampler,
    metadata,
    out_path,
    num_test_runs=10,
    plot_results=False,
):
    observation, missing_mask = next(iter(test_dataloader))
    assert (
        observation.shape[0] == 1 and missing_mask.shape[0] == 1
    ), "Evaluation batch size should be 1!"

    # put model in eval mode and move to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    mask_sampler.to(device)

    logs = {"mse_all": [], "mse_hid": [], "mse_vis": []}
    for sample, (observation, missing_mask) in enumerate(test_dataloader):
        # move tensors to the device
        observation = observation.to(device)
        missing_mask = missing_mask.to(device)
        land_mask = metadata["land_mask"].to(device)

        for idx in range(num_test_runs):
            # sample a missing mask
            sampled_mask = mask_sampler.sample_valid_mask(land_mask, missing_mask)
            sampled_mask = sampled_mask * missing_mask.squeeze(dim=0)

            # make prediction and estimate error
            rec, _ = model(observation, sampled_mask, metadata)
            target = observation[
                :,
                metadata["time_win"] // 2,
                metadata["feat_to_idx"]["measurement"],
                :,
                :,
            ]
            ld = mean_squared_error(target, rec, missing_mask, land_mask, sampled_mask)

            # plot reconstruction
            if plot_results:
                labels = get_observation_labels()
                labels += [r"$\mathbf{x}_t$", r"$\hat{\mathbf{x}}_t$"]
                observation__ = observation.clone()
                observation__ = observation__[
                    :, :, metadata["feat_to_idx"]["measurement"], :, :
                ]
                measurement_min = observation__[observation__ != 0].min()
                measurement_max = observation__.max()
                observation__[:, metadata["time_win"] // 2, :, :] *= sampled_mask
                plot_reconstruction(
                    measurement_min,
                    measurement_max,
                    observation__,
                    target,
                    ld["M_all"],
                    land_mask,
                    rec,
                    labels=labels,
                    save_path=os.path.join(out_path, f"test_sample={sample}_run={idx}"),
                )

            # accumulate the error over the samples and runs
            logs["mse_all"] += [ld["mse_all"].item()]
            logs["mse_hid"] += [ld["mse_hid"].item()]
            logs["mse_vis"] += [ld["mse_vis"].item()]

    # print the number of testcases
    print(f"The RMSE is estimated using {len(logs['mse_all'])} testcases.", flush=True)

    # print mean RMSE over the three regions of interest
    for k, mse in logs.items():
        print(f"r{k}: {np.sqrt(mse).mean()}")


def main(args):
    # fetch the dataloaders
    dataloaders, metadata = get_dataloaders(
        batch_size=1,
        time_win=args.time_win,
        auxiliary_feat=args.auxiliary_feat,
        cloud_coverage_threshold=args.cloud_coverage_threshold,
        data_path=args.data_path,
    )
    del dataloaders["train"]
    del dataloaders["val"]

    # initialize the mask sampler
    mask_sampler = MaskSampler(dataset=dataloaders["test"].dataset)

    # load the pretrained model
    observation, _ = next(iter(dataloaders["test"]))
    _, time_win, channels, height, width = observation.shape
    model = load_model(
        (height, width),
        channels,
        time_win,
        args.s_patch_size,
        args.t_patch_size,
        args.model_path,
    )

    # evaluate the performance of the model
    evaluate(
        model,
        dataloaders["test"],
        mask_sampler,
        metadata,
        args.out_path,
    )


def parse_arguments():
    parser = argparse.ArgumentParser()

    # data arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to the dataset in the .nc format",
    )
    parser.add_argument(
        "--time_win", type=int, default=3, help="Number of frames in a single sample"
    )
    parser.add_argument(
        "--auxiliary_feat",
        action="store_true",
        default=True,
        help="Enable the usage of auxiliary features",
    )
    parser.add_argument(
        "--cloud_coverage_threshold",
        type=float,
        default=1,
        help="Threshold used to filter observation fields with cloud coverage above it",
    )

    # model arguments
    parser.add_argument(
        "--model_path", type=str, default=None, help="Path to the pretrained model"
    )
    parser.add_argument(
        "--s_patch_size", type=int, default=8, help="Spatial patch size"
    )
    parser.add_argument(
        "--t_patch_size", type=int, default=1, help="Temporal patch size"
    )

    # log arguments
    parser.add_argument(
        "--out_path",
        type=str,
        default="./output",
        help="Path to the output directory",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # parse test arguments
    args = parse_arguments()
    config = {arg: getattr(args, arg) for arg in vars(args)}
    print(config, flush=True)
    main(args)
