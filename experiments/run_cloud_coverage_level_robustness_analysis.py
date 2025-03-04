import os
import sys
import argparse
import torch
import numpy as np

sys.path.append("../")
from modules.criter import load_model
from loss import mean_squared_error
from datasets.sst import MaskSampler, get_dataloaders
from analysis import CloudCoverageLevelRobustnessAnalysis


@torch.no_grad()
def evaluate(
    model,
    test_dataloader,
    mask_sampler,
    metadata,
    out_path,
    num_test_runs=10,
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

    cclra = CloudCoverageLevelRobustnessAnalysis(
        ["RMSE_all", "RMSE_del", "RMSE_vis"], group_max_coverage=[0.6, 0.75]
    )
    for observation, missing_mask in test_dataloader:
        # move tensors to the device
        observation = observation.to(device)
        missing_mask = missing_mask.to(device)
        land_mask = metadata["land_mask"].to(device)

        for idx in range(num_test_runs):
            # sample a missing mask
            sampled_mask = mask_sampler.sample_valid_mask(land_mask, missing_mask)

            # make prediction and estimate error
            rec, var = model(
                observation,
                missing_mask,
                land_mask,
                sampled_mask,
                metadata,
                inference=True,
            )
            target = observation[
                :,
                metadata["time_win"] // 2,
                metadata["feat_to_idx"]["measurement"],
                :,
                :,
            ]
            ld = mean_squared_error(
                target,
                rec,
                missing_mask,
                land_mask,
                sampled_mask,
            )

            data = {
                "RMSE_all": np.sqrt(ld["mse_all"].item()).mean(),
                "RMSE_del": np.sqrt(ld["mse_hid"].item()).mean(),
                "RMSE_vis": np.sqrt(ld["mse_vis"].item()).mean(),
            }
            cclra.update(data, sampled_mask, missing_mask, land_mask)

    print(f"exporting CCE result...", flush=True)
    cclra.export(os.path.join(out_path, "cce_result.json"))


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
        num_iterations=args.num_refinement_steps,
        rm_layers=[32, 64, 128],
        extraction_layer=11,
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
    parser.add_argument(
        "--num_refinement_steps", type=int, default=1, help="Number of refinement steps"
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
