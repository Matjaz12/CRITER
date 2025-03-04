import os
import argparse
import math
import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
import wandb

from modules.crm import load_model
from modules.irm import IterativeRefinementModule
from modules.criter import CRITER
from train import train, SEED
from loss import neg_log_likelihood_gaussian
from datasets.sst import (
    MaskSampler,
    get_dataloaders,
    get_observation_labels,
)
from plot_utils import plot_reconstruction


def train_one_epoch(
    model,
    optimizer,
    dataloaders,
    mask_sampler,
    mode,
    epoch,
    plot_period,
    metadata,
    out_path,
    num_validation_runs=10,
):

    # put model in the appropriate mode and move to the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train() if mode == "train" else model.eval()
    model.to(device)
    mask_sampler.to(device)

    # train or validate the model
    epoch_loss = 0
    num_runs = 1 if mode == "train" else num_validation_runs

    with torch.set_grad_enabled(mode == "train"):
        for batch, (observation, missing_mask) in enumerate(dataloaders[mode]):
            # move tensors to the device
            observation = observation.to(device)
            missing_mask = missing_mask.to(device)
            land_mask = metadata["land_mask"].to(device)

            for idx in range(num_runs):
                # sample a missing mask
                sampled_mask = mask_sampler()

                # make prediction and compute loss
                rec, var = model(
                    observation, missing_mask, land_mask, sampled_mask, metadata
                )
                target = observation[
                    :,
                    metadata["time_win"] // 2,
                    metadata["feat_to_idx"]["measurement"],
                    :,
                    :,
                ]
                ld = neg_log_likelihood_gaussian(
                    target, rec, var, missing_mask, land_mask
                )
                loss = ld["loss_all"]

                # update model parameters
                if mode == "train":
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                if batch < 2 and epoch % plot_period == 0:
                    labels = get_observation_labels()
                    labels += [
                        r"$\mathbf{x}_t$",
                        r"$\tilde{\mathbf{x}}$",
                        r"$\mathbf{\sigma}^{2}$",
                    ]
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
                        var,
                        labels=labels,
                        save_path=os.path.join(
                            out_path, f"{mode}_epoch={epoch}_batch={batch}_run={idx}"
                        ),
                    )
                epoch_loss += loss.item()

        assert not math.isnan(epoch_loss), f"Loss for epoch: {epoch} is nan!"
        epoch_loss = epoch_loss / (num_runs * len(dataloaders[mode]))
        return epoch_loss


def main(args, wandb_run):
    # fetch the dataloaders
    dataloaders, metadata = get_dataloaders(
        batch_size=args.batch_size,
        time_win=args.time_win,
        auxiliary_feat=args.auxiliary_feat,
        cloud_coverage_threshold=args.cloud_coverage_threshold,
        data_path=args.data_path,
    )

    # initialize the mask sampler
    mask_sampler = MaskSampler(
        dataset=dataloaders["train"].dataset,
    )

    observation, missing_mask = next(iter(dataloaders["train"]))
    _, time_win, channels, height, width = observation.shape
    print(
        f"observation.shape: {observation.shape} (batch, time_win, channels, height, width)"
    )
    print(f"missing_mask.shape: {missing_mask.shape} (batch_size, height, width)")
    print(f"feat_to_idx: {metadata['feat_to_idx']}")

    # initialize the CRITER model
    extraction_layer = 11
    crm = load_model(
        (height, width),
        channels,
        time_win,
        args.s_patch_size,
        args.t_patch_size,
        args.model_path,
        extraction_layer,
    )

    irm = IterativeRefinementModule(
        time_win,
        2,
        [32, 64, 128],
        args.num_refinement_steps,
        crm.decoder.embed_dim,
        crm.num_patches_h,
        crm.num_patches_w,
        crm.num_patches_t,
    )

    # initialize the model
    model = CRITER(crm, irm)

    # NOTE: It seems like we need to re-seed after model initialization
    # to ensure deterministic batches across different models (with different number of parameters).
    # https://discuss.pytorch.org/t/shuffle-issue-in-dataloader-how-to-get-the-same-data-shuffle-results-with-fixed-seed-but-different-network/45357/6
    torch.manual_seed(SEED)

    # train the model
    del dataloaders["test"]
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=0)
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=0.5)
    train(
        model,
        optimizer,
        scheduler,
        dataloaders,
        metadata,
        train_one_epoch,
        mask_sampler,
        args.n_epochs,
        args.plot_period,
        args.out_path,
        wandb_run,
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
        "--model_path", type=str, default=None, help="Path to the pretrained MAE model"
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

    # training arguments
    parser.add_argument(
        "--n_epochs", type=int, default=80, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Number of samples in a single batch"
    )
    parser.add_argument(
        "--step_size",
        type=int,
        default=20,
        help="Step size of the learning rate scheduler",
    )

    # other arguments
    parser.add_argument(
        "--plot_period",
        type=int,
        default=20,
        help="Period at which reconstruction plots are saved",
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
    # parse training arguments
    args = parse_arguments()
    config = {arg: getattr(args, arg) for arg in vars(args)}
    print(config, flush=True)

    wandb_run = wandb.init(project="CRITER", config=config, mode="disabled")
    # wandb_run = wandb.init(
    #     project="CRITER", config=config, settings=wandb.Settings(_service_wait=300)
    # )
    main(args, wandb_run)
    wandb_run.finish()
