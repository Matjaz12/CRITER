import time
import os
import torch
import random
import numpy as np

# set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# set torch cuDNN configurations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def train(
    model,
    optimizer,
    scheduler,
    dataloaders,
    metadata,
    train_one_epoch,
    mask_sampler,
    n_epochs,
    plot_period,
    out_path,
    wandb_run,
):
    """
    Train the model by running `train_one_epoch` `n_epoch` times.
    :param model: the model to train
    :param optimizer: the optimizer used to update model parameters
    :param scheduler: the learning rate scheduler
    :param dataloaders: a dict. of dataloaders with keys: ["train", "val"]
    :param metadata: metadata dictionary
    :param train_one_epoch: function that trains the model for one epoch
    :param mask_sampler: cloud mask sampler
    :param n_epochs: number of epochs to train for
    :param plot_period: a period of epochs after which we plot the reconstruction results
    :param out_path: path where model and logs are saved
    :param wandb_run: wandb logger associated with the experiment
    """

    print(
        f"training on device:{'cuda' if torch.cuda.is_available() else 'cpu'} for n_epochs:{n_epochs}",
        flush=True,
    )

    min_val_loss = float("inf")
    for epoch in range(n_epochs):
        loss = {"train": 0, "val": 0}

        # train and validate the model
        t0 = time.time()
        for mode in loss.keys():
            epoch__ = epoch + 1 if epoch == n_epochs - 1 else epoch
            loss[mode] = train_one_epoch(
                model,
                optimizer,
                dataloaders,
                mask_sampler,
                mode,
                epoch__,
                plot_period,
                metadata,
                out_path,
            )

        if scheduler:
            scheduler.step()

        # save the train stats for current epoch
        print(
            f"epoch: {epoch} \t dt: {time.time() - t0}[sec] \t train_loss: {loss['train']} \t val_loss: {loss['val']}",
            flush=True,
        )
        wandb_run.log({"train_loss": loss["train"], "val_loss": loss["val"]})

        # if val loss has decreased save the model
        if loss["val"] < min_val_loss:
            min_val_loss = loss["val"]
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "opt": optimizer.state_dict(),
                "loss": loss,
            }
            torch.save(
                checkpoint, os.path.join(out_path, f"{model.CHECKPOINT_NAME}.pt")
            )
