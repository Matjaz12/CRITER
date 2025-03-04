import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from datasets.raw_sst_data import get_raw_sst_data


# set random seed for reproducibility
SEED = 42
np.random.seed(SEED)


class SST_Dataset(Dataset):
    """Pytorch wrapper around the SST data."""

    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return self.data["observation"].shape[0]

    def __getitem__(self, idx):
        observation, missing_mask = (
            self.data["observation"][idx],
            self.data["missing_mask"][idx],
        )
        if self.transform:
            observation = self.transform(observation)
        return observation, missing_mask


class MaskSampler:
    """Class used to sample (cloud) masks from the dataset"""

    def __init__(self, dataset: SST_Dataset) -> None:
        # init. device to CPU
        self._device = torch.device("cpu")

        # store the dataset and pre-compute indices
        self.dataset = dataset
        self.indices = np.arange(len(self.dataset))

    def __call__(self):
        return self.sample_mask()

    def to(self, device: torch.device):
        self._device = device

    def sample_mask(self):
        """Sample a cloud mask"""
        idx = np.random.choice(self.indices, size=1)
        _, mask = self.dataset[idx]
        mask = mask.squeeze(dim=0)
        return mask.to(self._device)

    def sample_valid_mask(self, land_mask, missing_mask, max_attempts=100):
        """
        Sample a valid mask, i.e., mask which conceals at least one pixel,
        mask which keeps at least one visible pixel
        """
        idx = 0
        mask = None
        while idx < max_attempts:
            # sample mask
            mask = self.sample_mask()
            mask.to(missing_mask.device)

            # make sure that the sampled mask is valid
            mask_vis = missing_mask * land_mask * mask
            mask_hid = missing_mask * land_mask * (1 - mask)
            if torch.sum(mask_vis) > 0 and torch.sum(mask_hid) > 0:
                break

            idx += 1

        assert (
            mask is not None
        ), f"Failed to sample a valid mask in max_attempts:{max_attempts}"
        return mask.to(self._device)


def get_dataloaders(
    train_ratio=0.90,
    val_ratio=0.05,
    batch_size=8,
    shuffle=True,
    time_win=3,
    auxiliary_feat=True,
    cloud_coverage_threshold=1.0,
    n_samples=None,
    data_path="./data/SST_L3_CMEMS_2006-2021_Mediterranean.nc",
):
    """
    Get dataloaders
    :param train_ratio: the ratio of samples used for training
    :param val_ratio: the ratio of samples used for validation
    :param batch_size: the number of samples in a single batch
    :param shuffle: whether to shuffle the **training set** or not
    :param time_win: number of time steps
    :param auxiliary_feat: whether to utilize auxiliary features or not
    :param cloud_coverage_threshold: threshold used to filter observation fields with cloud coverage above it
    :param data_path: path to the dataset
    :return dataloaders: dictionary with keys: ["train", "val", "test"]
    :return metadata: metadata dictionary
    """

    # construct the dataset
    data, feat_to_idx, idx_to_feat, land_mask = _construct_dataset(
        data_path=data_path,
        cloud_coverage_threshold=cloud_coverage_threshold,
        n_samples=n_samples,
        time_win=time_win,
        auxiliary_feat=auxiliary_feat,
    )

    # compute the number of training and validation samples
    n_samples = data["observation"].shape[0]
    n_train = round(train_ratio * n_samples)
    n_val = round(val_ratio * n_samples)

    # split into train, validation and test sets
    train_dataset = SST_Dataset(
        {
            "observation": data["observation"][:n_train, :, :, :, :],
            "missing_mask": data["missing_mask"][:n_train, :, :],
        }
    )

    val_dataset = SST_Dataset(
        {
            "observation": data["observation"][n_train : n_train + n_val, :, :, :, :],
            "missing_mask": data["missing_mask"][n_train : n_train + n_val, :, :],
        }
    )

    test_dataset = SST_Dataset(
        {
            "observation": data["observation"][n_train + n_val :, :, :, :, :],
            "missing_mask": data["missing_mask"][n_train + n_val :, :, :],
        }
    )

    # construct data loaders
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle),
        "val": DataLoader(val_dataset, batch_size=batch_size),
        "test": DataLoader(test_dataset, batch_size=batch_size),
    }

    # gather metadata
    metadata = {
        "land_mask": land_mask,
        "feat_to_idx": feat_to_idx,
        "idx_to_feat": idx_to_feat,
        "time_win": time_win,
    }

    return dataloaders, metadata


def get_observation_labels(time_win=3):
    labels = []
    for dt in range(-(time_win // 2), time_win // 2 + 1):
        if dt == 0:
            labels += ["$\mathbf{x}_t \odot \mathbf{M}_m$"]
        else:
            dstring = f"+{dt}" if dt > 0 else str(dt)
            labels += ["$\mathbf{x}_{t" + dstring + "}$"]
    return labels


def _construct_dataset(
    data_path, cloud_coverage_threshold, n_samples, time_win=3, auxiliary_feat=False
):
    """Construct the entire dataset"""
    assert time_win % 2 == 1, "time_win must be an odd number!"

    # init. the feature to index mapping
    feat_to_idx = (
        {"measurement": 0, "doy_cos": 1, "doy_sin": 2}
        if auxiliary_feat
        else {"measurement": 0}
    )

    # construct the individual frames
    raw_data = get_raw_sst_data(data_path, n_samples)
    T, H, W = raw_data.sst.shape
    C = len(feat_to_idx)

    n_skipped = 0
    data = {"observation": [], "missing_mask": []}
    for t in range(T):
        x = np.zeros((time_win, C, H, W))

        # construct a training sample by taking into account time_win frames around t
        for dt in range(-(time_win // 2), time_win // 2 + 1):
            # clamp the time t between [0, T-1]
            t_clamped = np.clip(t + dt, 0, T - 1)

            # construct the current frame
            x_ = np.zeros((C, H, W))
            x_[feat_to_idx["measurement"]] = raw_data.sst[t_clamped].filled(0)

            if auxiliary_feat:
                x_[feat_to_idx["doy_cos"], :, :] = raw_data.doy_cos[t_clamped]
                x_[feat_to_idx["doy_sin"], :, :] = raw_data.doy_sin[t_clamped]

            x[time_win // 2 + dt, :, :, :] = x_

        # filter: skip samples with the fraction of missing values above the threshold
        missing_mask = 1 - raw_data.sst.mask[t] * raw_data.land_mask
        if cloud_coverage(missing_mask, raw_data.land_mask) >= cloud_coverage_threshold:
            n_skipped += 1
            continue

        data["observation"] += [torch.tensor(x, dtype=torch.float32)]
        data["missing_mask"] += [torch.tensor(missing_mask, dtype=torch.float32)]

    # convert to torch tensors
    land_mask = torch.tensor(raw_data.land_mask, dtype=torch.float32)
    data["observation"] = torch.stack(data["observation"], dim=0)
    data["missing_mask"] = torch.stack(data["missing_mask"], dim=0)
    print(f"n_skipped: {n_skipped}", flush=True)
    print(f"n_samples: {data['observation'].shape[0]}", flush=True)
    idx_to_feat = {v: k for k, v in feat_to_idx.items()}
    return data, feat_to_idx, idx_to_feat, land_mask


def cloud_coverage(missing_mask, land_mask):
    return np.sum(1 - missing_mask) / np.sum(land_mask)
