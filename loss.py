import torch


def _masked_mean_squared_error(target, rec, mask):
    """Compute mean squared error (MSE) wrt to the binary mask"""
    se = _masked_squared_error(target, rec, mask)
    return torch.sum(se) / torch.count_nonzero(mask)


def _masked_squared_error(target, rec, mask):
    """Compute squared error (SE) wrt to the binary mask"""
    return ((target - rec) * mask).square()


def mean_squared_error(target, rec, missing_mask, land_mask, sampled_mask):
    """
    Compute mean squared error over the following three regions:
    all pixels, visible pixels, masked pixels
    """
    loss_dict = {}

    # compute mse over all pixels
    M_all = missing_mask * land_mask
    loss_dict["mse_all"] = _masked_mean_squared_error(target, rec, M_all)
    loss_dict["M_all"] = M_all

    # compute mse over visible pixels
    M_vis = M_all * sampled_mask
    loss_dict["mse_vis"] = _masked_mean_squared_error(target, rec, M_vis)
    loss_dict["M_vis"] = M_vis

    # compute mse over hidden pixels
    M_hid = M_all * (1 - sampled_mask)
    loss_dict["mse_hid"] = _masked_mean_squared_error(target, rec, M_hid)
    loss_dict["M_hid"] = M_hid

    # add the loss key
    loss_dict["loss_all"] = loss_dict["mse_all"]
    return loss_dict


def neg_log_likelihood_gaussian(target, rec, var, missing_mask, land_mask):
    """
    Compute the negative log likelihood of a gaussian as specified in the DINCAE2 paper.
    (See: https://gmd.copernicus.org/articles/15/2183/2022/gmd-15-2183-2022.pdf)
    """
    M_all = missing_mask * land_mask
    y_hat, y = target * M_all, rec * M_all

    log_loss = torch.log(var) * M_all
    var_scaled_se = ((y_hat - y).square() / var) * M_all
    loss = (var_scaled_se + log_loss).sum() / torch.count_nonzero(M_all)
    return {"loss_all": loss, "M_all": M_all}
