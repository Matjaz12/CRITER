import torch
import torch.nn as nn
from modules.crm import CoarseReconstructionModule
from modules.irm import IterativeRefinementModule
from typing import Tuple, List


class CRITER(nn.Module):
    """CRITER (Coarse Reconstruction with ITerative Refinement network)"""

    CHECKPOINT_NAME = "CRITER"

    def __init__(
        self,
        crm: CoarseReconstructionModule,
        irm: IterativeRefinementModule,
    ):
        super(CRITER, self).__init__()
        self.crm = crm
        self.irm = irm

        # freeze the CRM
        for param in self.crm.parameters():
            param.requires_grad = False
        self.crm.eval()

    def forward(
        self,
        observation: torch.Tensor,
        missing_mask: torch.Tensor,
        land_mask: torch.Tensor,
        sampled_mask: torch.Tensor,
        metadata: dict,
        inference: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate reconstruction and estimate the corresponding variance.
        :param observation: observation tensor, of shape (batch, time_win, channels, height, width)
        :param missing_mask: mask with 0's at locations where measurements are undefined, of shape (batch, height, width)
        :param land_mask: mask with 0's at locations corresponding to land areas, of shape (height, width)
        :param sampled_mask: sampled missing mask, of shape (height, width)
        :param metadata: metadata dictionary
        :param inference: boolean flag controlling whether a full reconstruction is generated or not
        """

        # compute the coarse reconstruction
        with torch.no_grad():
            mask = (
                sampled_mask * missing_mask.squeeze(dim=0)
                if inference
                else sampled_mask
            )
            rec, toks = self.crm(observation, mask, metadata)
            rec = rec * land_mask

        # compute the refined reconstruction and corresponding variance
        rec, var = self.irm(rec, toks, observation, land_mask, sampled_mask, metadata)
        return (rec, var)


def load_model(
    img_size: Tuple[int, int],
    channels: int,
    time_win: int,
    s_patch_size: int,
    t_patch_size: int,
    model_path: str = None,
    num_iterations: int = 1,
    rm_layers: List[int] = [32, 64, 128],
    extraction_layer: int = None,
) -> CRITER:
    """
    :param img_size: (height, width) of the image
    :param channels: number of channels / number of features
    :param time_win: number of time steps
    :param s_patch_size: height and width of the patch
    :param t_patch_size: temporal extent of the patch
    :param model_path: path to the pre-trained CRITER model
    :parma num_iterations: number of refinement steps
    :param rm_layers: refinement module layers
    :param extraction_layer: layer at which decoder tokens are extracted
    """
    # initialize the Coarse Reconstruction Module
    crm = CoarseReconstructionModule(
        img_size,
        channels,
        time_win,
        s_patch_size,
        t_patch_size,
        extraction_layer=extraction_layer,
    )

    # initialize the Iterative Refinement Module
    irm = IterativeRefinementModule(
        time_win,
        2,
        rm_layers,
        num_iterations,
        crm.decoder.embed_dim,
        crm.num_patches_h,
        crm.num_patches_w,
        crm.num_patches_t,
    )

    # initialize the model
    model = CRITER(crm, irm)

    # load the parameters
    if model_path:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        if "loss" in checkpoint.keys():
            print(f"loaded model (loss): {checkpoint['loss']}")
    return model
