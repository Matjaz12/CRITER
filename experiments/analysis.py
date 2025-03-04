import math
import json
import torch
import numpy as np
from typing import List
from welford import Welford
from streamhist import StreamHist


class CloudCoverageLevelRobustnessAnalysis:
    """Analyse the robustness of the model to clouds with varying coverage levels"""
    
    min_cloud_coverage = float("inf")
    max_cloud_coverage = float("-inf")

    def __init__(
        self, metrics: List[str], group_max_coverage: List[float] = [0.6, 0.75]
    ) -> None:
        self._data = {m: {} for m in metrics}
        self.groups = {}

        # add groups
        group_max_coverage = [0.0] + group_max_coverage + [1.0]
        for idx in range(len(group_max_coverage) - 1):
            # compute group bounds
            lower_bound, upper_bound = (
                group_max_coverage[idx],
                group_max_coverage[idx + 1],
            )
            self.groups[f"g_{idx}"] = {
                "lower_bound": lower_bound,
                "upper_bound": upper_bound,
            }

            # add group to all metrics
            for m in metrics:
                self._data[m][f"g_{idx}"] = {
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "vals": [],
                }

        print(f"groups: {self.groups}", flush=True)

    def update(self, data, sampled_mask, missing_mask, land_mask):
        # find the group corresponding to cloud coverage
        coverage = self._compute_cloud_coverage(sampled_mask, missing_mask, land_mask)
        group = self._find_group(coverage)
        assert (
            group != None
        ), f"Coverage:{coverage} out of range! Could not find the appropriate group!"

        # store data in the corresponding group
        for key, val in data.items():
            assert key in data.keys(), f"Unknown metric:{key}"
            self._data[key][group]["vals"] += [val]

        # update cloud coverage
        self.min_cloud_coverage = min(self.min_cloud_coverage, coverage)
        self.max_cloud_coverage = max(self.max_cloud_coverage, coverage)

    def export(self, filename: str) -> None:
        _data = self._data.copy()
        for metric in _data.keys():
            for group in _data[metric].keys():
                _data[metric][group] = {
                    "lower_bound": self._data[metric][group]["lower_bound"],
                    "upper_bound": self._data[metric][group]["upper_bound"],
                    "mean": np.mean(self._data[metric][group]["vals"]),
                    "std": np.std(self._data[metric][group]["vals"]),
                    "n_samples": len(self._data[metric][group]["vals"]),
                }

        # store the minimum and maximum (observed) cloud coverage
        _data["min_cloud_coverage"] = self.min_cloud_coverage
        _data["max_cloud_coverage"] = self.max_cloud_coverage

        with open(filename, "w") as file:
            json.dump(_data, file, indent=4)
        print(f"Results saved to:{filename}", flush=True)

    def _find_group(self, coverage) -> str:
        _group = None
        for group, val in self.groups.items():
            if coverage > val["lower_bound"] and coverage <= val["upper_bound"]:
                _group = group
                break
        return _group

    def _compute_cloud_coverage(self, sampled_mask, missing_mask, land_mask):
        mask = missing_mask * sampled_mask
        assert (
            torch.unique(mask[:, land_mask == 0]) == 1
        ), "Missing mask defined on land!"
        area_cloud = torch.sum(1 - mask)
        area_sea = torch.sum(land_mask)
        return (area_cloud / area_sea).item()


class VarianceBiasAnalysis:
    """Analyse the statistical properties of the estimated variance and reconstruction bias"""

    def __init__(self, maxbins=100):
        # initialize the streaming histograms
        # (See: https://github.com/carsonfarmer/streamhist)
        self.maxbins = maxbins
        self.hist_del = StreamHist()
        self.hist_vis = StreamHist()

        # Use the Welford's algorithm to estimate accurate statistics over a stream of data
        # (See: https://pypi.org/project/welford/)
        self.w_del = Welford()
        self.w_vis = Welford()

    def update(self, target, rec, var, land_mask, missing_mask, sampled_mask):
        # compute the scaled difference (eps) and unscaled difference (bias)  
        eps = (target - rec) / torch.sqrt(var)
        bias = target - rec

        # compute deleted and visible masks
        M_all = missing_mask * land_mask
        M_del = M_all * (1 - sampled_mask)
        M_vis = M_all * sampled_mask

        # separate into deleted and visible regions
        eps_del = eps[M_del == 1].flatten().tolist()
        eps_vis = eps[M_vis == 1].flatten().tolist()
        assert (
            len(eps_del) + len(eps_vis) == eps[M_all == 1].numel()
        ), f"difference vector was split incorrectly! {len(eps_del) + len(eps_vis)} != {eps[M_all == 1].numel()}"

        bias_del = bias[M_del == 1].flatten().tolist()
        bias_vis = bias[M_vis == 1].flatten().tolist()
        assert len(bias_del) == len(eps_del) and len(bias_vis) == len(eps_vis)
        
        # update the streaming histogram
        self.hist_del.update(eps_del)
        self.hist_vis.update(eps_vis)

        # update mu_eps, sigma_eps and the bias 
        for (eps_i, bias_i) in zip(eps_del, bias_del):
            self.w_del.add(np.array([eps_i, bias_i]))

        for (eps_i, bias_i) in zip(eps_vis, bias_vis):
            self.w_vis.add(np.array([eps_i, bias_i]))

    def export(self, filename) -> str:
        # store mu_eps, sigma_eps and the bias
        def get_means_counts_widths(hist):
            # compute counts and bin edges
            counts, bins = hist.compute_breaks(self.maxbins)

            # estimate the means, i.e., the center of each bin and the width of each bin
            # the width should be a constant!
            means = [(a + b)/2. for a, b in zip(bins[:-1], bins[1:])]
            widths = [a - b for a, b in zip(bins[1:], bins[:-1])]

            # return in dict format
            return {"means": means, "counts": counts, "widths": widths}

        hist_del = get_means_counts_widths(self.hist_del)
        hist_del["mu_eps"] = self.w_del.mean[0]
        hist_del["sigma_eps"] = math.sqrt(self.w_del.var_p[0])
        hist_del["bias"] = self.w_del.mean[1]

        # store mu_eps, sigma_eps and the bias
        hist_vis = get_means_counts_widths(self.hist_vis)
        hist_vis["mu_eps"] = self.w_vis.mean[0]
        hist_vis["sigma_eps"] = math.sqrt(self.w_vis.var_p[0])
        hist_vis["bias"] = self.w_vis.mean[1]   
     
        # save data
        _data = {"deleted_regions": hist_del, "visible_regions": hist_vis}
        with open(filename, "w") as file:
            json.dump(_data, file, indent=4)
        print(f"Results saved to:{filename}", flush=True)
