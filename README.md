# CRITER — A coarse reconstruction with iterative refinement network for sparse spatio-temporal satellite data

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13923156.svg)](https://doi.org/10.5281/zenodo.13923156)

This repository contains the official PyTorch implementation of the paper [CRITER 1.0: a coarse reconstruction with iterative refinement network for sparse spatio-temporal satellite data](https://gmd.copernicus.org/articles/18/5549/2025/gmd-18-5549-2025.html):
```
@article{ZupancicMuc2025,
  author       = {Zupančič Muc, M. and Zavrtanik, V. and Barth, A. and Alvera-Azcarate, A. and Ličer, M. and Kristan, M.},
  title        = {{CRITER 1.0: a coarse reconstruction with iterative refinement network for sparse spatio-temporal satellite data}},
  journal      = {Geoscientific Model Development},
  volume       = {18},
  pages        = {5549--5573},
  year         = {2025},
  doi          = {10.5194/gmd-18-5549-2025},
  url          = {https://doi.org/10.5194/gmd-18-5549-2025}
}
```

## Abstract
Satellite observations of sea surface temperature (SST) are essential for accurate weather forecasting and climate modeling. However, this data often suffers from incomplete coverage due to cloud obstruction and limited satellite swath width, which requires development of dense reconstruction algorithms. The current state-of-the-art struggles to accurately recover high-frequency variability, particularly in SST gradients in ocean fronts, eddies, and filaments, which are crucial for downstream processing and predictive tasks. To address this challenge, we propose a novel two-stage method CRITER (Coarse Reconstruction with ITerative Refinement Network), which consists of two stages. First, it reconstructs low-frequency SST components utilizing a Vision Transformer-based model, leveraging global spatio-temporal correlations in the available observations. Second, a UNet type of network iteratively refines the estimate by recovering high-frequency details. Extensive analysis on datasets from the Mediterranean, Adriatic, and Atlantic seas demonstrates CRITER's superior performance over the current state-of-the-art. Specifically, CRITER achieves up to 44 % lower reconstruction errors of the missing values and over 80 % lower reconstruction errors of the observed values compared to the state-of-the-art.

![CRITER](CRITER.jpg)


## Pretrained Models

We offer pretrained checkpoints for the CRITER model, trained on various datasets. You can download them using the links below:

| Dataset        | Pretrained Checkpoint |
|----------------|----------------------|
| Mediterranean  | [download](https://drive.google.com/file/d/13ll0Sr5NR1qUtsuZu6C4B-NxjvPfmJ1u/view?usp=drive_link) |
| Adriatic       | [download](https://drive.google.com/file/d/1whCB9QL876SjW4afnXI-G0Q7DHOFrsVI/view?usp=drive_link) |
| Atlantic       | [download](https://drive.google.com/file/d/1qyYqte3QkOXwEdS-R1qYqFob3d4L7_Ki/view?usp=drive_link) |

## Datasets

The datasets used for training and testing the CRITER model are available for download:

| Dataset        | Download Link |
|----------------|---------------|
| Mediterranean  | [download](https://drive.google.com/file/d/1f35PqectvdRN4UsKrWSPb9vAIVZUMGhb/view?usp=drive_link) |
| Adriatic       | [download](https://drive.google.com/file/d/1iMk0lHHVhO43R6PJDtSk5cz7Ys0ej5Yv/view?usp=drive_link) |
| Atlantic       | [download](https://drive.google.com/file/d/1qyYqte3QkOXwEdS-R1qYqFob3d4L7_Ki/view?usp=drive_link) |
