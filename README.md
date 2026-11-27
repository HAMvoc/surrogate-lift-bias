# surrogate-lift-bias

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24593213.svg)](https://doi.org/10.5281/zenodo.24593213)

Code to reproduce the results of the paper

> **Diagnosing and Correcting the Lift Bias of Neural Surface-Pressure Surrogates on an Unseen Vehicle Body Type**
> Quang Van Huy Nguyen, Huyen Ngoc Truong, Minh Anh Hoang, Khuong Nguyen-Vinh (submitted to *Computers & Fluids*).

Surface-pressure surrogates are trained on two DrivAerNet++ body types and tested on the third. The predicted pressure is integrated into
lift and drag coefficients, the bias of these coefficients is decomposed into a class-mean gap and a model overshoot, and it is corrected
with a few simulations of the new body type.

## Repository layout

```
configs/                      data, training, model, and surface-region settings
src/                          library code: data (grids, SDF, cache loader), models (voxel network M2,
                              baselines B0-B3, Transolver), training (trainer), evaluation (metrics, regions)
scripts/                      reproduction scripts grouped by paper section (index: scripts/README.md)
  data/                       download, subset, preprocessing, splits, label checks            (Section 4.1)
  training/                   training, inference, evaluation, cost                            (Section 4.1)
  diagnosis/                  bias = class-mean gap + overshoot, regions, cumulative lift,
                              centerline, axles, independent test set                          (Section 4.2)
  interventions/              ablations, inference-time interventions, descriptors, panel      (Section 4.3)
  correction/                 reference correction, two-stage method, stage B, intervals,
                              label budget                                                     (Section 4.4)
  baselines/                  response surfaces, Gaussian process, POD                         (Section 4.5)
data/manifests/, data/splits/ the 750-car subset, splits with sha256, label draws, folds, independent test cars
kaggle/, modal_run/           runners used for the GPU jobs (Kaggle, Modal)
REPRODUCE.md                  step-by-step commands, inputs, and environment for reproducing every section
```

Splits: `E1` is in-distribution; `E2A` holds out the estateback type (training on fastback and notchback), `E2B` the notchback, `E2C` the fastback.

## Setup

```bash
python -m pip install -r requirements.txt
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128   # CUDA build used for the paper
python scripts/data/setup_dirs.py                                                       # create local output folders
```

Scripts are run from the repository root (`python scripts/<folder>/<script>.py`); they import shared code as `src.*` and `scripts.*`.

## Data

DrivAerNet++ is not redistributed. The scripts download the STL meshes, surface pressure (VTK), and drag tables of the 750 cars in
`data/manifests/manifest.csv` and build a local cache:

```bash
python scripts/data/download_subset.py --manifest data/manifests/manifest.csv   # about 84 GB
python scripts/data/preprocess.py --workers 4                                    # SDF grids, occupancy, 100k-point pool -> data/cache/<id>/
python scripts/data/make_splits.py                                               # seeded; rewrites data/manifests/splits/ (sha256 in SHA256SUMS.json)
```

## Reproducing the results

Run the pipeline from scratch by following the stages in `REPRODUCE.md`:

1. **Data & preprocessing (`scripts/data/`):** download the subset, build SDF/occupancy grids and the 100k-point pool cache.
2. **Training & inference (`scripts/training/`, `kaggle/`, `modal_run/`):** train M2, B0-B3, ablations, and Transolver on single T4 GPUs (Kaggle / Modal) or locally, generating per-car metrics and predictions in `results/`.
3. **Diagnosis, interventions, correction, and baselines (`scripts/diagnosis/`, `scripts/interventions/`, `scripts/correction/`, `scripts/baselines/`):** run on CPU from the saved model predictions and local cache.

## License

Code: MIT (`LICENSE`). The car lists and splits are derived from DrivAerNet++ and shared under its CC BY-NC 4.0 license;
see `DATA_LICENSE.md`.

## Citation

```bibtex
@misc{nguyen2026code,
  author    = {Nguyen, Quang Van Huy and Truong, Huyen Ngoc and Hoang, Minh Anh and Nguyen-Vinh, Khuong},
  title     = {Code and data splits for ``Diagnosing and Correcting the Lift Bias of Neural Surface-Pressure Surrogates on an Unseen Vehicle Body Type''},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.24593213}
}
```

Please also cite the paper (reference to follow on publication) and DrivAerNet++ (Elrefaie et al., NeurIPS 2024 Datasets and Benchmarks).
