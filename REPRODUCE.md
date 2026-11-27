# Reproducing the paper

Every command is run from the repository root after creating local output folders with `python scripts/data/setup_dirs.py`. Outputs are written locally to `results/<analysis>/` and `reports/`. Scripts are listed in the order of the paper; `scripts/README.md` describes every script.

**Environments.** *Laptop*: Windows 11, Python 3.10.11, torch 2.10.0+cu128 (preprocessing and CPU analyses). *Kaggle*: 2 x T4, Python 3.13, torch 2.11.0+cu130 (training of M2 and B0-B3 on E1/E2A/E2B/E2C, ablations, Transolver). *Modal*: T4, Python 3.11, torch 2.10.0 (fine-tuning and retraining of the corrections, the `loc` run, the environment check). GPU runs are not bit-reproducible (atomic additions in the backward pass of trilinear grid sampling, half precision); re-runs differ within the seed spread.

**Consistency check used throughout:** the voxel network M2 on E2A has a three-seed lift bias of +183.9 counts.

## Section 4.1: data, splits, models

| Step | Command | Writes |
|---|---|---|
| download, subset, preprocessing | `scripts/data/download_subset.py`, `scripts/data/make_subset.py`, `scripts/data/preprocess.py`, `scripts/data/local_features.py` | `data/cache/<id>/` |
| splits, label draws, folds | `scripts/data/make_splits.py`; draws and folds in `data/splits/` | `data/manifests/splits/`, `data/splits/` |
| pool estimator of the coefficients | `scripts/data/lift_pool_validation.py`, `scripts/data/pool_weight_impact.py` | `reports/` |
| training (Kaggle) | `scripts/training/train.py --split E2A --model m2 --res R2 --seed 0 --eval-test` (one run; runners in `kaggle/`) | `runs/<run_id>/`, `results/<run_id>/` |
| baselines B0, B1 | `scripts/training/run_baselines.py` | `results/<split>_B0_*`, `results/<split>_B1_*` |
| Transolver checks | `scripts/training/check_transolver_code.py`, `scripts/training/check_transolver_run.py` | `results/transolver/` |
| cost (Table S8) | `scripts/training/cost_table.py` | `reports/cost_table.md` |

## Section 4.2: anatomy of the bias

| Paper item | Script | Main inputs |
|---|---|---|
| integrated coefficients and their error per car | `scripts/diagnosis/force_alignment_terms.py` | `results/<run_id>/`, per-point predictions |
| bias = class-mean gap + overshoot (Fig. 3, Fig. S2) | `scripts/diagnosis/gap_and_overshoot.py` | `results/force_alignment/` |
| surface regions with the wheel fit | `scripts/diagnosis/regions.py` | `data/cache/` -> `results/decomposition/T1/labels_v2.npz` |
| regional bias and variance (Fig. 4) | `scripts/diagnosis/regional_bias.py` | `results/decomposition/T1/` |
| error maps (Fig. 5), scatter (Fig. S1) | `scripts/diagnosis/diagnostic_figures.py` | per-point predictions, `data/cache/` |
| class-mean gap by region (Table 2) | `scripts/diagnosis/regional_gap.py` (per-car inputs: `scripts/data/per_car_coefficients.py`) | `results/classical_baselines/` |
| cumulative lift (Fig. 6) | `scripts/diagnosis/cumulative_lift.py` | `results/margin_and_cumulative_lift/` |
| centerline pressure, axle lift, Reynolds number, wind tunnel (Fig. 7) | `scripts/diagnosis/centerline_and_axles.py` | `results/centerline_and_axles/` |
| Transolver bias on E2A, E2B, E2C | `scripts/diagnosis/transolver_bias.py`, `scripts/diagnosis/transolver_other_types.py` | `results/transolver/`, `results/transolver_types/` |
| independent test set (250 unused cars) | `scripts/data/select_independent_cars.py`, `scripts/data/prepare_independent_cars.py`, `scripts/training/infer_cars.py`, `scripts/diagnosis/independent_set_test.py` | `results/independent_set/` |
| shift indicators, axle split on E2B/E2C (Tables S5, S7, Fig. S6) | `scripts/diagnosis/shift_indicators_and_axles.py` | `results/shift_and_axles/` |

## Section 4.3: interventions that do not remove the bias

| Paper item | Script | Main inputs |
|---|---|---|
| training ablations (Fig. 8) | `scripts/interventions/training_ablations.py`, `scripts/interventions/ablation_seeds.py` | `results/<run_id>/` of the ablation runs |
| inference-time interventions (Fig. 8) | `scripts/interventions/inference_interventions.py` | saved checkpoints |
| local surface descriptors | `scripts/interventions/local_descriptors.py` (training run on Modal: `modal_run/train_modal.py`) | `results/decomposition/T6/` |
| integral penalty, data scaling | `scripts/interventions/integral_loss_and_long_training.py`, `scripts/interventions/data_scaling.py` | `results/<run_id>/` |
| potential-flow prior | `scripts/interventions/panel_method.py sphere\|cars\|explore` | `results/decomposition/T8/` |

## Section 4.4: few-shot correction

| Paper item | Script | Main inputs |
|---|---|---|
| shared statistics and corrections | `scripts/correction/calibration.py` (library), `scripts/correction/statistics_cache.py` | per-point predictions |
| frozen models with the reference correction and stage B, 50 draws | `scripts/correction/frozen_draws.py` | `results/two_stage/T2/` |
| two-stage method with out-of-fold stage B (Modal) | `modal run -d modal_run/correction_jobs.py::main --what t1`; `scripts/correction/two_stage_oof.py`; analysis `scripts/correction/two_stage_analysis.py` | `results/two_stage/T1/` |
| retraining with stage B (Modal) | `modal run -d modal_run/correction_jobs.py::main --what retrain --ks 10,25 --draws 0-4` and `--what oof`; `scripts/correction/retrain_oof.py` | `results/components/T3/` |
| stage B across models, axle-wise correction (Table 4) | `scripts/correction/cross_model.py`, `scripts/correction/transolver_stage_b.py`; axles: `scripts/diagnosis/shift_indicators_and_axles.py` | `results/components/T2/`, `results/transolver/T4/`, `results/shift_and_axles/` |
| reference correction against stage B | `scripts/correction/reference_vs_stage_b.py` | `results/reference_correction/` |
| swapped halves, independent set | `scripts/correction/swapped_halves.py`, `scripts/correction/swapped_halves_two_stage.py`; `scripts/diagnosis/independent_set_test.py` | `results/swapped_halves/`, `results/independent_set/` |
| component effects (Table S1, Fig. S3) | `scripts/correction/components.py` | `results/two_stage/T1/raw/` |
| out-of-fold against in-sample stage B (Table S2) | `scripts/correction/oof_vs_in_sample_finetune.py` (GPU), `scripts/correction/oof_vs_in_sample.py` | `results/label_budget/T3/` |
| prediction intervals | `scripts/correction/interval_comparison.py`, `scripts/correction/prediction_powered_inference.py` | `results/intervals/`, `results/ppi/` |
| label-budget rule (Fig. 9, Fig. S5, Table S4) | `scripts/correction/label_budget_test.py`, `scripts/correction/budget_partitions.py`, `scripts/correction/label_budget_bootstrap.py` | `results/label_budget/`, `results/budget_partitions/` |
| sequential procedure with a safety margin (Table S3) | `scripts/correction/sequential_margin.py` | `results/margin_and_cumulative_lift/` |
| shrinkage of the offsets | `scripts/correction/shrinkage.py` | `results/label_budget/T2/` |

## Section 4.5: classical baselines and cost

| Paper item | Script | Main inputs |
|---|---|---|
| response surfaces and Gaussian process (Table 5) | `scripts/baselines/parametric_surrogates.py` | `results/classical_baselines/` |
| POD reconstruction (Table 5) | `scripts/baselines/pod_reconstruction.py`, `scripts/baselines/pod_transfer_error.py` | `results/pod/` |
| cost (Table S8) | `scripts/training/cost_table.py` | `reports/cost_table.md` |
