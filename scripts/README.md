# Scripts

Every script is run from the repository root, e.g. `python scripts/correction/frozen_draws.py`. Each folder is a Python package,
so shared helpers are imported as `from scripts.<folder>.<module> import ...`. `REPRODUCE.md` gives the order of the commands and
the script behind every analysis of the paper; outputs are written locally to `results/<analysis>/` and `reports/`.

## `scripts/data/`: Data (Section 4.1)

| Script | Purpose |
|---|---|
| `audit_dataset.py` | Audit the downloaded cars (spec 4.6) -> reports/audit.md. |
| `car_geometry.py` | Per-car reference areas for the pressure-drag diagnostic (spec 10.5): total surface area of the CFD surface mesh (the mesh the Cp labels live on) and projected frontal area A_ref. |
| `check_cp_scale.py` | Check the Cp scale (E0 no. 6): stagnation point at the car nose, and C_D,p integrated from pressure vs total Cd. |
| `check_lift_labels.py` | Cheap checks of the lift labels. |
| `download_subset.py` | Selectively download DrivAerNet++ from the HF mirror: only STL + pressure VTK + the Cd file. |
| `inspect_probe.py` | Quick audit of the probe-downloaded cars: pressure variables, bbox, watertight mesh, p range. |
| `lift_pool_validation.py` | Pool (IPW) estimator vs the exact full-mesh integral for C_D,p AND C_L,p on 45 cars (15 per body style, evenly spaced), CFD labels only (exploratory; closes the lift part of the 8-car check in cdp_analysis). |
| `list_hf_mirror.py` | Counts of the files of the Hugging Face mirror (names and sizes only, no file is downloaded). |
| `local_features.py` | Compute src/data/local_features.compute for cars and write data/cache/<id>/localfeat.npy (float16, (100000, 6), pool order). |
| `make_splits.py` | Generate splits (spec 6): E1, E2A/B/C, E2b. |
| `make_subset.py` | Pick a seeded random subset: 250 cars per body style + 50 reserve, only from cars with a parameter table entry (excludes F_D_*). |
| `per_car_coefficients.py` | Per-car CFD C_L,p and C_D,p (counts), and the contribution of each v2 region to C_L,p, for the 750 cars of the subset (inputs of W1 and W2). |
| `pool_weight_impact.py` | Impact of the non-exact area-proportionality of the 100k point pool (found while building scripts/diagnosis/pressure_coefficients.py). |
| `prepare_independent_cars.py` | Pressure files, preprocessing and per-car geometry of the selected new Estateback cars (data/splits/fresh_estateback.csv). 1) download the VTK (pressure) of each selected car (the STL was downloaded by select_independent_cars.py);... |
| `preprocess.py` | Preprocessing (spec 5): each car -> data/cache/<id>/{sdf_R1,sdf_R2,sdf_R3,occ_R2}.npy, pool.npz, meta.json. |
| `sdf_probe.py` | Probe SDF R1-R3 on a few cars: timing, sign, level 0 at surface points, and how far to trust the flood fill versus the exact signed SDF. |
| `select_independent_cars.py` | Selection of 250 new Estateback cars. |
| `setup_dirs.py` | Create the output folders that the analysis scripts write into (results/, reports/, paper_assets/). |

## `scripts/training/`: Training and inference (Section 4.1)

| Script | Purpose |
|---|---|
| `check_transolver_code.py` | Check that src/models/transolver.py equals the CarBench Transolver (same state dict after renaming, same output). |
| `check_transolver_run.py` | Checks of a trained Transolver run (best.pt of runs/<run_id>) on a few cars of a split. |
| `cost_table.py` | Model cost table (spec 7.5, T2/F7): parameters, FLOPs, inference time and peak memory (measured here, on the local GPU), training step time / peak VRAM / session hours (read from the Kaggle run logs). |
| `environment_check.py` | M2 original on Modal (E2A, seed 0, 150 epochs) against the three Kaggle seeds. |
| `evaluate.py` | Result tables: E1, E2 (three folds), confirmatory M2 vs B1 comparison (Wilcoxon + Holm), D_X, S3/S4, sensitivity to dropping near-duplicate cars, raw labels. |
| `infer_cars.py` | Inference of the E2A models (as trained, no training) on a list of cars; writes <out>/<label>/{test_per_car.csv, test_pred_fp16.npy} in the layout of results/<run_id>/. |
| `predict_from_checkpoint.py` | Inference only (no training): per-point Cp predictions of a saved M2 run from its best.pt (EMA weights), computed like Trainer.evaluate (fp16 autocast as in the run's cfg, all 100k pool points per car in chunks, output * sd + mu, ... |
| `run_baselines.py` | Run B0 and B1 on the splits; writes results/<run_id>/{val,test}_per_car.csv (rel L2 on clipped Cp and on raw Cp). |
| `train.py` | Train one run (spec 8). |

## `scripts/diagnosis/`: Anatomy of the bias (Section 4.2)

| Script | Purpose |
|---|---|
| `centerline_and_axles.py` | Reynolds number (A), centreline pressure profiles (B) and front-/rear-axle lift (C). |
| `clip_sensitivity.py` | Sensitivity of rel L2 and of the degradation ratio D to the Cp clipping range of the target (exploratory; prereg promised this check). |
| `cumulative_lift.py` | Cumulative lift along the body, from the nose to the relative position x, for the training types and the estateback type, the class-mean gap curve with a bootstrap band, and the same curve for the M2 lift error. |
| `diagnostic_figures.py` | Diagnostic tables and figures (CPU, from existing results). |
| `force_alignment.py` | Force-alignment decomposition of the integral pressure-coefficient error (Force-alignment study of the C1 investigation). |
| `force_alignment_sensitivity.py` | Sensitivity of the M2 force-alignment numbers (rho_L, //e//, C_L,p bias) to the Cp clip range of the reference and to the IPW weights (naive w = area). |
| `force_alignment_terms.py` | Force-alignment study of the C1 investigation (force alignment of the Cp error), tasks T1-T5. |
| `gap_and_overshoot.py` | Bias = G + overshoot for every model family and fold. |
| `independent_set_geometry.py` | Geometry of the new Estateback cars against the 250 old test cars: the descriptors theta_rear (4) and theta_under (6) of the force-alignment and zero-shot checks studies (computed from the pools) and the eight design parameters of... |
| `independent_set_test.py` | Independent test on the new Estateback cars. |
| `integral_errors.py` | Per-car C_D,p and C_L,p (CFD labels and model predictions, counts) for every held-out-style model run with saved predictions -> reports/integral_errors.csv. |
| `physics_plots.py` | Physics-style diagnostics from the saved per-point test predictions (exploratory): (1) Cp along the symmetry plane (y ~ 0): upper and lower surface profiles of the six qualitative cars (CFD vs M2 vs B2); (2) pressure lift C_L,p = ... |
| `plot_regions.py` | Plot surface regions on the pool of 3 cars per body style (spec 10.4: check visually before use). |
| `pressure_coefficients.py` | Pressure-drag diagnostic (spec 10.5, exploratory): C_D,p = -(1/A_ref) * sum_i Cp_i A_i (n_i . e), e = +x, from the 100k-point pool. |
| `qualitative_figures.py` | Qualitative figures (spec 10.8): for each held-out body style (E2A/E2B/E2C) the car with median per-car error and the worst one. |
| `reference_levels.py` | Reference levels for the between-car RMSE of the lift. |
| `regdgcnn_checks.py` | Is the RegDGCNN (B3) lift bias on E1 and its sign change between E2A and E2C a pipeline artefact? |
| `region_errors.py` | Exploratory error analysis on the saved per-point test predictions (spec 10.4, E6): (1) error by surface region (area share, MAE, rel L2, share of total squared error, enrichment = error share / area share), M2 and B2; (2) error v... |
| `regional_bias.py` | Re-run the regional analyses with the v2 regions (wheels split off) and put old and new side by side. |
| `regional_gap.py` | Classical-baseline and regional-gap study / W2: regional class-mean gap G_r = mean_train(c_r) - mean_test(c_r) of the lift coefficient, from the per-car CFD contributions of results/classical_baselines/cars.csv. |
| `regions.py` | Regions v2 (Decomposition and first corrections study T1): the force-alignment study rules plus a `wheels` region found by fitting the tyre circle on each car's own pool (no part labels exist in the data). |
| `render_regions.py` | Render the T3 (Force-alignment study) regions on three cars (one per body style) from the side, the top and the bottom, for a visual check of the region rules. |
| `render_regions_with_wheels.py` | Render the v2 regions (with wheels) on the same three cars as the zero-shot checks study T5. |
| `seed_spread_check.py` | Seed-ensemble uncertainty gate (prereg addendum 1, B; exploratory): does the spread of the 3 M2 seeds flag where / for which cars the prediction is wrong? |
| `shift_indicators_and_axles.py` | Label-free shift indicators (A), axle-wise reference correction (B), axle split and centreline on E2B/E2C (C). |
| `transolver_bias.py` | Transolver study, task T2 and T3: capability gate on E1 and the Estateback bias of Transolver on E2A. |
| `transolver_other_types.py` | Transolver (seed 0) on E2B and E2C (E2B: /bias C_L,p/ <= 50 counts; E2C: description, label 'second error in the opposite direction' if /bias/ > 100). |
| `wheel_and_bias.py` | Exploratory (DB-4): (1) error by wheel configuration (WW open / WWS smooth, from the car id) within each body style; (2) signed region bias of every model on each held-out fold (pred - CFD, clipped Cp, mean over cars with bootstra... |
| `wheel_configuration.py` | Car configuration codes (wheel variant) and the between-car variation of the M2 lift bias on E2A. |
| `wheel_fit_variant.py` | Try to fix the wheel region (rear wheel outer face, tyre top). |

## `scripts/interventions/`: Interventions that do not remove the bias (Section 4.3)

| Script | Purpose |
|---|---|
| `ablation_seeds.py` | Prereg addendum 4 (exploratory): A) seeds 0-2 for the Occupancy, noXYZ and R3 ablations on E2A, B) 5-stage encoder at R3. |
| `data_scaling.py` | Data-scaling curve (exploratory, single seed): M2 R2 test rel L2 vs number of train cars on E1 and E2A (held out Estateback). |
| `inference_interventions.py` | Inference-time interventions on saved M2 R2 checkpoints (no training). |
| `integral_loss_and_long_training.py` | Prereg addendum 3 (exploratory): A) integral loss, B) active calibration selection, C) B2 300 epochs. |
| `local_descriptors.py` | `loc` zero-shot on E2A (3 seeds, run on Modal) against the registered thresholds (PREREG T6). |
| `panel_method.py` | Prototype of a constant-source panel method (non-lifting potential flow, triangular panels, mirror image through the ground plane z = 0, no wake). |
| `training_ablations.py` | Zero-shot checks study, task T1 extra (existing runs, no new training): C_L,p bias on E2A of the M2 training-level ablations (no decoder xyz, no normals, occupancy, grid R3) from their saved predictions, for comparison with the in... |
| `underbody_descriptors.py` | Underbody descriptors and extended oracle R2 (ridge, 10-fold CV by car, 10 repeats) for the M2 lift bias on E2A. |

## `scripts/correction/`: Few-shot correction (Section 4.4)

| Script | Purpose |
|---|---|
| `budget_partitions.py` | Partition-level test of the label-budget relation (Q1), replay of the sequential procedure (Q2, Q3) and consistency of the pool-integrated lift with the dataset tables (B). |
| `calibration.py` | Shared calibration library for the decomposition and first corrections study (T1, T5, T7 stage B): per-region sufficient statistics of a set of predictions, post-hoc corrections that subtract a constant per region from the predict... |
| `components.py` | Component and cross-model study, task T1 (EXPLORATORY): component effects of the two-stage C2 from the two-stage correction study per-draw results. |
| `conformal_first_look.py` | Few-shot conformal calibration of integral coefficients under body-style shift (prereg addendum 5, exploratory). |
| `cross_model.py` | Component and cross-model study, task T2 (EXPLORATORY): does the integral correction (frozen + region_const_intmatch, 8 regions v2, lambda by LOO inside the k cars) generalise over model families and shift directions? |
| `finetune_stage_b.py` | Balanced fine-tuning of the E2A M2 checkpoints on k Estateback cars + stage B (region_const, 8 regions) with cross-fitting, plus the pieces needed for CV+ (K = 5). |
| `finetune_stage_b_analysis.py` | Decomposition and first corrections study, task T7 analysis of the fine-tune runs (results of scripts/correction/finetune_stage_b.py, downloaded to results/decomposition/T7/raw/{dec,full}/). |
| `first_calibration.py` | C2 first round, calibration of the frozen M2 on E2A from k labelled Estateback cars (CPU only). |
| `frozen_draws.py` | Two-stage correction study, task T2 (frozen part) and T3 (frozen curve): frozen M2 (3 seeds) + none / scalar_shift / region_const / region_const_intmatch (lambda by LOO) on the shared draws. |
| `interval_comparison.py` | Coverage and half-width of the jackknife+ interval of the reference correction (stored, the reference-correction study) against an unconditional-spread interval (computed here), original halves, seven E2A models, k in {5, 10, 25},... |
| `label_budget_bootstrap.py` | Number-source study, task T3 (EXPLORATORY; does not change the registered 'partial' verdict of the label-budget study T1): cluster bootstrap of the slope and R^2 of the statistical-floor regression (obs = slope * pred, no intercep... |
| `label_budget_cells.py` | Transolver on all held-out types study, task T4 (EXPLORATORY, description only): observed over predicted /bias/ (statistical-floor formula, sigma_pop) against k for every cell of the floor analysis: the six original families and m... |
| `label_budget_test.py` | Test of the statistical-floor hypothesis E/bias/ ~ sqrt(2/pi) sigma / sqrt(k). |
| `oof_vs_in_sample.py` | Label-budget study, task T3 analysis: B-same versus B-OOF on the same draws. |
| `oof_vs_in_sample_finetune.py` | B-same versus B-OOF on the same draws. |
| `prediction_powered_inference.py` | Prediction-powered inference (PPI) interval for the mean C_L,p of the new body type against the classical interval of the k labelled cars. |
| `reference_vs_stage_b.py` | Stage B (frozen + region_const_intmatch, stored numbers) against the minimal baseline scalar_shift on the same cars and draws. |
| `retrain_calibration.py` | Few-shot calibration on E2A (prereg addendum 1, A; exploratory): M2 R2 trained on 450 Fastback+Notchback plus k Estateback cars, tested on the fixed 225 Estateback cars. k = 0 is the existing E2A_M2_R2_s0 run restricted to the sam... |
| `retrain_oof.py` | Retrain + stage B fitted on out-of-fold residuals, several draws. |
| `sequential_margin.py` | Sequential label-budget procedure with a safety margin; selection on partitions of the 250 original estateback cars, test on the 250 new cars. |
| `shrinkage.py` | Label-budget study, task T2 (EXPLORATORY, not a main method): positive-part James-Stein shrinkage of the stage-B offsets, b_shrunk = b * max(0, 1 - SE^2 / b^2), SE = sigma_k / sqrt(k). |
| `statistics_cache.py` | Label-budget study shared: sufficient statistics (calibration.build_stats) of the model predictions used by T1/T2, cached under results/label_budget/cache (git-ignored). pair_stats(fold, family): the 250 test cars of the fold in t... |
| `swapped_halves.py` | Stage B (frozen + region_const_intmatch, settings of the two-stage correction and component and cross-model studies unchanged) with the roles of the two halves of the 250 test cars swapped. |
| `swapped_halves_two_stage.py` | M2 `full` ft_B_OOF (seed 0, k = 10) on the swapped halves, 10 draws. |
| `transolver_stage_b.py` | Stage B (frozen + region_const_intmatch, 8 v2 regions, lambda by LOO) and the out-of-sample test of the floor law on Transolver (PREREG.md and addendum A). |
| `two_stage_analysis.py` | Two-stage correction study analysis of T1 (two-stage C2, OOF stage B), T2 (baselines) and T3 (label-budget curve). |
| `two_stage_oof.py` | Two-stage C2 with out-of-fold stage B. |
| `two_stage_retrain.py` | Two-stage C2 on the existing retrain checkpoints. |
| `warm_statistics_cache.py` | Build the label-budget study statistics caches once (T1 and T2 share them). |

## `scripts/baselines/`: Classical baselines and cost (Section 4.5)

| Script | Purpose |
|---|---|
| `parametric_surrogates.py` | Classical-baseline and regional-gap study / W1: classical parametric baselines (linear and quadratic response surface, Gaussian process) on the 23 design parameters, trained on the E2A training cars, against M2 on the estateback t... |
| `pod_reconstruction.py` | POD field reconstruction baseline. |
| `pod_transfer_error.py` | POD-baseline study B, exploratory diagnostic (not pre-specified, added after the POD numbers were seen): how much of the POD field error comes from the nearest-point transfer through the 20000 common points alone. |
