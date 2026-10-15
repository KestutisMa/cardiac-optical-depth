# Prepared Synthetic Data

Datasets are stored under
`outputs/studies/<study>/derived/nn_dataset/emission_top_pixel_1d/`.

| Study ID | Total rows | Evaluated non-control rows | Design |
| --- | ---: | ---: | --- |
| `depth_sweep_v2` | 72 | 71 | 0.5--4.0 mm depth, 0.05 mm increments; 0.25 mm layer; conductivity multiplier 0.2 |
| `depth_sweep_optical_domains_v1` | 360 | 355 | The same 71 depths and a control at each of five optical settings |
| `substrate_factorial_r3` | 279 | 279 | 31 depths (0.5--3.5 mm, 0.1 mm increments) x 3 thicknesses x 3 conductivity multipliers |

The factorial thicknesses are 0.15, 0.25, and 0.50 mm; conductivity multipliers
are 0.15, 0.2, and 0.4. A multiplier scales conductivity directly (0.2 means an
80% reduction relative to surrounding tissue). The factorial reuses 31 matching
reference simulations and adds 248 coupled simulations, all with 15 ms EP duration.

Optical settings are `(mu_a, mu_s)` in inverse metres: reference `(120,436)`,
low/high absorption `(96,436)` / `(144,436)`, and low/high scattering
`(120,348.8)` / `(120,523.2)`. These are effective coefficients for the study's
finite geometry. All five settings are represented at training depths in the
grouped optical analysis; entire optical settings are not held out.

## Files

- `signals.npy`: float array `(N,100)` of CEEMDAN-denoised, aligned OAP windows,
  before fold-specific neural-network normalisation.
- `signals_raw.npy`: corresponding aligned windows before CEEMDAN denoising;
  not full-length detector movies or raw volumetric outputs.
- `targets_depth.npy` and `targets_thickness.npy`: labels in 50-micrometre voxel
  units. Multiply by **0.05** for millimetres. A zero thickness denotes control.
- `targets_conductivity_multiplier.npy`: direct multiplier, in the factorial dataset.
- `case_ids.json`: case IDs in the same order as the arrays.
- `domain_ids.json` and `source_case_ids.json`: optical-setting and source case
  identity for the combined optical dataset.
- `time_indices.npy`, `time_ms.npy`, `meta.json`: archived alignment and preparation
  metadata. The alignment window is 90 samples before and 9 after maximum dF/dt,
  with edge padding.

Arrays can be read with `numpy.load(path, allow_pickle=False)`. Rows are not
independent experimental replicates; variants share their underlying simulation
conditions. Keep the supplied depth groups intact when reproducing grouped CV.

## Normalisation And Timing

The released trained models use **shared bounds fitted only on each fold's
training traces**, using the 0.001 and 0.999 quantiles. Those bounds are applied
unchanged to validation/test traces and saved in each checkpoint as
`norm.gmin` and `norm.gmax`. This is not independent per-trace min--max scaling.
The channel construction and clipping are defined in `nn/depth_core.py`.

Historical primary/optical preparation metadata labelled the sample interval
as 1 ms by reading `ep_timedt_ms`. The source voxel files also record
`ep_spacedt_ms=0.1`, the optical snapshot interval used in the study. The factorial
preparer reads that latter attribute and records 0.1 ms. The historical metadata
and arrays are retained to reproduce the original analysis; use **0.1 ms per
sample** for physical-time interpretation of these windows. The network operates
on sample indices; changing a time-axis label does not change its inputs or
depth predictions. The original feature-baseline code reads the archived timing
metadata, which is retained for reproducibility of that baseline too.

## Results And Checkpoints

`outputs/studies/<study>/derived/nn/emission_top_pixel_1d/<run>/` contains
per-fold `splits.json`, metrics/history, and `best_depthcnn.pth`. Split indices
refer directly to dataset rows, including the original indexing of controls.
Controls are excluded from all train/validation/test splits. Test folds cover
each non-control row once; validation is a separate fold.

Saved predictions, including the frozen optical-perturbation analysis, are
included as CSV. That analysis uses the primary checkpoints without retraining;
duplicate copies are omitted and `checkpoint_hashes.json` identifies the shared
weights. The optical dataset contains the corresponding perturbed signals.
Results for the reported feature baseline, channel ablations and depth bands
are included. This release does not include abandoned exploratory runs.

Only JSON paths were made repository-relative. Dataset arrays, trained
checkpoints, and archived prediction values were not modified. `PROVENANCE.json`
records original hashes; `MANIFEST.sha256` identifies the released files.
