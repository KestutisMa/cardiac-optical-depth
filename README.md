# Cardiac optical depth

Analysis code, prepared synthetic optical signals, and trained checkpoints for:

**Physics-guided deep learning inference of intramural slow-conduction depth
from simulated epicardial optical mapping**

Kestutis Maciunas and coauthors, *Scientific Reports* (accepted).

The article DOI will be added when available. This repository is the
paper-specific analysis release, not a general-purpose simulation framework.
Release **v1.0.0** identifies the initial publication package.

## Contents

- `nn/`: the 1D residual network, input channels, normalisation, training, and evaluation.
- `configs/`: portable configurations exported from the actual saved runs.
- `scripts/`: cross-validation, interpretable feature baseline, depth summaries,
  factorial analysis, and verification/evaluation entry points.
- `outputs/studies/`: prepared synthetic traces, labels, case identifiers,
  per-fold splits, checkpoints, and numerical results for the reported analyses.
- `DATA.md`: data format and preprocessing details.
- `PROVENANCE.json` and `MANIFEST.sha256`: source/artifact provenance and release checksums.

The coupled EP/photon-transport simulation framework, raw volumetric simulation
outputs, remote job management, and experimental recordings are not included.
The supplied synthetic traces are sufficient to rerun the released ML analyses
without those components. No patient data are included.

## Quick Start

Run commands from the repository root. Python 3.10--3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/verify_release.py
```

The verification command independently recomputes the archived metrics, checks
dataset counts and the complete factorial grid, tests fold separation, checks
the frozen-model hashes, and verifies the release checksums. It requires only
NumPy in addition to the Python standard library; no GPU is needed.

| Analysis | Non-control traces | Pooled RMSE (mm) | R squared | Pearson r |
| --- | ---: | ---: | ---: | ---: |
| Primary depth sweep | 71 | 0.298 | 0.916 | 0.963 |
| Five optical settings, grouped by depth | 355 | 0.190 | 0.966 | 0.986 |
| Substrate factorial, grouped by depth | 279 | 0.476 | 0.717 | 0.852 |

These are separate evaluation designs, not directly interchangeable estimates.
The primary result uses interleaved folds; the other two withhold all variants
at each exact test depth. All exclude controls from training and evaluation.

## Evaluate Saved Models

```bash
python scripts/evaluate_saved.py --run primary
python scripts/evaluate_saved.py --run optical
python scripts/evaluate_saved.py --run factorial
```

The default is CPU; `--device cuda` selects a compatible GPU. New outputs go to
`generated/evaluation/`, leaving archived results unchanged. The evaluator uses
each checkpoint's own configuration, normalisation bounds, and splits rather
than the development YAML defaults. The primary and optical runs use stochastic
MC-dropout evaluation; reruns across devices/library versions can differ from
the archived predictions. Verification of the archived predictions is exact
to numerical tolerance and does not resample the network.

Additional `--run` choices are `raw_only`, `raw_diff`, and `raw_log` for the
reported channel ablations. The original checkpoints are small enough to be
included directly in Git, without Git LFS. Load checkpoints only from a trusted
source and verify their checksums.

## Repeat Cross-Validation

These commands **retrain** the network and may take substantially longer than
saved-model evaluation. Use new run identifiers to preserve the archived runs.

```bash
python scripts/run_nn_cv.py --outputs-root "$PWD/outputs" \
  --study-id depth_sweep_v2 --dataset-name emission_top_pixel_1d \
  --k 5 --cv-policy interleaved --config configs/primary.yaml \
  --run-id reproduction_primary

python scripts/run_nn_cv.py --outputs-root "$PWD/outputs" \
  --study-id depth_sweep_optical_domains_v1 --dataset-name emission_top_pixel_1d \
  --k 5 --cv-policy grouped_depth_interleaved --config configs/optical.yaml \
  --run-id reproduction_optical

python scripts/run_nn_cv.py --outputs-root "$PWD/outputs" \
  --study-id substrate_factorial_r3 --dataset-name emission_top_pixel_1d \
  --k 5 --cv-policy grouped_depth_interleaved --config configs/factorial.yaml \
  --run-id reproduction_factorial
```

The configurations default to CPU for portability; change `device` to `cuda`
when appropriate. Retraining is not expected to produce bitwise identical
weights across platforms. `nn/exp_dev.yaml` is the development configuration,
not a substitute for the run-specific `configs/*.yaml` files.

The interpretable feature baseline can be rerun without changing its archive:

```bash
python scripts/run_nn_feature_baseline_cv.py \
  --cv-run-dir outputs/studies/depth_sweep_v2/derived/nn/emission_top_pixel_1d/cv_20260104T134115Z \
  --out-dir generated/baseline_features
```

`prepare_nn_study_dataset.py` documents the original extraction/CEEMDAN pipeline.
It requires external raw simulation files, which are not distributed here, and
the optional `requirements-preprocessing.txt`. The already prepared traces are
the reproducible starting point of this release. CEEMDAN is stochastic; its
original outputs are retained rather than regenerated during ML verification.

## Environment And Provenance

The analysis container used Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3,
PyTorch 2.6.0 (CUDA 12.4 build), PyYAML 6.0.3, and Matplotlib 3.10.8.
The requirements allow compatible newer versions; recorded checkpoints and
predictions are the reference artifacts. Release-only compatibility changes
are listed in `PROVENANCE.json`.

## Licence And Citation

Original code and model checkpoints are available under the **PolyForm
Noncommercial License 1.0.0** (`LICENSE`). The synthetic data and numerical
results, excluding checkpoints, use **CC BY-NC 4.0** (`LICENSE-DATA`). These are
noncommercial/source-available terms, not an OSI-approved open-source licence.
PolyForm expressly permits the listed educational/public-research and other
noncommercial organisations regardless of funding; the full terms govern.

For commercial uses outside the permissions granted by those licences,
[open a GitHub issue](https://github.com/KestutisMa/cardiac-optical-depth/issues/new)
to request a contact channel for separate permission. Do not post confidential
information in public issues. Third-party dependencies retain
their own licences; see `THIRD_PARTY.md`. No licence prevents independent
implementation of the published scientific method or overrides statutory rights.

When redistributing original code or checkpoints, including modified versions,
retain the `Required Notice:` lines in `NOTICE`, crediting Kestutis Maciunas
and contributors, identifying the accompanying paper, and linking to this
repository, as required by the PolyForm licence. When sharing the licensed
data or result files, provide attribution and the other notices required by CC BY-NC 4.0;
the supplied attribution details are in `NOTICE`.

For research publications using this work, please cite the accompanying article.
GitHub's **Cite this repository** feature provides the article citation from
`CITATION.cff`; its DOI will be added when available. This scholarly-citation
request is separate from the licences' attribution requirements and does not
create an additional licence condition.
This code is for research, not clinical decision-making.
