# Repository guidance

## Project contract

This repository implements lesion segmentation, radiomics extraction and the staged
four-class classifier (screening, Optuna tuning, frozen test). Classification must not
use HAM10000. Ensembling and SHAP are not implemented yet.

- HAM10000: Mask R-CNN segmentation only; split 70/10/20 by `lesion_id`.
- ISIC: four-class classification; grouped 80/20 holdout plus five development folds.
- UDEM: future external validation only.
- Never read or commit raw data while performing repository maintenance.
- Notebooks are exploratory and must not define production preprocessing.

## Python runtimes

```text
.venv-mask/        Python 3.12.10  segmentation, EDA, splits and tests
.venv-features/    Python 3.7.17   standalone PyRadiomics extractor only
```

Use pyenv + venv + pip. Do not add `uv`, `uv.lock`, or a third environment. Dependencies
belong in `requirements/*.txt`; the repository is not installed as a package and has no
`pyproject.toml`. Main-runtime commands use `PYTHONPATH=src`. Never import `scd_ml` from
`scripts/extract_radiomics.py` because the script is a Python 3.7 process boundary.

## Canonical entrypoints

1. `scripts/prepare_ham10000_split.py`
2. `scripts/prepare_isic_split.py`
3. `scripts/train_segmenter.py`
4. `scripts/evaluate_segmenter.py`
5. `scripts/segment_isic.py`
6. `scripts/extract_radiomics.py`
7. `scripts/validate_radiomics.py`
8. `scripts/screen_classifiers.py`
9. `scripts/tune_classifiers.py`
10. `scripts/evaluate_classifier.py`

There are no numbered compatibility wrappers. New logic belongs under `src/scd_ml/`,
except for the standalone Python 3.7 extractor.

## Invariants

- Preserve patient/lesion/image identifiers in manifests; never use them as classifier features.
- Never impute clinical fields from `target`.
- Fit imputation, winsorization, feature selection, scaling and resampling inside each
  training fold; the classifier head/tail split in `classification/pipeline.py` enforces it.
- `pixels_x`/`pixels_y` are acquisition fields and never classifier features.
- The frozen test is scored only from a complete `winner.json`, never with hand-picked settings.
- MLflow (`results/mlflow/`) mirrors classification results; files under
  `results/classification/` stay authoritative and MLflow never feeds a decision.
- Early stopping maximizes validation Dice and always restores the best checkpoint before test.
- A missing segmentation prediction has pixel precision 0, not 1.
- Every ISIC image receives a segmentation and radiomics status; no silent row dropping.
- Existing outputs require an explicit `--overwrite` before replacement.

See `README.md`, `docs/environments.md`, `docs/data_setup.md`, and `docs/pipeline.md`
for commands, data acquisition, GCP mounts and schemas.
