# Repository guidance

## Project contract

This repository currently implements lesion segmentation and radiomics extraction.
Classification is the next module and must not use HAM10000.

- HAM10000: Mask R-CNN segmentation only; split 70/10/20 by `lesion_id`.
- ISIC: future four-class classification; grouped 80/20 holdout plus five development folds.
- UDEM: future external validation only.
- Never read or commit raw data while performing repository maintenance.
- Notebooks are exploratory and must not define production preprocessing.

## Python runtimes

```text
.venv/             Python 3.12.10  main package, segmentation, EDA and tests
.venv-features/    Python 3.7.17   standalone PyRadiomics extractor only
```

Use pyenv + venv + pip. Do not add `uv`, `uv.lock`, or a third environment. Dependencies
belong in `requirements/*.txt`; `pyproject.toml` contains package/tool metadata only.
Never import `scd_ml` from `scripts/extract_radiomics.py` because the script is a Python
3.7 process boundary.

## Canonical entrypoints

1. `scripts/prepare_ham10000_split.py`
2. `scripts/prepare_isic_split.py`
3. `scripts/train_segmenter.py`
4. `scripts/evaluate_segmenter.py`
5. `scripts/segment_isic.py`
6. `scripts/extract_radiomics.py`
7. `scripts/validate_radiomics.py`

The numbered scripts are compatibility wrappers only. New logic belongs under
`src/scd_ml/`, except for the standalone Python 3.7 extractor.

## Invariants

- Preserve patient/lesion/image identifiers in manifests; never use them as classifier features.
- Never impute clinical fields from `target`.
- Fit future imputation, winsorization, feature selection and scaling inside each training fold.
- Early stopping maximizes validation Dice and always restores the best checkpoint before test.
- A missing segmentation prediction has pixel precision 0, not 1.
- Every ISIC image receives a segmentation and radiomics status; no silent row dropping.
- Existing outputs require an explicit `--overwrite` before replacement.

See `README.md`, `docs/environments.md`, and `docs/pipeline.md` for commands and schemas.
