# Methodology — ML Classification Stage

Experimental protocol for the final classification model trained on radiomic features
extracted from ISIC images (Mask R-CNN segmentation → pyradiomics → ML).

**Dataset:** 76,715 images · 27,343 lesions · 5,791 patients · 4 classes
**Features:** radiomic features (4 channels × 7 feature classes). The active pipeline
does not apply a global correlation filter; preprocessing will be implemented inside
the classification folds.

| Class | Images | Share |
|-------|--------|-------|
| Benign-melanocytic | 48,974 | 63.6% |
| Malignant-melanocytic | 10,669 | 13.9% |
| Benign-non-melanocytic | 9,585 | 12.5% |
| Malignant-non-melanocytic | 7,746 | 10.1% |

---

## 1. Partition protocol

Everything else depends on this step being right.

**Frozen test set.** Hold out ~20% of **patients** (not images). Touched exactly **once**,
at the very end, to report the final number.

**Development set.** On the remaining 80% of patients, use
`StratifiedGroupKFold(n_splits=5, groups=patient_id)` for all model selection,
hyperparameter tuning, and ensemble weight fitting.

**The rule:** *no patient ever crosses a partition boundary.*

### Why grouping is mandatory

The dataset has 76,974 images but only 27,343 lesions — 29.8% of lesions have multiple
images (one lesion has 45). **28,394 images (37% of the dataset) belong to a lesion with
siblings.**

Under a random split the same physical lesion lands in both train and test. Radiomic
features of the same lesion photographed twice are nearly identical, so this is
effectively duplicating test rows into train. The model learns to *recognize lesions it
has seen*, not to *diagnose lesions it has not*. Reported metrics become inflated and the
evaluation no longer reflects the clinical use case.

Stratification (by class) and grouping (by patient) are orthogonal — you need both.

| | Guarantees | Does NOT prevent |
|---|---|---|
| Stratify by `target` | Equal class proportions per fold | Same lesion in train **and** test |
| Group by `patient_id` | No patient crosses the split | Class-imbalanced folds |

Grouping by `patient_id` (rather than `lesion_id`) is the conservative choice: it also
blocks the "same skin, same camera, same clinic" shortcut. With 5,791 patients there is
ample room for 5 folds.

---

## 2. Everything that learns parameters goes INSIDE the fold

Scaling, feature selection, SMOTE, imputation — all fit on the fold's training split only,
then applied to its validation/test split. Never fit on the full dataset.

Chain them in an `imblearn` pipeline so leakage is structurally impossible:

```python
from imblearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler",   StandardScaler()),
    ("selector", <filter | embedded | RFE>),
    ("balancer", <None | SMOTE()>),          # or class_weight on the model
    ("model",    <classifier>),
])
```

Passing this object to `cross_val_score` guarantees each step is fit on train only.

> The global preprocessing notebook is an archived prototype. Its historical outputs
> are not valid classifier inputs. Winsorization and correlation filtering must be
> estimators inside the fold-local pipeline.

---

## 3. Experimental design

### Factors

| Factor | Levels | Role |
|--------|--------|------|
| **Channels** | all · RGB · gray | **Ablation** — how much does each channel group contribute? |
| **Selection** | none · filter (ANOVA F) · embedded (L1 / tree importance) · RFE | Family comparison |
| **Balancing** | none · `class_weight` · SMOTE | Strategy comparison |
| **Models** | LogReg · LightGBM · RandomForest · MLP | One per inductive-bias family |

The channel factor is an **ablation, not a competition**: "all" is a superset of the other
two. The question it answers is *does colour add anything over greyscale?* If gray-only
lands within a point or two of the full pipeline at a fraction of the compute, that is a
finding — not a failed experiment.

### Staged protocol

A full factorial is 3 × 4 × 3 × 5 = **180 configurations** — computationally infeasible
and, worse, statistically unsound: selecting the max of 180 noisy estimates on the same
folds is a selection-bias machine.

| Stage | What varies | Configurations |
|-------|-------------|----------------|
| **1 — Screening** | All factors, but only 2 cheap models (LogReg, LightGBM), default hyperparameters | 3 × 4 × 3 × 2 = **72** |
| **2 — Deep dive** | Best (selection, balancing) fixed; 3 channels × 5 models, full hyperparameter search | 3 × 5 = **15** |
| **3 — Ensemble** | Top 3–5 models; weighting schemes: equal · score-proportional · genetic algorithm · logistic stacking | **4** |
| **4 — Frozen test** | Winner only | **1** |

**Total: 87 configurations**, only 15 of which carry expensive tuning.

Stage 2 yields the channel-ablation table and the model comparison in a single pass.

**Assumption to declare:** the staged design assumes factors do not interact strongly
(i.e. the best balancing strategy is the same regardless of channel). Standard and
reasonable, but it belongs in the limitations section.

**On SVM:** RBF-SVM is O(n²) on ~61k training rows — hours per fit. Excluded in favour of
the MLP as the dense non-linear model. Use `LinearSVC` or subsampling if it must be included.

---

## 4. Metrics

**Primary: macro-F1** (or balanced accuracy).

Plain accuracy is **not** a valid primary metric here — always predicting
"Benign-melanocytic" yields 63.6% and means nothing.

**Secondary:**
- One-vs-rest AUC
- Confusion matrix
- **Per-class recall for the malignant classes, reported explicitly** — in a clinical
  setting a false negative on melanoma does not cost the same as a false positive, and the
  thesis must say so.

---

## 5. Statistical comparison

Report **mean ± standard deviation across the 5 folds**, never a bare number.

To claim one model beats another, use a corrected paired t-test (Nadeau–Bengio) or
Friedman + Nemenyi when comparing many. Without this, 0.812 vs 0.809 is noise and the
ranking is anecdotal.

---

## 6. Ensemble

Take the top 3–5 models from the winning experiment and weight them.

**Genetic-algorithm weights are fit on the CV folds, never on the frozen test set.**

Compare the GA against simple baselines: equal weights, score-proportional weights, and
logistic-regression stacking. If the GA does not beat equal weighting, it does not go in
the thesis. If it does, that is evidence rather than an assertion.

---

## 7. Interpretability

**TreeSHAP on the best tree-based model** — fast and exact. Not KernelSHAP on the
heterogeneous ensemble, which is prohibitively slow.

Report global feature importance plus a handful of individual cases. Connect the findings
to the radiomics literature: are the dominant features texture, shape, or first-order?
Does the pattern make clinical sense?

---

## 8. What gets reported

1. Channel ablation table (with confidence intervals across folds)
2. Selection × balancing × model comparison table
3. Statistical test backing the winner
4. Ensemble vs best single model vs weighting baselines
5. **Final metric on the frozen test set** — the honest number
6. SHAP analysis + clinical discussion
7. Limitations — SMOTE in high dimensions and no-interaction assumption

---

## References for the write-up

- Grouped cross-validation: Roberts et al. (2017), *Cross-validation strategies for data
  with temporal, spatial, hierarchical, or phylogenetic structure*
- Corrected paired t-test: Nadeau & Bengio (2003), *Inference for the Generalization Error*
- SHAP: Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*
- SMOTE: Chawla et al. (2002), *SMOTE: Synthetic Minority Over-sampling Technique*
