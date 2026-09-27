# Contratos del pipeline

## Manifiesto ISIC

`results/splits/isic_classification.csv`:

| Columna | Contrato |
|---|---|
| `image_id` | ID único de imagen |
| `patient_id` | ID original; puede faltar |
| `lesion_id` | ID original; puede faltar |
| `group_id` | `patient:*`, fallback `lesion:*`, fallback `image:*` |
| `target` | Una de las cuatro clases ISIC |
| `split` | `train` o `test` |
| `cv_fold` | 0–4 solo dentro de train; vacío para test |

El test representa aproximadamente el 20 % debido a la restricción por grupos. No
puede compartir `group_id` con train.

## Manifiesto HAM10000

`results/splits/ham10000_segmentation.csv` contiene `image_id`, `lesion_id`,
`group_id`, `split`, `image_filename` y `mask_filename`. Sus splits son train, val
y test con proporciones aproximadas 70/10/20. Una lesión pertenece a un solo split.
El diagnóstico `dx` no se exporta.

## Evaluación de segmentación

Las predicciones se filtran con `score_threshold=0.5` y se binarizan con
`mask_threshold=0.5`. Se selecciona la detección válida con mayor score.

Si no existe detección válida, la máscara predicha es vacía y:

- precision, recall, Dice e IoU son 0 cuando el target contiene lesión;
- precision y recall son 0 cuando ambas máscaras están vacías;
- Dice e IoU son 1 cuando ambas máscaras están vacías.

El CSV por imagen incluye `image_id`, score, detecciones válidas, estado, TP, FP,
FN, TN y todas las métricas. El resumen reporta media/desviación macro y valor micro.

## Manifiesto de máscaras ISIC

`results/segmentation/isic_masks_manifest.csv` contiene una fila por imagen:

```text
image_id,image_path,mask_path,score,num_detections,status,error
```

Estados esperados: `segmented`, `no_detection`, `missing_image` y `error`.

## Handoff PyRadiomics

Python 3.7 produce dos archivos:

- `radiomics_features.csv`: una fila por extracción con estado `ok`.
- `radiomics_status.csv`: una fila por cada imagen del manifiesto de máscaras.

Estados radiomics: `ok`, `empty_mask`, `error` o `upstream_<status>`. El validador
Python 3.12 exige que los IDs `ok` sean exactamente los IDs presentes en features y
que todos los IDs del manifiesto tengan un estado. Esto hace visible la pérdida de
cohorte antes de construir cualquier clasificador.

El extractor usa el CSV de características como checkpoint durable y se reanuda
por `image_id`. También conserva las máscaras vacías ya verificadas y vuelve a
intentar estados incompletos o con error. Mientras está ejecutándose mantiene un
journal `radiomics_status.csv.resume`, que compacta de forma atómica al finalizar.
Solo `--overwrite` elimina el avance existente.

## Datasets fold-local de clasificación

`notebooks/preprocessing/02 fold-local classification datasets.ipynb` consume el
manifiesto ISIC bloqueado, la metadata cruda y el handoff validado de PyRadiomics.
Rechaza una extracción parcial: todo `image_id` del manifiesto debe tener un estado
radiomics explícito y cada estado `ok` debe tener exactamente una fila de features.

Para cada fold 0–4 ajusta imputación por mediana, winsorización p1–p99 y filtro de
correlación `|r| > 0.95` exclusivamente sobre el train del fold. Luego aplica esos
parámetros a su validación. La metadata clínica (`age_approx`,
`anatom_site_1`, `pixels_x`, `pixels_y` y `sex`) se conserva cruda en cada
Parquet; su imputación, codificación y escalado pertenecen al pipeline del modelo
y también se ajustan exclusivamente con train. La notebook escribe:

```text
results/classification/fold_datasets/
├── fold_0/ ... fold_4/
│   ├── train.parquet
│   ├── validation.parquet
│   ├── preprocessing_parameters.csv
│   └── dropped_features.csv
├── test_raw.parquet
├── cohort_exclusions.csv
├── cohort_coverage_by_class.csv
├── fold_summary.csv
├── preprocessing_parameters_all_folds.csv
├── dropped_features_all_folds.csv
├── feature_selection_frequency.csv
└── feature_selection_jaccard.csv
```

El test permanece crudo y congelado. Solo se transforma después de elegir la
configuración, al reajustar el pipeline completo sobre todo development. Las
extracciones no exitosas se excluyen de las matrices de modelado, pero permanecen
documentadas en `cohort_exclusions.csv`; nunca se pierden mediante un `inner join`.
Las tasas de cobertura por split y clase se conservan en
`cohort_coverage_by_class.csv` para hacer visible una posible exclusión diferencial.

## Clasificación por etapas

Los tres entrypoints reconstruyen el cohort desde el manifiesto, el handoff
PyRadiomics validado y la metadata cruda (`scd_ml.classification.data.load_cohort`).
No consumen los Parquet de `fold_datasets/`, que quedan como auditoría, porque la
ablación de canales exige reajustar el filtro de correlación sobre cada subconjunto.

Cada pipeline tiene dos partes, ambas ajustadas solo con el train de cada fold:

```text
head: ColumnTransformer[FoldLocalRadiomicsTransformer(canales), MetadataEncoder?] → StandardScaler
tail: selector (none | anova | l1 | rfe) → balancer (none | SMOTE) → modelo
```

La CV ajusta el head una sola vez por (canales, metadata, fold) y reconstruye el tail
en cada configuración. El refit final ejecuta las mismas dos partes sobre todo
development y serializa el pipeline completo. `class_weight="balanced"` se aplica en
el modelo; el MLP y XGBoost, que no lo admiten, reciben `sample_weight` equivalente.
El SVM RBF usa una aproximación de Nyström del kernel (`gamma = gamma_scale / k`)
seguida de `LinearSVC` y calibración sigmoide, porque un `SVC` exacto es O(n²). La metadata
clínica se limita a `age_approx` (mediana + indicador de faltante), `sex` y
`anatom_site_1` (one-hot con categoría `missing`). `pixels_x`/`pixels_y` e IDs nunca
son features.

### Etapa 1 — `results/classification/screening/`

| Archivo | Contenido |
|---|---|
| `run_config.json` | Protocolo; reanudar con otro protocolo exige `--overwrite` |
| `fold_scores.csv` | Una fila por configuración × fold con todas las métricas |
| `selected_features.csv` | Features retenidas por cada selector, por fold |
| `summary.csv` | Media y desviación estándar por configuración |
| `strategy_ranking.csv` | Rank medio de cada par (selección, balanceo) entre celdas canal × modelo |
| `selected_strategy.json` | Par ganador, `k` y `complete` |
| `feature_stability.csv` / `feature_stability_jaccard.csv` | Frecuencia de selección y Jaccard entre folds |
| `figures/strategy_ranking.png` | Rank medio de cada par (selección, balanceo) |
| `figures/screening_f1_heatmap.png` | F1-macro medio por estrategia × (canal, modelo) |
| `figures/feature_selection_stability.png` | Jaccard entre folds por selector y canal |
| `figures/selected_features_<canal>.png` | Top features por frecuencia y familias radiómicas, por selector |

### Etapa 2 — `results/classification/tuning/`

Estudios `<canal>_<modelo>` (2a, solo radiomics) y `<mejor_canal>_<modelo>_meta` (2b).
El mejor canal es el de mayor F1-macro medio entre modelos en 2a.

| Archivo | Contenido |
|---|---|
| `optuna.db` | Estudios TPE reanudables; cada trial reporta F1 por fold para el pruning |
| `best_params/<estudio>.json` | Configuración completa del mejor trial |
| `oof/<estudio>.parquet` | Probabilidades out-of-fold del mejor trial (insumo del ensamble) |
| `cv/<estudio>.csv`, `fold_scores.csv` | Métricas por fold del mejor trial |
| `selected_features/<estudio>.csv` | Features retenidas por el selector del mejor trial, por fold |
| `studies_summary.csv` | Media ± sd por estudio, ordenado por F1-macro |
| `channel_ablation.csv` | F1 por modelo y canal, con t-test corregido contra gray |
| `metadata_ablation.csv` | Radiomics vs radiomics + metadata por modelo |
| `pairwise_tests.csv` | Nadeau–Bengio del ganador contra cada estudio |
| `tuning_report.json` | Friedman entre modelos, estudios esperados/presentes |
| `winner.json` | Configuración ganadora, métricas CV y `complete` |
| `figures/studies_f1.png` | F1-macro ± sd por estudio (con `k` elegido) |
| `figures/per_class_recall.png` | Recall medio por clase y estudio |
| `figures/channel_ablation.png` | F1 por modelo y canal, con `*` si p < 0.05 contra gray |
| `figures/metadata_ablation.png` | Radiomics vs radiomics + metadata, con p-valor |
| `figures/optimization_history.png` | Mejor F1 acumulado por trial de Optuna |
| `figures/winner_selected_features.png` | Features del ganador por frecuencia entre folds y familia |

### Etapa 4 — `results/classification/test/`

`metrics.json`, `per_class.csv`, `confusion_matrix.csv`, `predictions.parquet`
(IDs + probabilidades), `coverage.csv` (imágenes de test excluidas por radiomics,
por clase) y `selected_features.csv` (features del modelo final). En `figures/`
quedan la matriz de confusión (conteos y normalizada), las curvas ROC y
precision–recall one-vs-rest, recall/F1 por clase y las familias de las features
seleccionadas. El modelo queda en `models/classification/winner.joblib` junto con su
configuración, columnas de entrada y orden de clases. Se ejecuta solo con un
`winner.json` completo y se niega a reemplazar resultados sin `--overwrite`.

### MLflow — `results/mlflow/`

Los tres entrypoints replican sus resultados en MLflow (desactivable con
`--no-mlflow`). Los archivos de `results/classification/` siguen siendo la fuente de
verdad: `winner.json` es lo único que decide qué se evalúa en el test.

| Experimento | Run padre | Runs hijos |
|---|---|---|
| `classification-screening` | Protocolo, estrategia elegida, CSV y figuras | Una por configuración: parámetros, media/sd de métricas y F1 por fold (`step` = fold) |
| `classification-tuning` | Configuración de la etapa 2, métricas del ganador, CSV y figuras | Una por estudio con el mejor trial; se reemplaza si Optuna encuentra otro mejor |
| `classification-test` | Una run por evaluación: métricas de test y CV, predicciones, figuras y modelo | — |

El id del run padre se guarda en `<output-dir>/mlflow_run.json`: una etapa reanudada
reabre ese run y solo agrega hijos faltantes; `--overwrite` crea un run nuevo. El
store por defecto es `sqlite:///results/mlflow/mlflow.db` con artefactos en
`results/mlflow/artifacts/`; `--mlflow-uri` o `MLFLOW_TRACKING_URI` lo reemplazan.
En el tuning, MLflow se escribe solo en `--stage all` o `report`, así que los
procesos paralelos de `--stage 2a --study` no compiten por el run padre.
