# Skin Cancer Detection — ML Thesis

Pipeline de investigación para segmentación de lesiones, extracción de variables
radiomics y clasificación de cáncer de piel en cuatro clases.

## Alcance actual

```text
HAM10000 metadata ──► split 70/10/20 por lesión ──► Mask R-CNN
                                                        │
ISIC metadata ─────► split 80/20 por paciente           │
        │                                               ▼
        └──────────────────────────────────────► máscaras ISIC
                                                        │
                                                        ▼ Python 3.7.17
                                                PyRadiomics CSV
                                                        │
                                                        ▼ Python 3.12.10
                                                validación de contrato
                                                        │
                                                        ▼
                              screening ─► tuning Optuna ─► test congelado
```

- HAM10000 se usa exclusivamente para segmentación.
- ISIC aporta el conjunto de clasificación y mantiene un test bloqueado del 20 %.
- La clasificación sigue el protocolo por etapas de [docs/metodologia.md](docs/metodologia.md):
  screening, tuning con CV agrupada y una única evaluación en test (el ensamble y SHAP
  quedan pendientes).
- UDEM queda reservado para validación externa futura.
- Los notebooks son EDA; no definen el pipeline de producción.

## Entornos de Python

Se usan dos entornos creados con `pyenv`, `venv` y `pip`. No se usa `uv`.

| Entorno | Python | Responsabilidad |
|---|---:|---|
| `.venv-mask` | 3.12.10 | EDA, splits, segmentación, validación y futura clasificación |
| `.venv-features` | 3.7.17 | Solo `scripts/extract_radiomics.py` |

`.python-version` fija Python 3.12.10 como intérprete predeterminado. Para crear
ambos entornos sin cambiar repetidamente `pyenv local`:

```bash
pyenv install -s 3.12.10
pyenv install -s 3.7.17
pyenv local 3.12.10

PYENV_VERSION=3.12.10 pyenv exec python -m venv .venv-mask
PYENV_VERSION=3.7.17 pyenv exec python -m venv .venv-features
```

Instalación del entorno principal:

```bash
.venv-mask/bin/python -m pip install -r requirements/main.txt

# Elegir exactamente una variante de Torch:
.venv-mask/bin/python -m pip install -r requirements/torch-cpu.txt
# .venv-mask/bin/python -m pip install -r requirements/torch-cu126.txt  # GCP/L4

.venv-mask/bin/python -m pip install -r requirements/dev.txt
```

Instalación del extractor radiomics. El orden es importante en Python 3.7:

```bash
.venv-features/bin/python -m pip install \
  pip==24.0 setuptools==68.0.0 wheel==0.42.0

.venv-features/bin/python -m pip install \
  numpy==1.21.6 \
  pandas==1.3.5 \
  opencv-python-headless==4.13.0.92

.venv-features/bin/python -m pip install \
  PyWavelets==1.3.0 \
  pykwalify==1.8.0 \
  ruamel.yaml==0.17.21 \
  docopt==0.6.2 \
  six==1.17.0 \
  python-dateutil==2.9.0.post0 \
  pytz==2024.2 \
  tqdm==4.68.0

.venv-features/bin/python -m pip install \
  SimpleITK==2.2.1 \
  --only-binary=:all:

.venv-features/bin/python -m pip install \
  pyradiomics==3.1.0 \
  --only-binary=:all: \
  --no-deps

.venv-features/bin/python -m pip check
.venv-features/bin/python -c \
  "import cv2, numpy, pandas, SimpleITK, radiomics; print(radiomics.__version__)"
```

Primero se fijan las dependencias compatibles, después se instala SimpleITK desde
su wheel binario y PyRadiomics se instala al final sin volver a resolver ni cambiar
las dependencias. El extractor actual no usa `pydicom`.

No se instala el repositorio como paquete y no se usa `pyproject.toml`. El código
principal vive bajo `src/`; los comandos canónicos establecen `PYTHONPATH=src`.
Todas las dependencias viven exclusivamente en `requirements/`.

Más detalles y smoke tests: [docs/environments.md](docs/environments.md).

La preparación de datos tiene dos recorridos documentados:

- **PC local:** descarga directa con el CLI oficial de ISIC y Torch CPU.
- **VM GCP:** buckets, IAM, carga con `gcloud storage`, montaje con GCS Fuse,
  redimensionamiento y remonte después de reiniciar.

Comandos y estructura completa: [docs/data_setup.md](docs/data_setup.md).

## Estructura

```text
configs/                    parámetros explícitos de segmentación/radiomics
notebooks/                  análisis exploratorio, no pipeline canónico
requirements/               dependencias separadas por runtime/plataforma
scripts/                    entrypoints de cada etapa
src/scd_ml/data/            manifiestos y validaciones de splits
src/scd_ml/segmentation/    dataset, modelo, entrenamiento, inferencia y métricas
src/scd_ml/features/        contrato CSV radiomics
src/scd_ml/classification/  cohort, pipelines fold-local, CV, tuning y estadística
tests/                      pruebas sintéticas; no leen data/
```

Los entrypoints canónicos tienen nombres descriptivos y no dependen de una
numeración: `train_segmenter.py`, `segment_isic.py`, `extract_radiomics.py`,
`screen_classifiers.py`, `tune_classifiers.py` y `evaluate_classifier.py`.

## Ejecución del pipeline

### 1. Crear los manifiestos

```bash
PYTHONPATH=src .venv-mask/bin/python scripts/prepare_ham10000_split.py
PYTHONPATH=src .venv-mask/bin/python scripts/prepare_isic_split.py
```

HAM10000 queda dividido aproximadamente 70/10/20 por `lesion_id`. ISIC usa
`patient_id`, con fallback a `lesion_id` e `image_id`, y asigna cinco folds dentro
del 80 % de desarrollo. El script puede derivar el target ISIC a partir de
`diagnosis_1` y `melanocytic`; no imputa variables clínicas.

Los manifiestos se escriben en:

```text
results/splits/ham10000_segmentation.csv
results/splits/isic_classification.csv
```

Para reemplazar un manifiesto existente debe pasarse `--overwrite`.

### 2. Entrenar y evaluar segmentación

```bash
PYTHONPATH=src .venv-mask/bin/python scripts/train_segmenter.py
```

El entrenamiento:

- optimiza solo con HAM10000 train;
- monitorea Dice macro sobre validación;
- guarda cada mejora en `models/segmentation/mask_rcnn_best.pt`;
- restaura siempre el mejor checkpoint;
- evalúa HAM10000 test una sola vez después de restaurarlo.

Las métricas por imagen y el resumen macro/micro se guardan bajo
`results/segmentation/evaluation/`.

Para analizar las curvas, auditar el resumen de test y localizar los peores casos,
abrir [notebooks/evaluation/01 segmentation results.ipynb](notebooks/evaluation/01%20segmentation%20results.ipynb)
en la misma máquina donde se ejecutó el entrenamiento. La notebook solo lee:

```text
results/segmentation/training/training_history.csv
results/segmentation/evaluation/ham10000_test_summary.csv
results/segmentation/evaluation/ham10000_test_per_image.csv
```

Los CSV bajo `results/` son artefactos generados y están excluidos de Git. Por eso
no llegan con `git pull`; deben analizarse en la VM que los produjo o sincronizarse
mediante almacenamiento de objetos.

### 3. Segmentar ISIC

```bash
PYTHONPATH=src .venv-mask/bin/python scripts/segment_isic.py
```

El resultado canónico es `results/segmentation/isic_masks_manifest.csv`. Cada fila
conserva `image_id`, rutas, score, número de detecciones, estado y error. Una imagen
sin detección genera una máscara vacía y estado `no_detection`; nunca desaparece de
la cohorte.

### 4. Extraer radiomics en Python 3.7

```bash
.venv-features/bin/python scripts/extract_radiomics.py \
  --masks-manifest results/segmentation/isic_masks_manifest.csv
```

La extracción se reanuda automáticamente si ya existen
`radiomics_features.csv` o `radiomics_status.csv`: conserva las características
terminadas y procesa únicamente IDs pendientes, inconsistentes o con error. Para
descartar el avance y comenzar nuevamente debe pasarse `--overwrite`.

El extractor es autocontenido: no importa `scd_ml` y no genera Parquet. Produce:

```text
results/features/radiomics_features.csv
results/features/radiomics_status.csv
```

`radiomics_status.csv` contiene una fila para cada imagen del manifiesto, incluidas
máscaras vacías, errores de lectura y fallos upstream.

### 5. Validar el handoff

```bash
PYTHONPATH=src .venv-mask/bin/python scripts/validate_radiomics.py
```

La validación exige IDs únicos, cobertura completa, correspondencia exacta entre
estado `ok` y filas de features, y variables radiomics numéricas.

El esquema completo está documentado en [docs/pipeline.md](docs/pipeline.md).

### 6. Clasificación

```bash
# Etapa 1: canales × selección × balanceo con LogReg/XGBoost (72 configs × 5 folds)
PYTHONPATH=src .venv-mask/bin/python scripts/screen_classifiers.py

# Etapa 2: Optuna por canal × modelo (2a) y ablación con metadata (2b), más el reporte
PYTHONPATH=src .venv-mask/bin/python scripts/tune_classifiers.py

# Etapa 4: refit del ganador sobre todo development y evaluación única del test
PYTHONPATH=src .venv-mask/bin/python scripts/evaluate_classifier.py
```

El screening y el tuning se reanudan si se interrumpen; `--overwrite` descarta el
avance. Para repartir el tuning entre procesos se puede ejecutar
`--stage 2a --study <canal>_<modelo>` en paralelo y cerrar con `--stage 2b` y
`--stage report`. `evaluate_classifier.py` solo acepta un `winner.json` completo
(5 folds, 20 estudios) y se niega a reemplazar un test ya evaluado sin `--overwrite`.

La etapa 2 ajusta LogReg, XGBoost, RandomForest, MLP y SVM RBF (Nyström). Cada
etapa guarda figuras en `<output-dir>/figures/` (ranking de estrategias, estabilidad
y familias de features seleccionadas, ablaciones, historial de Optuna, matriz de
confusión y curvas ROC/PR) y replica sus resultados en MLflow:

```bash
.venv-mask/bin/mlflow ui --backend-store-uri sqlite:///results/mlflow/mlflow.db
```

`--no-mlflow` desactiva el registro y `--mlflow-uri` (o `MLFLOW_TRACKING_URI`) apunta
a otro servidor. Los archivos en `results/classification/` siguen siendo la fuente
de verdad.

## Configuración y rutas

Los parámetros de entrenamiento y thresholds están en `configs/segmentation.yaml`.
Las clases de features PyRadiomics están en `configs/radiomics.yaml`.
El diseño experimental de clasificación (factores, `k`, presupuesto de trials) está
en `configs/classification.yaml`; los rangos de búsqueda de Optuna están en
`src/scd_ml/classification/tuning.py`.

Las rutas pueden sobrescribirse con:

```bash
export SCD_DATA_DIR=/path/to/data
export SCD_MODELS_DIR=/path/to/models
export SCD_RESULTS_DIR=/path/to/results
export SCD_HAM10000_DIR=/path/to/HAM10000
export SCD_ISIC_DIR=/path/to/isic
export SCD_ISIC_IMAGES_DIR=/path/to/isic/images
export SCD_ISIC_MASKS_DIR=/path/to/isic/masks
```

Ejemplo para los puntos de montaje recomendados dentro del repositorio:

```bash
export SCD_HAM10000_DIR="$PWD/data/buckets/HAM10000"
export SCD_ISIC_DIR="$PWD/data/buckets/ISIC_archive"
export SCD_ISIC_IMAGES_DIR="$PWD/data/buckets/ISIC_archive"
export SCD_ISIC_MASKS_DIR="$PWD/data/buckets/ISIC_masks"
export PYTHONPATH="$PWD/src"
```

La descarga local con `isic-cli`, creación/carga de buckets y montaje en GCP se
documentan en [docs/data_setup.md](docs/data_setup.md).

## Pruebas

Las pruebas son sintéticas y no acceden a `data/`:

```bash
PYTHONPATH=src .venv-mask/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Cubren aislamiento de grupos, ausencia de imputación por target, early stopping,
casos de máscara vacía, precisión sin detección, contrato radiomics, compatibilidad
estática del extractor con Python 3.7, preprocesamiento fold-local, ausencia de IDs
como features, cobertura OOF, t-test corregido y bloqueo del test congelado.
