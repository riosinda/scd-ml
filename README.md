# Skin Cancer Detection — ML Thesis

Pipeline de investigación para segmentación de lesiones, extracción de variables
radiomics y futura clasificación de cáncer de piel.

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
```

- HAM10000 se usa exclusivamente para segmentación.
- ISIC aporta el conjunto de futura clasificación y mantiene un test bloqueado del 20 %.
- El clasificador aún no forma parte del repositorio.
- UDEM queda reservado para validación externa futura.
- Los notebooks son EDA; no definen el pipeline de producción.

## Entornos de Python

Se usan dos entornos creados con `pyenv`, `venv` y `pip`. No se usa `uv`.

| Entorno | Python | Responsabilidad |
|---|---:|---|
| `.venv` | 3.12.10 | EDA, splits, segmentación, validación y futura clasificación |
| `.venv-features` | 3.7.17 | Solo `scripts/extract_radiomics.py` |

`.python-version` fija Python 3.12.10 como intérprete predeterminado. Para crear
ambos entornos sin cambiar repetidamente `pyenv local`:

```bash
pyenv install -s 3.12.10
pyenv install -s 3.7.17
pyenv local 3.12.10

PYENV_VERSION=3.12.10 pyenv exec python -m venv .venv
PYENV_VERSION=3.7.17 pyenv exec python -m venv .venv-features
```

Instalación del entorno principal:

```bash
.venv/bin/python -m pip install -r requirements/main.txt

# Elegir exactamente una variante de Torch:
.venv/bin/python -m pip install -r requirements/torch-cpu.txt
# .venv/bin/python -m pip install -r requirements/torch-cu126.txt  # GCP/L4

.venv/bin/python -m pip install -r requirements/dev.txt
.venv/bin/python -m pip install -e . --no-deps
```

Instalación del extractor radiomics:

```bash
.venv-features/bin/python -m pip install \
  pip==24.0 setuptools==68.0.0 wheel==0.42.0
.venv-features/bin/python -m pip install \
  -r requirements/radiomics-py37.txt
```

`pyproject.toml` no administra entornos ni dependencias. Solo declara el paquete
`scd_ml` para que `pip install -e . --no-deps` permita importarlo desde scripts,
tests y notebooks sin modificar `sys.path`. Las dependencias viven únicamente en
`requirements/`.

Más detalles y smoke tests: [docs/environments.md](docs/environments.md).

## Estructura

```text
configs/                    parámetros explícitos de segmentación/radiomics
notebooks/                  análisis exploratorio, no pipeline canónico
requirements/               dependencias separadas por runtime/plataforma
scripts/                    entrypoints de cada etapa
src/scd_ml/data/            manifiestos y validaciones de splits
src/scd_ml/segmentation/    dataset, modelo, entrenamiento, inferencia y métricas
src/scd_ml/features/        contrato CSV radiomics
tests/                      pruebas sintéticas; no leen data/
```

Los antiguos scripts `01_*`, `02_*` y `03_*` son wrappers temporales. Los nombres
sin numeración son los entrypoints canónicos.

## Ejecución del pipeline

### 1. Crear los manifiestos

```bash
.venv/bin/python scripts/prepare_ham10000_split.py
.venv/bin/python scripts/prepare_isic_split.py
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
.venv/bin/python scripts/train_segmenter.py
```

El entrenamiento:

- optimiza solo con HAM10000 train;
- monitorea Dice macro sobre validación;
- guarda cada mejora en `models/segmentation/mask_rcnn_best.pt`;
- restaura siempre el mejor checkpoint;
- evalúa HAM10000 test una sola vez después de restaurarlo.

Las métricas por imagen y el resumen macro/micro se guardan bajo
`results/segmentation/evaluation/`.

### 3. Segmentar ISIC

```bash
.venv/bin/python scripts/segment_isic.py
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

El extractor es autocontenido: no importa `scd_ml` y no genera Parquet. Produce:

```text
results/features/radiomics_features.csv
results/features/radiomics_status.csv
```

`radiomics_status.csv` contiene una fila para cada imagen del manifiesto, incluidas
máscaras vacías, errores de lectura y fallos upstream.

### 5. Validar el handoff

```bash
.venv/bin/python scripts/validate_radiomics.py
```

La validación exige IDs únicos, cobertura completa, correspondencia exacta entre
estado `ok` y filas de features, y variables radiomics numéricas.

El esquema completo está documentado en [docs/pipeline.md](docs/pipeline.md).

## Configuración y rutas

Los parámetros de entrenamiento y thresholds están en `configs/segmentation.yaml`.
Las clases de features PyRadiomics están en `configs/radiomics.yaml`.

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

## Pruebas

Las pruebas son sintéticas y no acceden a `data/`:

```bash
PYTHONPATH=src .venv/bin/python -m unittest discover -s tests -p 'test_*.py' -v
```

Cubren aislamiento de grupos, ausencia de imputación por target, early stopping,
casos de máscara vacía, precisión sin detección, contrato radiomics y compatibilidad
estática del extractor con Python 3.7.
