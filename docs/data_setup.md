# Datos: PC local y VM de Google Cloud

Esta guía cubre la obtención y disposición de los datos sin versionarlos en Git.
El pipeline espera archivos locales normales; en GCP esos archivos pueden ser
objetos expuestos mediante Cloud Storage FUSE.

## Estructura requerida

```text
data/buckets/
├── HAM10000/
│   ├── metadata.csv
│   ├── images/
│   │   └── ISIC_*.jpg
│   └── masks/
│       └── ISIC_*_segmentation.png
├── ISIC_archive/
│   ├── metadata.csv
│   └── ISIC_*.jpg
└── ISIC_masks/
    └── ISIC_*.png
```

`segment_isic.py` busca las imágenes como archivos inmediatos de
`SCD_ISIC_IMAGES_DIR`; no hace una búsqueda recursiva. Si la descarga crea un
subdirectorio `images/`, la variable debe apuntar a ese subdirectorio.

HAM10000 contiene las imágenes y máscaras *ground truth* usadas exclusivamente
para entrenar segmentación. `ISIC_masks` contiene predicciones del modelo y puede
regenerarse; nunca se deben confundir o limpiar las máscaras ground truth de
HAM10000.

## Opción A: PC local

### Recursos orientativos

- Python 3.12.10 para EDA, splits, pruebas y ejecución principal.
- 16 GB de RAM como base; 32 GB recomendados para EDA completo.
- Al menos 80 GB libres si se descarga el ISIC Archive completo, además del
  espacio necesario para HAM10000, entornos y resultados.
- CPU es suficiente para EDA y pruebas. Entrenar Mask R-CNN sin GPU es posible,
  pero no es una ejecución práctica del pipeline completo.

En macOS Apple Silicon, el runtime principal funciona de forma nativa. El runtime
legacy PyRadiomics 3.7 debe ejecutarse en Linux x86_64 (por ejemplo, la VM GCP):
las ruedas usadas por este proyecto no están disponibles para CPython 3.7 arm64.

### Crear los entornos

```bash
pyenv install -s 3.12.10
pyenv install -s 3.7.17
pyenv local 3.12.10

PYENV_VERSION=3.12.10 pyenv exec python -m venv .venv-mask

.venv-mask/bin/python -m pip install -r requirements/main.txt
.venv-mask/bin/python -m pip install -r requirements/torch-cpu.txt
.venv-mask/bin/python -m pip install -r requirements/dev.txt
```

Crear `.venv-features` localmente solo en Linux x86_64. En macOS Apple Silicon,
ejecutar ese paso en la VM GCP:

```bash
PYENV_VERSION=3.7.17 pyenv exec python -m venv .venv-features
.venv-features/bin/python -m pip install \
  pip==24.0 setuptools==68.0.0 wheel==0.42.0
.venv-features/bin/python -m pip install \
  -r requirements/radiomics-py37.txt
```

### Descargar ISIC Archive con el CLI oficial

El paquete `isic-cli` requiere Python 3.12 o posterior y es una herramienta de
adquisición opcional; no es una dependencia del entrenamiento.

```bash
.venv-mask/bin/python -m pip install -r requirements/data-tools.txt
.venv-mask/bin/isic --version

mkdir -p data/buckets/ISIC_archive
.venv-mask/bin/isic image download data/buckets/ISIC_archive
```

El comando completo descarga imágenes y metadata. Para una descarga selectiva:

```bash
.venv-mask/bin/isic image download \
  --search 'diagnosis_3:"Melanoma Invasive"' \
  data/buckets/ISIC_archive
```

También se pueden consultar colecciones y descargar solamente metadata:

```bash
.venv-mask/bin/isic collection list
.venv-mask/bin/isic metadata download
```

El proyecto oficial recomienda el snapshot de AWS Open Data para descargar el
archivo público completo y el CLI para acceso programático o filtrado. Véanse el
[repositorio oficial de isic-cli](https://github.com/ImageMarkup/isic-cli) y la
[documentación de la API de ISIC Archive](https://api.isic-archive.com/api/docs/swagger/).

Verificar el resultado antes de ejecutar notebooks o scripts:

```bash
test -f data/buckets/ISIC_archive/metadata.csv
find data/buckets/ISIC_archive -maxdepth 1 -type f \
  \( -name '*.jpg' -o -name '*.jpeg' -o -name '*.png' \) | wc -l
```

Si las imágenes quedaron dentro de `images/`, usar esa carpeta en
`SCD_ISIC_IMAGES_DIR`.

### Configurar rutas locales

```bash
export SCD_HAM10000_DIR="$PWD/data/buckets/HAM10000"
export SCD_ISIC_DIR="$PWD/data/buckets/ISIC_archive"
export SCD_ISIC_IMAGES_DIR="$PWD/data/buckets/ISIC_archive"
export SCD_ISIC_MASKS_DIR="$PWD/data/buckets/ISIC_masks"
export PYTHONPATH="$PWD/src"
```

Crear la carpeta local de predicciones si no existe:

```bash
mkdir -p "$SCD_ISIC_MASKS_DIR"
```

## Opción B: VM de Google Cloud con buckets

### Recursos orientativos

- Ubuntu x86_64 y disco persistente.
- Python 3.12.10 para el runtime principal y Python 3.7.17 para PyRadiomics.
- Una NVIDIA L4 de 24 GB para entrenamiento e inferencia.
- Como referencia, `g2-standard-16` (64 GB RAM) es un punto de partida para
  entrenamiento; aumentar a `g2-standard-32` (128 GB) si EDA/cachés lo requieren.
- Para inferencia imagen por imagen, `g2-standard-8` (32 GB RAM, la misma L4) es
  suficiente para el código actual.
- PyRadiomics no requiere GPU y puede ejecutarse después en una VM CPU x86_64.

Los tamaños anteriores son referencias operativas, no mínimos garantizados. El
uso real depende de `batch_size`, workers, tamaño de imagen y cachés.

### Buckets y contrato

Esta guía utiliza:

| Variable | Bucket | Acceso desde la VM | Contenido |
|---|---|---|---|
| `HAM_BUCKET` | `scd-ml-ham10000-$PROJECT_ID` | solo lectura | `metadata.csv`, `images/`, `masks/` ground truth |
| `ISIC_BUCKET` | `scd-ml-dataset` | solo lectura | `metadata.csv` e imágenes ISIC |
| `ISIC_MASKS_BUCKET` | `scd-ml-masks` | lectura/escritura | máscaras ISIC generadas |

Los nombres de bucket son globales. Ajustarlos si ya pertenecen a otro proyecto.
La service account de la VM necesita `roles/storage.objectViewer` para los dos
buckets de entrada y `roles/storage.objectUser` para el bucket de salida.

Definir el proyecto, ubicación y nombres:

```bash
export PROJECT_ID="$(gcloud config get-value project)"
export BUCKET_LOCATION="us-central1"
export HAM_BUCKET="scd-ml-ham10000-${PROJECT_ID}"
export ISIC_BUCKET="scd-ml-dataset"
export ISIC_MASKS_BUCKET="scd-ml-masks"
```

### Crear buckets (solo la primera vez)

Comprobar antes si existen:

```bash
gcloud storage buckets describe "gs://${HAM_BUCKET}"
gcloud storage buckets describe "gs://${ISIC_BUCKET}"
gcloud storage buckets describe "gs://${ISIC_MASKS_BUCKET}"
```

Crear únicamente los que falten:

```bash
gcloud storage buckets create "gs://${HAM_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${BUCKET_LOCATION}" \
  --uniform-bucket-level-access

gcloud storage buckets create "gs://${ISIC_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${BUCKET_LOCATION}" \
  --uniform-bucket-level-access

gcloud storage buckets create "gs://${ISIC_MASKS_BUCKET}" \
  --project="${PROJECT_ID}" \
  --location="${BUCKET_LOCATION}" \
  --uniform-bucket-level-access
```

### Cargar datos preparados

Desde la máquina que contiene los archivos locales:

```bash
gcloud storage cp \
  data/buckets/HAM10000/metadata.csv \
  "gs://${HAM_BUCKET}/metadata.csv"

gcloud storage rsync --recursive \
  data/buckets/HAM10000/images \
  "gs://${HAM_BUCKET}/images"

gcloud storage rsync --recursive \
  data/buckets/HAM10000/masks \
  "gs://${HAM_BUCKET}/masks"

gcloud storage rsync --recursive \
  data/buckets/ISIC_archive \
  "gs://${ISIC_BUCKET}"
```

`gcloud storage rsync` actualiza el destino para que coincida con el origen, pero
no elimina objetos adicionales salvo que se use explícitamente
`--delete-unmatched-destination-objects`. Consultar la
[referencia oficial de rsync](https://docs.cloud.google.com/sdk/gcloud/reference/storage/rsync).

### Instalar Cloud Storage FUSE en Ubuntu/Debian

```bash
sudo apt-get update
sudo apt-get install -y curl lsb-release

export GCSFUSE_REPO="gcsfuse-$(lsb_release -c -s)"
echo "deb [signed-by=/usr/share/keyrings/cloud.google.asc] https://packages.cloud.google.com/apt ${GCSFUSE_REPO} main" \
  | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
  | sudo tee /usr/share/keyrings/cloud.google.asc

sudo apt-get update
sudo apt-get install -y gcsfuse
```

En Compute Engine, Cloud Storage FUSE puede usar la service account adjunta. En
una sesión local usar Application Default Credentials:

```bash
gcloud auth application-default login
```

Véanse la [instalación oficial](https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/install)
y la [guía oficial de montaje](https://docs.cloud.google.com/storage/docs/cloud-storage-fuse/mount-bucket).

### Montar dentro del proyecto

Los puntos de montaje deben estar vacíos antes de ejecutar `gcsfuse`.

```bash
cd ~/scd-ml
mkdir -p \
  data/buckets/HAM10000 \
  data/buckets/ISIC_archive \
  data/buckets/ISIC_masks

gcsfuse --implicit-dirs -o ro \
  "${HAM_BUCKET}" \
  "$PWD/data/buckets/HAM10000"

gcsfuse --implicit-dirs -o ro \
  "${ISIC_BUCKET}" \
  "$PWD/data/buckets/ISIC_archive"

gcsfuse --implicit-dirs \
  "${ISIC_MASKS_BUCKET}" \
  "$PWD/data/buckets/ISIC_masks"
```

Exportar las rutas en cada nueva sesión:

```bash
export SCD_HAM10000_DIR="$PWD/data/buckets/HAM10000"
export SCD_ISIC_DIR="$PWD/data/buckets/ISIC_archive"
export SCD_ISIC_IMAGES_DIR="$PWD/data/buckets/ISIC_archive"
export SCD_ISIC_MASKS_DIR="$PWD/data/buckets/ISIC_masks"
export PYTHONPATH="$PWD/src"
```

Si el bucket ISIC usa el prefijo `images/`:

```bash
export SCD_ISIC_IMAGES_DIR="$PWD/data/buckets/ISIC_archive/images"
```

Verificar:

```bash
mount | grep gcsfuse
test -f "$SCD_HAM10000_DIR/metadata.csv"
test -d "$SCD_HAM10000_DIR/images"
test -d "$SCD_HAM10000_DIR/masks"
test -f "$SCD_ISIC_DIR/metadata.csv"
find "$SCD_ISIC_IMAGES_DIR" -maxdepth 1 -type f | head
```

### Vaciar solamente las predicciones ISIC

No se necesita eliminar el bucket para regenerar máscaras. Primero revisar los
objetos. El segundo comando es destructivo y debe ejecutarse únicamente contra el
bucket de predicciones, nunca contra HAM10000:

```bash
gcloud storage ls --recursive "gs://${ISIC_MASKS_BUCKET}" | head -50
gcloud storage rm "gs://${ISIC_MASKS_BUCKET}/**"
```

Esto conserva el bucket y elimina sus objetos actuales. `segment_isic.py
--overwrite` reemplaza también el manifiesto local.

### Detener o redimensionar la VM

Guardar notebooks, detener procesos y desmontar antes de apagar:

```bash
cd ~/scd-ml
sync
fusermount -u "$PWD/data/buckets/ISIC_masks"
fusermount -u "$PWD/data/buckets/ISIC_archive"
fusermount -u "$PWD/data/buckets/HAM10000"
mount | grep gcsfuse
```

Usar **Stop**, no **Delete**, para conservar el disco persistente. Los buckets no
se pierden, pero los montajes y variables de shell no sobreviven al reinicio: al
encender la VM hay que repetir la sección de montaje y los `export`.

## Smoke test común

```bash
PYTHONPATH=src .venv-mask/bin/python - <<'PY'
from scd_ml.paths import HAM10000_DIR, ISIC_DIR, ISIC_IMAGES_DIR, ISIC_MASKS_DIR

for name, path in {
    "HAM10000_DIR": HAM10000_DIR,
    "ISIC_DIR": ISIC_DIR,
    "ISIC_IMAGES_DIR": ISIC_IMAGES_DIR,
    "ISIC_MASKS_DIR": ISIC_MASKS_DIR,
}.items():
    print(f"{name}: {path} (exists={path.exists()})")
PY
```

Después de validar datos y rutas, continuar con el orden canónico documentado en
[pipeline.md](pipeline.md).
