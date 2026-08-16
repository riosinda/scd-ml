# Entornos Python

## Responsabilidades

El runtime principal usa Python 3.12.10. El extractor PyRadiomics se mantiene
aislado en Python 3.7.17 y se comunica exclusivamente mediante CSV.

```text
.venv-mask/         Python 3.12.10  scd_ml + Torch + EDA/tests
.venv-features/     Python 3.7.17   extract_radiomics.py
```

El repositorio no se instala como paquete. El runtime principal encuentra
`scd_ml` mediante `PYTHONPATH=src`; el extractor fue diseñado para no importarlo.
El entorno `.venv-features` está soportado en Linux x86_64. En macOS Apple
Silicon debe ejecutarse en la VM GCP por disponibilidad de wheels CPython 3.7.

## Creación con pyenv

```bash
pyenv install -s 3.12.10
pyenv install -s 3.7.17
pyenv local 3.12.10

PYENV_VERSION=3.12.10 pyenv exec python -m venv .venv-mask
PYENV_VERSION=3.7.17 pyenv exec python -m venv .venv-features
```

## Instalación con pip

```bash
.venv-mask/bin/python -m pip install -r requirements/main.txt
.venv-mask/bin/python -m pip install -r requirements/torch-cpu.txt
.venv-mask/bin/python -m pip install -r requirements/dev.txt

.venv-features/bin/python -m pip install \
  pip==24.0 setuptools==68.0.0 wheel==0.42.0
.venv-features/bin/python -m pip install \
  -r requirements/radiomics-py37.txt
```

En una VM GCP con CUDA 12.6, sustituir `torch-cpu.txt` por `torch-cu126.txt`.
No instalar las dos variantes en el mismo entorno.

La herramienta oficial `isic-cli` es opcional y se instala solamente cuando se
necesita descargar datos públicos del ISIC Archive:

```bash
.venv-mask/bin/python -m pip install -r requirements/data-tools.txt
```

La preparación completa para PC local y VM GCP está en
[data_setup.md](data_setup.md).

## Smoke tests documentados

Runtime principal:

```bash
.venv-mask/bin/python --version
.venv-mask/bin/python -m pip check
PYTHONPATH=src .venv-mask/bin/python -c "import scd_ml, torch, torchvision, pycocotools; print(scd_ml.__version__, torch.__version__, torchvision.__version__)"
.venv-mask/bin/python -c "import torch; from torchvision.ops import nms; print(torch.cuda.is_available(), nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5))"
```

Runtime radiomics:

```bash
.venv-features/bin/python --version
.venv-features/bin/python -m pip check
.venv-features/bin/python -c "import cv2, numpy, pandas, SimpleITK, radiomics; print(radiomics.__version__)"
.venv-features/bin/python -m compileall -q scripts/extract_radiomics.py
```

Estos comandos son instrucciones de verificación; el repositorio no crea ni modifica
automáticamente los entornos.
