# Entornos Python

## Responsabilidades

El runtime principal usa Python 3.12.10. El extractor PyRadiomics se mantiene
aislado en Python 3.7.17 y se comunica exclusivamente mediante CSV.

```text
.venv/              Python 3.12.10  scd_ml + Torch + EDA/tests
.venv-features/     Python 3.7.17   extract_radiomics.py
```

No se debe ejecutar `pip install -e .` dentro de `.venv-features`: el paquete
principal declara Python 3.12 y el extractor fue diseñado para no importarlo.

## Creación con pyenv

```bash
pyenv install -s 3.12.10
pyenv install -s 3.7.17
pyenv local 3.12.10

PYENV_VERSION=3.12.10 pyenv exec python -m venv .venv
PYENV_VERSION=3.7.17 pyenv exec python -m venv .venv-features
```

## Instalación con pip

```bash
.venv/bin/python -m pip install -r requirements/main.txt
.venv/bin/python -m pip install -r requirements/torch-cpu.txt
.venv/bin/python -m pip install -r requirements/dev.txt
.venv/bin/python -m pip install -e . --no-deps

.venv-features/bin/python -m pip install \
  pip==24.0 setuptools==68.0.0 wheel==0.42.0
.venv-features/bin/python -m pip install \
  -r requirements/radiomics-py37.txt
```

En una VM GCP con CUDA 12.6, sustituir `torch-cpu.txt` por `torch-cu126.txt`.
No instalar las dos variantes en el mismo entorno.

## Smoke tests documentados

Runtime principal:

```bash
.venv/bin/python --version
.venv/bin/python -m pip check
.venv/bin/python -c "import scd_ml, torch, torchvision, pycocotools; print(scd_ml.__version__, torch.__version__, torchvision.__version__)"
.venv/bin/python -c "import torch; from torchvision.ops import nms; print(torch.cuda.is_available(), nms(torch.tensor([[0.,0.,1.,1.]]), torch.tensor([1.]), 0.5))"
```

Runtime radiomics:

```bash
.venv-features/bin/python --version
.venv-features/bin/python -m pip check
.venv-features/bin/python -c "import cv2, numpy, pandas, SimpleITK, radiomics; print(radiomics.__version__)"
.venv-features/bin/python -m compileall -q scripts/extract_radiomics.py scripts/03_extract_features.py
```

Estos comandos son instrucciones de verificación; el repositorio no crea ni modifica
automáticamente los entornos.
