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
