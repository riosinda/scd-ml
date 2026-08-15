# Metodología — Etapa de Clasificación ML

Protocolo experimental para el modelo final de clasificación entrenado sobre características
radiómicas extraídas de imágenes ISIC (segmentación Mask R-CNN → pyradiomics → ML).

**Dataset:** 76,715 imágenes · 27,343 lesiones · 5,791 pacientes · 4 clases
**Características:** features radiómicas (4 canales × 7 clases de features). El pipeline
activo no aplica un filtro global de correlación; el preprocesamiento se implementará
dentro de los folds de clasificación.

| Clase | Imágenes | Proporción |
|-------|----------|------------|
| Benign-melanocytic | 48,974 | 63.6% |
| Malignant-melanocytic | 10,669 | 13.9% |
| Benign-non-melanocytic | 9,585 | 12.5% |
| Malignant-non-melanocytic | 7,746 | 10.1% |

---

## 1. Protocolo de partición

Todo lo demás depende de que este paso esté bien hecho.

**Conjunto de prueba congelado.** Aparta ~20% de los **pacientes** (no de las imágenes). Se
toca exactamente **una vez**, al final, para reportar el número definitivo.

**Conjunto de desarrollo.** Con el 80% restante de pacientes, usa
`StratifiedGroupKFold(n_splits=5, groups=patient_id)` para toda la selección de modelo, el
ajuste de hiperparámetros y el cálculo de pesos del ensamble.

**La regla:** *ningún paciente cruza jamás una frontera de partición.*

### Por qué la agrupación es obligatoria

El dataset tiene 76,974 imágenes pero solo 27,343 lesiones — el 29.8% de las lesiones tiene
múltiples imágenes (una lesión llega a tener 45). **28,394 imágenes (37% del dataset)
pertenecen a una lesión con hermanas.**

Con una partición aleatoria, la misma lesión física cae en train y en test. Las features
radiómicas de la misma lesión fotografiada dos veces son casi idénticas, así que esto
equivale a duplicar filas de test dentro de train. El modelo aprende a *reconocer lesiones
que ya vio*, no a *diagnosticar lesiones que no ha visto*. Las métricas se inflan y la
evaluación deja de reflejar el caso de uso clínico.

Estratificar (por clase) y agrupar (por paciente) son ejes ortogonales — necesitas ambos.

| | Garantiza | NO evita |
|---|---|---|
| Estratificar por `target` | Misma proporción de clases en cada fold | Que la misma lesión esté en train **y** en test |
| Agrupar por `patient_id` | Que ningún paciente cruce la partición | Que un fold quede desbalanceado en clases |

Agrupar por `patient_id` (en vez de `lesion_id`) es la opción conservadora: también bloquea
el atajo "misma piel, misma cámara, misma clínica". Con 5,791 pacientes hay margen de sobra
para 5 folds.

> **Advertencia:** al aplicar esto, las métricas van a **bajar** respecto a un split ingenuo.
> No es que el modelo empeore — es que por fin lo estás midiendo bien. Reportar ambos números
> y cuantificar la magnitud del sesgo es material valioso para la discusión.

---

## 2. Todo lo que aprende parámetros va DENTRO del fold

Escalamiento, selección de características, SMOTE, imputación — todo se ajusta únicamente con
la partición de entrenamiento del fold, y luego se aplica a validación/test. Nunca sobre el
dataset completo.

Encadénalos en un pipeline de `imblearn` para que la fuga sea estructuralmente imposible:

```python
from imblearn.pipeline import Pipeline

pipe = Pipeline([
    ("scaler",   StandardScaler()),
    ("selector", <filtro | embebido | RFE>),
    ("balancer", <None | SMOTE()>),          # o class_weight en el modelo
    ("model",    <clasificador>),
])
```

Al pasar este objeto a `cross_val_score`, sklearn garantiza que cada paso se ajuste solo con
train.

> El notebook de preprocesamiento global es un prototipo archivado. Sus outputs históricos
> no son entradas válidas para el clasificador. Winsorización y filtrado de correlación deben
> implementarse como estimadores dentro de cada fold.

---

## 3. Diseño experimental

### Factores

| Factor | Niveles | Rol |
|--------|---------|-----|
| **Canales** | todos · RGB · gray | **Ablación** — ¿cuánto aporta cada grupo de canales? |
| **Selección** | ninguna · filtro (ANOVA F) · embebido (L1 / importancia de árboles) · RFE | Comparación de familias |
| **Balanceo** | ninguno · `class_weight` · SMOTE | Comparación de estrategias |
| **Modelos** | LogReg · LightGBM · RandomForest · MLP | Uno por familia de sesgo inductivo |

El factor de canales es una **ablación, no una competencia**: "todos" es superconjunto de los
otros dos. La pregunta que responde es *¿el color aporta algo sobre la escala de grises?* Si
gray-only queda a uno o dos puntos del pipeline completo con una fracción del cómputo, eso es
un hallazgo — no un experimento fallido.

### Protocolo por etapas

El factorial completo son 3 × 4 × 3 × 5 = **180 configuraciones** — inviable
computacionalmente y, peor, estadísticamente insostenible: elegir el máximo de 180
estimaciones ruidosas sobre los mismos folds es una máquina de sesgo de selección.

| Etapa | Qué varía | Configuraciones |
|-------|-----------|-----------------|
| **1 — Screening** | Todos los factores, pero solo 2 modelos baratos (LogReg, LightGBM), hiperparámetros por defecto | 3 × 4 × 3 × 2 = **72** |
| **2 — Profundización** | Mejor (selección, balanceo) fija; 3 canales × 5 modelos, búsqueda completa de hiperparámetros | 3 × 5 = **15** |
| **3 — Ensamble** | Top 3–5 modelos; esquemas de ponderación: iguales · proporcional al score · algoritmo genético · stacking logístico | **4** |
| **4 — Test congelado** | Solo el ganador | **1** |

**Total: 87 configuraciones**, de las cuales solo 15 llevan tuning costoso.

La etapa 2 produce la tabla de ablación de canales y la comparación de modelos en una sola
pasada.

**Supuesto a declarar:** el diseño por etapas asume que los factores no interactúan
fuertemente (es decir, que la mejor estrategia de balanceo es la misma sin importar el canal).
Es estándar y razonable, pero va en la sección de limitaciones.

**Sobre el SVM:** un SVM-RBF es O(n²) sobre ~61k filas de entrenamiento — horas por ajuste.
Se excluye en favor del MLP como modelo denso no lineal. Si debe incluirse, usar `LinearSVC` o
submuestrear.

---

## 4. Métricas

**Primaria: F1-macro** (o balanced accuracy).

La *accuracy* simple **no** es una métrica primaria válida aquí — predecir siempre
"Benign-melanocytic" da 63.6% y no significa nada.

**Secundarias:**
- AUC one-vs-rest
- Matriz de confusión
- **Recall por clase maligna, reportado explícitamente** — en contexto clínico un falso
  negativo de melanoma no cuesta lo mismo que un falso positivo, y la tesis debe decirlo.

---

## 5. Comparación estadística

Reporta **media ± desviación estándar sobre los 5 folds**, nunca un número suelto.

Para afirmar que un modelo supera a otro, usa un t-test pareado corregido (Nadeau–Bengio), o
Friedman + Nemenyi si comparas muchos. Sin esto, 0.812 contra 0.809 es ruido y el ranking es
anecdótico.

---

## 6. Ensamble

Toma los 3–5 mejores modelos del experimento ganador y ponderálos.

**Los pesos del algoritmo genético se ajustan sobre los folds de CV, jamás sobre el test
congelado.**

Compara el GA contra baselines simples: pesos iguales, pesos proporcionales al score, y
stacking con regresión logística. Si el GA no le gana a promediar con pesos iguales, no va en
la tesis. Si le gana, entonces tienes evidencia en vez de una afirmación.

---

## 7. Interpretabilidad

**TreeSHAP sobre el mejor modelo de árboles** — rápido y exacto. No KernelSHAP sobre el
ensamble heterogéneo, que es prohibitivamente lento.

Reporta importancia global de features más algunos casos individuales. Conecta los hallazgos
con la literatura radiómica: ¿las features dominantes son de textura, de forma, o de primer
orden? ¿Tiene sentido clínico el patrón?

---

## 8. Qué se reporta al final

1. Tabla de ablación de canales (con intervalos de confianza sobre folds)
2. Tabla comparativa selección × balanceo × modelo
3. Prueba estadística que respalde al ganador
4. Ensamble vs mejor modelo individual vs baselines de ponderación
5. **Métrica final sobre el test congelado** — el número honesto
6. Análisis SHAP + discusión clínica
7. Limitaciones — SMOTE en alta dimensión y supuesto de no interacción

---

## Referencias para la redacción

- Validación cruzada agrupada: Roberts et al. (2017), *Cross-validation strategies for data
  with temporal, spatial, hierarchical, or phylogenetic structure*
- T-test pareado corregido: Nadeau & Bengio (2003), *Inference for the Generalization Error*
- SHAP: Lundberg & Lee (2017), *A Unified Approach to Interpreting Model Predictions*
- SMOTE: Chawla et al. (2002), *SMOTE: Synthetic Minority Over-sampling Technique*
