# Segmentación de miocardio y diagnóstico cardíaco con U-Net 2D

Segmenta estructuras del corazón en resonancia magnética de eje corto y usa esas máscaras para medir el corazón y predecir el diagnóstico. Toda la segmentación es 2D: cada corte se procesa por separado, sin cortes vecinos ni secuencia temporal.

El proyecto tiene dos etapas:

1. Segmentar el miocardio, como pide el enunciado del curso (maximizar el IoU promedio).
2. Una extensión: segmentar ventrículo derecho (VD), miocardio y ventrículo izquierdo (VI), medir biomarcadores, clasificar al paciente en 5 grupos y marcar los casos dudosos.

## Resultados

### Etapa 1: miocardio

IoU promedio por imagen, a la resolución original de 256×256:

| Validación | Test |
|---|---|
| 0,728 | 0,747 |

Una versión anterior entrenaba a 128×128 y reducía la máscara con `nearest`, que la corre medio píxel respecto de la imagen. Esa versión reportaba 0,748 en test medido a 128×128, pero a 256×256 daba 0,737. Con la misma red y la misma semilla, medido a 256×256:

| Entrenamiento | Validación | Test |
|---|---|---|
| 128×128, máscara con `nearest` | 0,705 | 0,737 |
| 128×128, máscara con `nearest-exact` | 0,703 | 0,727 |
| 128×128, máscara con promedio y umbral | 0,719 | 0,745 |
| 256×256, sin reducir (el modelo actual) | 0,728 | 0,747 |

Se eligió el de 256×256 por su IoU de validación. Es una corrida por variante, y el IoU de validación salta hasta 0,03 entre épocas, así que las diferencias chicas no son concluyentes.

### Etapa 2: ACDC, los 50 pacientes del test oficial

Dice 3D por paciente, en fin de diástole (FD) y fin de sístole (FS):

| Estructura | FD | FS |
|---|---|---|
| Ventrículo izquierdo | 0,969 | 0,931 |
| Miocardio | 0,904 | 0,917 |
| Ventrículo derecho | 0,941 | 0,883 |

El mejor método de la [tabla oficial del ACDC](https://www.creatis.insa-lyon.fr/Challenge/acdc/results.html) (Isensee) tiene 0,967/0,928 en VI, 0,896/0,919 en miocardio y 0,946/0,904 en VD. La comparación es orientativa, porque esta evaluación no pasó por la plataforma oficial.

Biomarcadores contra los medidos con las máscaras de los expertos:

| Biomarcador | Sesgo | Desvío del error | Correlación |
|---|---|---|---|
| Fracción de eyección (FE) del VI | +0,1 puntos | 3,5 puntos | 0,988 |
| FE del VD | +2,1 puntos | 7,3 puntos | 0,870 |
| Volumen del VI en FD | −2,1 ml | 5,5 ml | 0,997 |
| Masa del miocardio | +2,9 g | 8,9 g | 0,985 |

Diagnóstico en 5 grupos:

| Biomarcadores usados | Validación cruzada en training | Test | IC 95 % en test |
|---|---|---|---|
| Máscaras predichas (el sistema) | 92,1 % | 94 % (47/50) | 84 a 98 % |
| Máscaras de expertos (cota superior) | 93,9 % | 94 % (47/50) | 84 a 98 % |

Los errores en test fueron un infarto predicho como dilatada, una dilatada predicha como infarto y un VD anormal predicho como normal.

El módulo de seguridad marcó 6 de 50 pacientes y entre ellos estaban 2 de los 3 errores. Los 44 no marcados aciertan el 97,7 %. En training marcó 20 de 100, con 6 de los 7 errores.

## Datos

Este repositorio no incluye datos, modelos ni imágenes de pacientes.

Etapa 1: versión preprocesada de ACDC y York, entregada por el curso. Se copia en `data/cardiac_data/` como `{train,val,test}_{imgs,annot}.npy`. Son imágenes uint8 de 256×256 (4.564, 502 y 1.240), y las máscaras traen 0 fondo, 1 miocardio y 2 cavidad del VI. El miocardio ocupa cerca del 1,2 % de los píxeles.

Etapa 2: [ACDC original](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb), en `data/ACDC/database/`. Son 150 pacientes (100 de training y 50 de test), con 30 por grupo: normal (NOR), infarto previo (MINF), miocardiopatía dilatada (DCM), miocardiopatía hipertrófica (HCM) y VD anormal (RV). Cada uno trae imágenes y anotaciones en FD y FS, altura y peso. Las máscaras traen 0 fondo, 1 VD, 2 miocardio y 3 VI. La licencia es CC BY-NC-SA 4.0 y su uso exige citar a Bernard et al. (2018).

## Metodología

### Etapa 1

- Preprocesamiento: la clase 1 pasa a ser el único positivo. Las imágenes se dividen por 255 y se estandarizan con la media y el desvío de train.
- Modelo: U-Net de 4 niveles con `base_channels=16` (1,9 millones de parámetros), a 256×256. Las imágenes no se reducen, así que la máscara no necesita remuestreo.
- Entrenamiento: BCE + Dice, porque el miocardio es muy minoritario. Adam con tasa 1e-3 y `ReduceLROnPlateau` sobre el IoU de validación, 25 épocas, lotes de 16. La aumentación usa flips y rotaciones de 90°.
- Selección: se guarda la época con mejor IoU de validación (la 14).

### Etapa 2

1. Preprocesamiento. Cada volumen se recorta a los percentiles 1 y 99 y se estandariza con sus propios valores, porque cada equipo usa otra escala. Cada corte se remuestrea a 1,5 mm por píxel y se recorta a 192×192; el caso más exigente del ACDC necesita 179 píxeles.
2. Segmentación. U-Net de 4 clases con `base_channels=32` (7,8 millones de parámetros). Pérdida de entropía cruzada + Dice, AdamW con tasa 1e-3 y coseno, precisión mixta, 100 épocas fijas. Se entrenan 5 modelos con validación cruzada por paciente.
3. Inferencia. Cada paciente de training lo segmenta el modelo que no lo vio; los de test, el promedio de los 5. Siempre se promedian las 8 simetrías del cuadrado (TTA). La predicción vuelve a la resolución original del paciente antes de medir.
4. Biomarcadores. Son 16 por paciente: volúmenes del VI y del VD, FE de ambos, masa, valores por superficie corporal, razones VD/VI y masa/VI, grosor de pared y su engrosamiento de FD a FS.
5. Clasificador. Se comparan regresión logística y bosque aleatorio con validación cruzada en training, y gana el mejor al redondear a 3 decimales; en empate, la logística. Ganó la logística. Se entrena con los biomarcadores de las máscaras predichas por modelos que no vieron a esos pacientes.
6. Módulo de seguridad. Un paciente se marca si sus biomarcadores cambian mucho entre las 8 vistas (sobre el percentil 90 de training) o si el clasificador duda entre dos grupos (margen bajo el percentil 10 de training).

### Cómo se evita inflar los resultados

- Las divisiones son por paciente, porque los cortes de un mismo corazón son casi iguales.
- La segmentación usa épocas fijas: el pliegue retenido nunca elige la época, porque sus predicciones entrenan después al clasificador.
- El clasificador y los umbrales de alerta se eligen solo con training.
- Los pesos de la etapa 1 no se reutilizan en la etapa 2, porque los datos comparten origen.

Hubo contactos menores con test, ninguno cambió una decisión. El tamaño del recorte se midió primero con los 150 pacientes, pero solo con training da lo mismo (179 píxeles). Las medianas por grupo incluyen test, y son solo descriptivas. La exactitud del oráculo en test se vio antes de implementar el módulo de seguridad, cuyos umbrales salen solo de training.

## Limitaciones

- Con 50 pacientes de test, cada error mueve la exactitud 2 puntos. Empatar con la cota superior no prueba que segmentar automáticamente no cueste nada: en validación cruzada la diferencia es de 1,8 puntos.
- Todo el ACDC viene de un solo centro. Falta validar con otros equipos, por ejemplo con M&Ms.
- La FE del VD es la medida más débil.
- Infarto y dilatada se confunden: ambos tienen el VI grande y una FE baja.
- El umbral de alertas se calibró con un modelo por paciente y en test se aplica a un promedio de 5, lo que probablemente explica que marque 12 % en test y 20 % en training. Ajustarlo ahora sería decidir mirando test.
- La relación entre la dispersión de la FE y su error real es débil (Spearman 0,23 a 0,37). La alerta ayuda a priorizar qué revisar y no garantiza nada.
- En los datos de la etapa 1, 16 imágenes consecutivas aparecen en val y en test con máscaras distintas, y parecen un sujeto anotado dos veces. Sin ellas, el IoU de test pasa de 0,7470 a 0,7469. Train no comparte imágenes con val ni con test.
- Es un proyecto académico, sin validación clínica.

## Instalación

Requiere [uv](https://docs.astral.sh/uv/). PyTorch se instala con CUDA 12.6; sin GPU, comenta el bloque `[tool.uv.sources]` de `pyproject.toml`.

```bash
uv sync
```

## Uso

Etapa 1:

```bash
uv run python scripts/train.py data/cardiac_data --epochs 25 --img-size 256 --base-channels 16
uv run python scripts/visualize_predictions.py data/cardiac_data --split test --orden peores --individuales
```

Etapa 2, en orden:

```bash
uv run python -m miocardio.acdc_data
for k in 0 1 2 3 4; do uv run python scripts/train_acdc.py --fold $k; done
uv run python scripts/inferir_acdc.py
uv run python scripts/diagnostico.py
uv run python scripts/seguridad.py
uv run python scripts/figuras_acdc.py
```

En una RTX 4050, la etapa 1 tarda unos 20 minutos y cada pliegue de la etapa 2 unos 18. Todo lo que se genera queda en `outputs/`, fuera de git.

Notebooks: `notebooks/EDA_miocardio.ipynb` explora los datos de la etapa 1 y `notebooks/resultados_acdc.ipynb` recorre los resultados de la etapa 2 sin reentrenar. En VS Code elige el kernel `.venv`. Sus salidas traen imágenes de pacientes, así que antes de un commit usa "Clear All Outputs". Para verlos como página web:

```bash
uv run jupyter nbconvert --to html --execute notebooks/resultados_acdc.ipynb --output-dir outputs/notebooks
```

## Pruebas

```bash
uv run pytest -q
```

Vigilan errores que no lanzan excepción pero dan resultados equivocados, por ejemplo una máscara transformada distinto que su imagen, etiquetas mal interpretadas, volúmenes o grosores mal medidos y datos de test filtrados en la selección del clasificador. Las que necesitan datos se saltan si no están en disco.

## Estructura

```
src/miocardio/              código que importan los scripts
  cardiac_dataset.py        datos de la etapa 1
  acdc_data.py              datos de la etapa 2
  model.py                  U-Net y checkpoints
  losses.py                 pérdidas y métricas
  biomarcadores.py          volúmenes, FE, masa y grosor de pared
scripts/                    lo que se ejecuta
  train.py                  entrenamiento de la etapa 1
  visualize_predictions.py  figuras de la etapa 1
  train_acdc.py             entrenamiento por pliegue
  inferir_acdc.py           segmentación con TTA y biomarcadores por paciente
  diagnostico.py            clasificador de los 5 grupos
  seguridad.py              alertas por paciente
  figuras_acdc.py           figuras de la etapa 2
notebooks/                  exploración y resultados
tests/                      pruebas
data/                       datos, fuera de git
outputs/                    resultados, fuera de git
```

Los comandos se ejecutan desde la raíz del repositorio.

## Referencias

- Ronneberger, Fischer y Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
- Bernard et al. *Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?* IEEE TMI 37(11), 2018. doi:10.1109/TMI.2018.2837502
- Andreopoulos y Tsotsos. *Efficient and generalizable statistical models of shape and appearance for analysis of cardiac MRI*. Medical Image Analysis, 2008.
