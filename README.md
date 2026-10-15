# Segmentacion de miocardio y diagnostico cardiaco con U-Net 2D

Proyecto en dos etapas sobre resonancia magnetica cardiaca de eje corto. Las dos usan una U-Net **2D**: cada corte se segmenta por separado, sin cortes vecinos ni secuencia temporal.

1. **Segmentacion del miocardio** (enunciado del curso): mascara binaria del miocardio sobre una version preprocesada de ACDC + York, maximizando el IoU promedio.
2. **De la segmentacion al diagnostico** (extension): sobre el ACDC original se segmentan el ventriculo derecho, el miocardio y el ventriculo izquierdo. Con esas mascaras se miden los biomarcadores que usa un cardiologo y se clasifica a cada paciente en 5 grupos. Un modulo de seguridad marca los casos dudosos para revision.

## Resultados

### Etapa 1: miocardio (datos del curso)

| | IoU |
|---|---|
| Validacion (mejor epoca, 19 de 25) | 0.7231 |
| Test | 0.7479 |

Imagenes a 128x128, `base_channels=16`, BCE + Dice, CPU. Es el IoU promedio por imagen, medido a 128x128.

### Etapa 2: ACDC original, test oficial de 50 pacientes

Ninguna decision se tomo mirando test (ver [docs/plan-acdc.md](docs/plan-acdc.md)): validacion cruzada por paciente, biomarcadores fuera de pliegue para entrenar el clasificador, y umbrales de alerta fijados en training.

| Segmentacion (Dice 3D, FD / FS) | |
|---|---|
| Ventriculo izquierdo | 0.969 / 0.931 |
| Miocardio | 0.904 / 0.917 |
| Ventriculo derecho | 0.941 / 0.883 |
| IoU 2D del miocardio (metrica del curso) | 0.802 / 0.827 |

| Biomarcador (prediccion contra mascara real) | sesgo | DE | r |
|---|---|---|---|
| FE del VI (puntos %) | 0.1 | 3.5 | 0.988 |
| FE del VD (puntos %) | 2.1 | 7.3 | 0.870 |
| Masa del miocardio (g) | 2.9 | 8.9 | 0.985 |

| Diagnostico (NOR, MINF, DCM, HCM, RV) | exactitud |
|---|---|
| Pipeline: mascaras predichas | **94 %** (47/50) |
| Oraculo: mascaras de los expertos | 94 % (47/50) |
| Validacion cruzada en training: pipeline / oraculo | 92.1 % / 93.9 % |

Modulo de seguridad en test:
- Marco 6 de 50 pacientes y atrapo 2 de los 3 errores.
- Los 44 no marcados aciertan el 97.7 %.

En training (fuera de pliegue) marco 20 de 100 y atrapo 6 de 7 errores; los no marcados aciertan el 98.8 %. Con 50 pacientes de test, cada error mueve la exactitud en 2 puntos.

## Datos

### Etapa 1

Las bases de datos originales son publicas ([ACDC](https://acdc.creatis.insa-lyon.fr/) y [York](http://www.cse.yorku.ca/~mridataset/)), pero cada una tiene sus propios terminos de uso, asi que este repositorio **no incluye datos** ni imagenes derivadas. Se usa una version preprocesada (normalizada a 256x256 con zero-padding) que hay que copiar en `cardiac_data/`:

```
cardiac_data/
  train_imgs.npy   train_annot.npy
  val_imgs.npy     val_annot.npy
  test_imgs.npy    test_annot.npy
```

Cada `*_imgs.npy` es `(N, 256, 256)` uint8 y cada `*_annot.npy` es una mascara con 0 = fondo, 1 = miocardio, 2 = cavidad del ventriculo izquierdo. Los splits no comparten sujetos. Detalle en `cardiac_data/data_description.txt`.

### Etapa 2

ACDC original (150 pacientes, NIfTI), descargado del [portal Human Heart Project de CREATIS](https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb) en `data/ACDC/database/`. Licencia CC BY-NC-SA 4.0. Cualquier uso debe citar:

> O. Bernard et al., "Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?", IEEE TMI 37(11), 2018. doi:10.1109/TMI.2018.2837502

## Instalacion

Requiere [uv](https://docs.astral.sh/uv/). PyTorch se instala con CUDA 12.6; sin GPU, comentar el bloque `[tool.uv.sources]` de `pyproject.toml`.

```bash
uv sync
```

## Uso

```bash
# entrenamiento (la configuracion del resultado de arriba)
uv run python train.py cardiac_data --epochs 25 --img-size 128 --base-channels 16

# imagenes de las segmentaciones: verde = acierto, rojo = falso positivo, azul = falso negativo
uv run python visualize_predictions.py cardiac_data --split test --n 8 --individuales
uv run python visualize_predictions.py cardiac_data --split val --orden peores    # casos mas dificiles
```

`--orden` elige las imagenes al azar (`aleatorio`) o las de menor o mayor IoU del split (`peores`, `mejores`). `--individuales` guarda ademas una figura por imagen.

Todos los resultados van a `outputs/`, que no se versiona: `train.py` guarda el mejor modelo segun el IoU de validacion (`best_model.pt`, que incluye la media y el std de normalizacion), el historial por epoca (`history.json`) y el resultado en test (`test_results.json`); `visualize_predictions.py` guarda la grilla `predicciones_<split>_<orden>.png` y, con `--individuales`, una figura por imagen en `segmentaciones_<split>_<orden>/`.

Etapa 2, en orden (las salidas van a `outputs/acdc/`):

```bash
uv run python acdc_data.py                 # cache de cortes preprocesados
uv run python train_acdc.py --fold 0       # repetir para los pliegues 0 a 4 (~18 min cada uno en una RTX 4050)
uv run python inferir_acdc.py              # segmentacion con TTA, biomarcadores e incertidumbre
uv run python diagnostico.py               # clasificador: pipeline y oraculo
uv run python seguridad.py                 # alertas
uv run python figuras_acdc.py              # figuras en outputs/acdc/figuras/
```

## Notebooks

- `EDA_miocardio.ipynb`: exploracion de los datos de la etapa 1, curvas de entrenamiento y predicciones. Funciona con los splits que haya en `cardiac_data/`.
- `resultados_acdc.ipynb`: recorrido completo por los resultados de la etapa 2 (no entrena nada). `mostrar_paciente(pid)` dibuja todos los cortes de un paciente con su segmentacion.

Se versionan sin salidas, porque las salidas traen imagenes de pacientes. Para ejecutarlos y verlos como pagina web en `outputs/notebooks/`:

```bash
uv run jupyter nbconvert --to html --execute resultados_acdc.ipynb --output-dir outputs/notebooks
```

En VS Code: abrir el notebook, elegir el kernel `.venv` y "Run All". Antes de un commit, "Clear All Outputs".

## Tests

```bash
uv run pytest -q                                   # todo
uv run pytest -q -m "not slow"                     # sin los que entrenan
uv run pytest -q -m "not slow and not datos"       # sin datos ni checkpoint
```

Los tests verifican invariantes que fallan en silencio, es decir, los errores que entrenan sin excepcion pero dan resultados equivocados. Por ejemplo:
- aumentacion distinta en imagen y mascara, o mascara corrida medio pixel al remuestrear;
- convencion de etiquetas del ACDC;
- volumenes y grosor de pared contra geometrias conocidas;
- inversion de las vistas de TTA;
- fuga de test hacia la seleccion del clasificador;
- reproduccion exacta del IoU del checkpoint de la etapa 1.

Los que necesitan datos se saltan si no estan en disco.

## Estructura

```
cardiac_dataset.py         carga, binarizacion, normalizacion, Dataset y DataLoaders
model.py                   U-Net y guardado/carga de checkpoints
losses.py                  IoU y perdida BCE + Dice
train.py                   entrenamiento y evaluacion en test
visualize_predictions.py   figura de predicciones
EDA_miocardio.ipynb        exploracion de datos y curvas de entrenamiento
acdc_data.py               etapa 2: lectura y preprocesamiento reversible del ACDC
train_acdc.py              etapa 2: U-Net de 4 clases por pliegue
inferir_acdc.py            etapa 2: TTA, grilla nativa, biomarcadores e incertidumbre
biomarcadores.py           etapa 2: volumenes, FE, masa y grosor de pared
diagnostico.py             etapa 2: clasificador de los 5 grupos
seguridad.py               etapa 2: alertas por paciente
figuras_acdc.py            etapa 2: figuras
resultados_acdc.ipynb      etapa 2: recorrido por los resultados
docs/plan-acdc.md          etapa 2: diseno y protocolo de evaluacion
tests/                     invariantes del pipeline
outputs/                   resultados (fuera de git)
```

## Referencias

- Ronneberger, Fischer y Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
- Bernard et al. *Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?* IEEE TMI 2018 (ACDC).
- Andreopoulos y Tsotsos. *Efficient and generalizable statistical models of shape and appearance for analysis of cardiac MRI*. Medical Image Analysis 2008 (York).
