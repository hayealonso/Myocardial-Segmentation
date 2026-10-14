# Segmentacion de miocardio con U-Net

Segmentacion binaria del miocardio del ventriculo izquierdo en imagenes de resonancia magnetica cardiaca de eje corto, con una U-Net 2D entrenada sobre una version preprocesada de ACDC (100 sujetos) y York (33 sujetos).

| | IoU |
|---|---|
| Validacion (mejor epoca, 19 de 25) | 0.7231 |
| Test | 0.7479 |

Configuracion: imagenes reducidas a 128x128, `base_channels=16` (~1.9 M parametros), BCE + Dice, Adam con ReduceLROnPlateau, CPU. El IoU es el promedio por imagen, medido a 128x128.

## Datos

Las bases de datos originales son publicas ([ACDC](https://acdc.creatis.insa-lyon.fr/) y [York](http://www.cse.yorku.ca/~mridataset/)), pero cada una tiene sus propios terminos de uso, asi que este repositorio **no incluye datos** ni imagenes derivadas. Se usa una version preprocesada (normalizada a 256x256 con zero-padding) que hay que copiar en `cardiac_data/`:

```
cardiac_data/
  train_imgs.npy   train_annot.npy
  val_imgs.npy     val_annot.npy
  test_imgs.npy    test_annot.npy
```

Cada `*_imgs.npy` es `(N, 256, 256)` uint8 y cada `*_annot.npy` es una mascara con 0 = fondo, 1 = miocardio, 2 = cavidad del ventriculo izquierdo. Los splits no comparten sujetos. Detalle en `cardiac_data/data_description.txt`.

## Instalacion

Requiere [uv](https://docs.astral.sh/uv/).

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

## Tests

```bash
uv run pytest -q                                   # todo
uv run pytest -q -m "not slow"                     # sin los que entrenan
uv run pytest -q -m "not slow and not datos"       # sin datos ni checkpoint
```

Los tests verifican invariantes que fallan en silencio: binarizacion, normalizacion solo con train, aumentacion identica en imagen y mascara, metrica por imagen, evaluacion sin efectos sobre el modelo, orientacion del scheduler y reproduccion exacta del IoU de validacion del checkpoint. Los que necesitan datos se saltan si `cardiac_data/` no esta.

## Estructura

```
cardiac_dataset.py         carga, binarizacion, normalizacion, Dataset y DataLoaders
model.py                   U-Net y guardado/carga de checkpoints
losses.py                  IoU y perdida BCE + Dice
train.py                   entrenamiento y evaluacion en test
visualize_predictions.py   figura de predicciones
EDA_miocardio.ipynb        exploracion de datos y curvas de entrenamiento
tests/                     invariantes del pipeline
outputs/                   resultados (fuera de git)
```

## Referencias

- Ronneberger, Fischer y Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation*. MICCAI 2015.
- Bernard et al. *Deep Learning Techniques for Automatic MRI Cardiac Multi-structures Segmentation and Diagnosis: Is the Problem Solved?* IEEE TMI 2018 (ACDC).
- Andreopoulos y Tsotsos. *Efficient and generalizable statistical models of shape and appearance for analysis of cardiac MRI*. Medical Image Analysis 2008 (York).
