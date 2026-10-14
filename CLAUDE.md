# CLAUDE.md

Proyecto en dos etapas sobre MRI cardiaca de eje corto, siempre con una U-Net **2D** (regla del enunciado del curso: una sola imagen, sin cortes vecinos ni secuencia temporal):

1. **Etapa 1** (el enunciado del curso): segmentacion binaria del miocardio con los `.npy` preprocesados (ACDC + York). Maximizar el IoU promedio.
2. **Etapa 2** (extension): ACDC original -> segmentacion multiclase -> biomarcadores -> diagnostico en 5 grupos, mas un modulo de seguridad que marca los casos dudosos. El plan y el protocolo estan en `docs/plan-acdc.md`. **Sin app web**: el usuario la descarto.

Python 3.13, uv, PyTorch CUDA (cu126) en una RTX 4050 de 6 GB. Windows 11.

Los datos (ACDC y York) son publicos pero tienen sus propios terminos de uso: el repo no versiona datos, checkpoints ni imagenes de pacientes (ver "Datos").

## Antes de terminar cualquier cambio

```bash
uv run pytest -q                    # todo (~1 min); tiene que pasar entero
uv run pytest -q -m "not slow"      # version rapida mientras se itera
```

`tests/` vigila los errores **silenciosos** (etapa 1: `test_miocardio.py`; etapa 2: `test_acdc.py`, `test_biomarcadores.py`, `test_diagnostico.py`): los que entrenan sin excepcion pero dan resultados equivocados (mascara aumentada distinto que su imagen, IoU agregado por pixel, scheduler mal orientado, evaluacion en modo train...). Si un test falla, el error esta en el codigo, no en el test: no relajar tolerancias ni borrar asserts para que pase. Si un cambio es intencional y rompe un invariante, decirlo explicitamente y actualizar el test y su docstring.

Los tests `datos` se saltan si `cardiac_data/` o `outputs/` no estan. `test_checkpoint_reproduce_su_iou_de_validacion` es la regresion de todo el pipeline de inferencia: con el checkpoint actual el val_iou tiene que salir 0.7230514777 exacto.

## Datos de la etapa 1

- `cardiac_data/{train,val,test}_{imgs,annot}.npy`, fuera de git. Formato en `cardiac_data/data_description.txt`: (N, 256, 256) uint8; mascara 0 = fondo, 1 = miocardio, 2 = cavidad del LV.
- Se segmenta solo la clase 1. La cavidad (2) es negativo.
- Tamanos: train 4564, val 502, test 1240. El miocardio es ~1.2 % de los pixeles y no hay imagenes sin miocardio.
- **En esta maquina solo esta `val`**: faltan `train_*` y `test_*`. Sin ellos no se puede reentrenar ni evaluar en test.
- Normalizacion: /255 y despues estandarizacion con media y std **de train** (mean=0.13197279, std=0.20662321). Para evaluar sin train en disco: `load_splits(dir, stats={"mean": ck["mean"], "std": ck["std"]}, splits=("test",))`.
- Nunca commitear `.npy`, `.pt`, figuras con imagenes de pacientes ni salidas de notebook que las muestren.

## Datos de la etapa 2 (ACDC original)

- `data/ACDC/database/{training,testing}/patientNNN/`, fuera de git: 100 + 50 pacientes, 20 + 10 por grupo (NOR, MINF, DCM, HCM, RV), NIfTI de FD y FS con anotacion e `Info.cfg` (grupo, altura, peso). Licencia CC BY-NC-SA 4.0: citar Bernard et al. 2018.
- **Etiquetas del ACDC: 0 fondo, 1 VD, 2 miocardio, 3 VI.** Son distintas de la etapa 1, donde 1 es miocardio.
- Preprocesamiento (`acdc_data.py`): z-score por volumen tras recortar a percentiles 1-99, remuestreo a 1.5 mm (mascara con `nearest-exact`), recorte central de 192 px. La cache va en `data/cache_acdc/` y se reconstruye sola.
- Protocolo:
  - Los 50 de `testing` no se usan para ninguna decision.
  - 5 pliegues por paciente, estratificados por grupo, con epocas fijas y sin elegir epoca con el pliegue retenido.
  - El clasificador se entrena con biomarcadores **fuera de pliegue**.
  - La eleccion del clasificador y los umbrales de alerta se fijan con training, antes de mirar test.
- Oraculo (biomarcadores de las mascaras reales): 94 % en test, 93.9 % en CV. Los errores son MINF <-> DCM.
- Resultado de referencia (`outputs/acdc/`, 5 pliegues x 100 epocas):
  - Test: Dice miocardio 0.904/0.917 (FD/FS), VI 0.969/0.931, VD 0.941/0.883.
  - FE del VI: sesgo 0.1 y DE 3.5 puntos.
  - Diagnostico: pipeline 94 % (CV 92.1 %).
  - Seguridad: 6/50 marcados, 2 de 3 errores atrapados, 97.7 % en los no marcados.
  - Cualquier cambio de pipeline se compara contra estos numeros.

## Codigo

- `cardiac_dataset.py`: carga, binarizacion, normalizacion, `CardiacDataset` (aumentacion: flips + rot90, sin interpolacion) y `get_dataloaders`.
- `model.py`: `UNet` (devuelve **logits**, sin sigmoid), `save_checkpoint` / `load_checkpoint`.
- `losses.py`: `iou_score` y `BCEDiceLoss`, ambos reciben logits; los dos promedian **por imagen**.
- `train.py`: `build_optimizer` (Adam + ReduceLROnPlateau con `mode="max"` sobre el IoU de val) y `run_epoch`.
- `visualize_predictions.py`: figura de predicciones sobre test.
- `EDA_miocardio.ipynb`: exploracion y curvas de entrenamiento.
- Etapa 2:
  - `acdc_data.py`: lectura, preprocesamiento reversible (`preparar_volumen` / `a_nativo`), cache, pliegues y Dataset.
  - `train_acdc.py`: U-Net de 4 clases con CE + Dice multiclase (Dice por batch, suavizado 1).
  - `inferir_acdc.py`: TTA de 8 vistas, grilla nativa, biomarcadores e incertidumbre -> `outputs/acdc/pacientes.csv`.
  - `biomarcadores.py`: volumenes, FE, masa y grosor de pared en mm con el espaciado real.
  - `diagnostico.py`: pipeline contra oraculo.
  - `seguridad.py`: alertas.
  - `figuras_acdc.py`: figuras.

## Resultado de referencia

Todo resultado va a `outputs/` (fuera de git): checkpoints, `history.json`, `test_results.json` y figuras. Ningun script escribe en la raiz.

`outputs/best_model.pt`: img_size=128, base_channels=16, 25 epocas, mejor val en la epoca 19.
val IoU 0.7231, test IoU 0.7479. El IoU esta medido a **128x128** (mascaras reducidas con nearest), no a la resolucion nativa de 256.
Errores tipicos vistos en `--orden peores` de val: el modelo marca tambien la pared del ventriculo derecho (idx 279-281, cortes consecutivos de un mismo sujeto) y falla en cortes apicales, donde la cavidad casi desaparece y el miocardio real es un disco lleno (idx 489, 492).

`outputs/best_model_sin_stats.bak.pt` es el original antes de agregarle mean/std; se puede borrar.

## Comandos

```bash
uv sync                                                          # entorno (pyproject + uv.lock)
uv run python train.py cardiac_data --epochs 25 --img-size 128 --base-channels 16
uv run python visualize_predictions.py cardiac_data --split val --orden peores --individuales
uv run python model.py                                           # chequeo de shapes
uv run python losses.py                                          # chequeo de la metrica

# etapa 2 (en orden)
uv run python acdc_data.py                                       # cache + resumen
for k in 0 1 2 3 4; do uv run python train_acdc.py --fold $k; done   # ~18 min por pliegue en GPU
uv run python inferir_acdc.py
uv run python diagnostico.py
uv run python seguridad.py
uv run python figuras_acdc.py
```

## Convenciones

- Codigo, comentarios y docstrings en espanol **sin tildes ni enie** (ASCII). Los comentarios explican por que, no que.
- Los modulos viven en la raiz y se importan por nombre (`from model import UNet`); pytest los encuentra por `pythonpath = ["."]`.
- Los flags nuevos llevan defaults que reproducen la corrida de referencia.
- Los tests nuevos se agregan solo para fallas silenciosas, con un docstring que diga cual es el sintoma y por que no se nota.
- Antes de afirmar un numero (IoU, tamanos, porcentajes) verificarlo contra los datos o los JSON de `outputs/`; no suponerlo.

## Git

- Conventional commits en espanol sin tildes: `feat(train): ...`, `fix(dataset): ...`, `test: ...`, `docs: ...`.
- **Sin linea `Co-Authored-By`** ni pie "Generated with Claude Code" en commits ni en PRs.
- Revisar `git status` antes de cada commit: ningun `.npy`, `.pt` ni nada de `outputs/`.
