# CLAUDE.md

Segmentacion binaria del miocardio del ventriculo izquierdo en MRI cardiaca de eje corto (ACDC + York) con una U-Net 2D. Python 3.13, PyTorch en CPU, uv. Windows 11.

Los datos (ACDC y York) son publicos pero tienen sus propios terminos de uso: el repo no versiona datos, checkpoints ni imagenes de pacientes (ver "Datos").

## Antes de terminar cualquier cambio

```bash
uv run pytest -q                    # todo (~25 s); tiene que pasar entero
uv run pytest -q -m "not slow"      # version rapida mientras se itera
```

`tests/test_miocardio.py` vigila los errores **silenciosos**: los que entrenan sin excepcion pero dan resultados equivocados (mascara aumentada distinto que su imagen, IoU agregado por pixel, scheduler mal orientado, evaluacion en modo train...). Si un test falla, el error esta en el codigo, no en el test: no relajar tolerancias ni borrar asserts para que pase. Si un cambio es intencional y rompe un invariante, decirlo explicitamente y actualizar el test y su docstring.

Los tests `datos` se saltan si `cardiac_data/` o `outputs/` no estan. `test_checkpoint_reproduce_su_iou_de_validacion` es la regresion de todo el pipeline de inferencia: con el checkpoint actual el val_iou tiene que salir 0.7230514777 exacto.

## Datos

- `cardiac_data/{train,val,test}_{imgs,annot}.npy`, fuera de git. Formato en `cardiac_data/data_description.txt`: (N, 256, 256) uint8; mascara 0 = fondo, 1 = miocardio, 2 = cavidad del LV.
- Se segmenta solo la clase 1. La cavidad (2) es negativo.
- Tamanos: train 4564, val 502, test 1240. El miocardio es ~1.2 % de los pixeles y no hay imagenes sin miocardio.
- **En esta maquina solo esta `val`**: faltan `train_*` y `test_*`. Sin ellos no se puede reentrenar ni evaluar en test.
- Normalizacion: /255 y despues estandarizacion con media y std **de train** (mean=0.13197279, std=0.20662321). Para evaluar sin train en disco: `load_splits(dir, stats={"mean": ck["mean"], "std": ck["std"]}, splits=("test",))`.
- Nunca commitear `.npy`, `.pt`, figuras con imagenes de pacientes ni salidas de notebook que las muestren.

## Codigo

- `cardiac_dataset.py`: carga, binarizacion, normalizacion, `CardiacDataset` (aumentacion: flips + rot90, sin interpolacion) y `get_dataloaders`.
- `model.py`: `UNet` (devuelve **logits**, sin sigmoid), `save_checkpoint` / `load_checkpoint`.
- `losses.py`: `iou_score` y `BCEDiceLoss`, ambos reciben logits; los dos promedian **por imagen**.
- `train.py`: `build_optimizer` (Adam + ReduceLROnPlateau con `mode="max"` sobre el IoU de val) y `run_epoch`.
- `visualize_predictions.py`: figura de predicciones sobre test.
- `EDA_miocardio.ipynb`: exploracion y curvas de entrenamiento.

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
