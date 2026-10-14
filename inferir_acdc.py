"""
Inferencia del ACDC: segmentacion con TTA -> grilla nativa -> biomarcadores e incertidumbre.

Uso:
    python inferir_acdc.py              # despues de entrenar los 5 pliegues

Por paciente:
  - training: lo segmenta el modelo del pliegue que NO lo vio (fuera de pliegue).
  - testing: lo segmentan los 5 modelos; la prediccion es el promedio.

TTA: las 8 simetrias del cuadrado (flips + rotaciones de 90). La prediccion
final promedia las probabilidades de todas las vistas; la dispersion de los
biomarcadores ENTRE vistas es la incertidumbre del modulo de seguridad. En test
se calcula por modelo y se promedia entre modelos, para que tenga la misma
escala que en training (un modelo por paciente).

Salidas en outputs/acdc/:
    pacientes.csv                       biomarcadores predichos, reales, Dice e incertidumbre
    mascaras/<pid>.npz                  mascaras predichas en la grilla nativa (FD y FS)
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

import acdc_data as A
from biomarcadores import NOMBRES, biomarcadores
from losses import dice_y_iou
from train_acdc import cargar_modelo

# clases en las que se reporta la segmentacion
CLASES = {"vd": A.VD, "mio": A.MIOCARDIO, "vi": A.VI}


def vistas_d4(x: torch.Tensor):
    """Las 8 simetrias del cuadrado sobre los dos ultimos ejes: (k, flip, tensor)."""
    for flip in (False, True):
        xf = torch.flip(x, dims=(-1,)) if flip else x
        for k in range(4):
            yield k, flip, torch.rot90(xf, k, dims=(-2, -1))


def deshacer_vista(y: torch.Tensor, k: int, flip: bool) -> torch.Tensor:
    """Inversa de una vista: primero se deshace la rotacion, despues el flip.

    En el orden contrario las vistas con flip quedan rotadas 180 grados respecto
    de la imagen, y el promedio de TTA mezcla predicciones desalineadas sin error.
    """
    y = torch.rot90(y, -k, dims=(-2, -1))
    return torch.flip(y, dims=(-1,)) if flip else y


@torch.no_grad()
def probabilidades_tta(model, cortes: np.ndarray, device, tta: bool = True) -> np.ndarray:
    """(Z, H, W) -> (V, Z, C, H, W): probabilidades de cada vista, ya desrotadas."""
    x = torch.from_numpy(cortes).unsqueeze(1).to(device)
    vistas = vistas_d4(x) if tta else [(0, False, x)]
    salida = []
    for k, flip, xv in vistas:
        p = torch.softmax(model(xv).float(), dim=1)
        salida.append(deshacer_vista(p, k, flip).cpu().numpy())
    return np.stack(salida)


def mascara_nativa(probs: np.ndarray, geo: A.Geometria) -> np.ndarray:
    """(Z, C, TAMANO, TAMANO) -> etiquetas (Z, H, W) en la grilla nativa."""
    return A.a_nativo(probs, geo).argmax(1).astype(np.uint8)


def procesar_paciente(p: A.Paciente, modelos, device) -> dict:
    fila = {"pid": p.pid, "split": p.split, "grupo": p.grupo, "sc": p.superficie_corporal}
    finales, gts, geos, por_vista = {}, {}, {}, {}
    for fase in A.FASES:
        img, gt, esp = A.cargar_fase(p, fase)
        cortes, _, geo = A.preparar_volumen(img, None, esp)
        probs = [probabilidades_tta(m, cortes, device) for m in modelos]  # [M](V, Z, C, h, w)
        finales[fase] = mascara_nativa(np.mean([pm.mean(0) for pm in probs], axis=0), geo)
        por_vista[fase] = [[mascara_nativa(pv, geo) for pv in pm] for pm in probs]  # [M][V]
        gts[fase], geos[fase] = np.moveaxis(gt, -1, 0), geo

    esp = geos["ED"].espaciado
    pred = biomarcadores(finales["ED"], finales["ES"], esp, p.superficie_corporal)
    real = biomarcadores(gts["ED"], gts["ES"], esp, p.superficie_corporal)
    fila.update({f"pred_{n}": pred[n] for n in NOMBRES})
    fila.update({f"real_{n}": real[n] for n in NOMBRES})

    # incertidumbre: desviacion entre las 8 vistas de cada modelo, promediada entre modelos
    stds = {n: [] for n in NOMBRES}
    for m in range(len(modelos)):
        vistas = [biomarcadores(fd, fs, esp, p.superficie_corporal)
                  for fd, fs in zip(por_vista["ED"][m], por_vista["ES"][m])]
        for n in NOMBRES:
            stds[n].append(np.nanstd([v[n] for v in vistas]))
    fila.update({f"std_{n}": float(np.mean(s)) for n, s in stds.items()})

    for fase in A.FASES:
        for nombre, c in CLASES.items():
            dice, iou = dice_y_iou(finales[fase], gts[fase], c)
            fila[f"dice_{nombre}_{fase}"], fila[f"iou_{nombre}_{fase}"] = dice, iou
        # metrica del curso: IoU 2D del miocardio, promedio por corte
        fila[f"iou2d_mio_{fase}"] = float(np.mean(
            [dice_y_iou(pc, gc, A.MIOCARDIO)[1] for pc, gc in zip(finales[fase], gts[fase])]))
    return fila, finales


def main() -> None:
    ap = argparse.ArgumentParser(description="Inferencia ACDC con TTA y biomarcadores")
    ap.add_argument("--modelos", type=str, default="outputs/acdc")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--out", type=str, default="outputs/acdc")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelos = [cargar_modelo(Path(args.modelos) / f"fold{f}" / "model.pt", device)[0]
               for f in range(args.k)]
    pliegue = A.pliegues_por_paciente(A.listar_pacientes(split="training"), k=args.k)
    out = Path(args.out)
    (out / "mascaras").mkdir(parents=True, exist_ok=True)

    filas = []
    for p in A.listar_pacientes():
        if p.split == "training":
            f = pliegue[p.pid]
            fila, finales = procesar_paciente(p, [modelos[f]], device)
            fila["fold"] = f
        else:
            fila, finales = procesar_paciente(p, modelos, device)
            fila["fold"] = -1
        np.savez_compressed(out / "mascaras" / f"{p.pid}.npz", ED=finales["ED"], ES=finales["ES"])
        filas.append(fila)
        print(f"{p.pid} {p.split:8s} {p.grupo:4s} dice mio={fila['dice_mio_ED']:.3f}/{fila['dice_mio_ES']:.3f} "
              f"FE VI pred={fila['pred_fe_vi']:5.1f} real={fila['real_fe_vi']:5.1f} "
              f"(+-{fila['std_fe_vi']:.1f})")

    columnas = list(filas[0].keys())
    with open(out / "pacientes.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=columnas)
        w.writeheader()
        w.writerows(filas)
    print(f"Guardado {out / 'pacientes.csv'} ({len(filas)} pacientes)")


if __name__ == "__main__":
    main()
