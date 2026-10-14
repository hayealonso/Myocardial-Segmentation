"""
Visualiza las segmentaciones del modelo entrenado sobre val o test.

Uso:
    python visualize_predictions.py <data_dir> [--split test] [--n 8]
                                    [--orden aleatorio|peores|mejores] [--individuales]

Para cada imagen elegida muestra: imagen original, mascara real, mascara
predicha, y el mapa de errores (verde = acierto, rojo = falso positivo,
azul = falso negativo).

Salidas en --out-dir (por defecto outputs/, fuera de git):
    predicciones_<split>_<orden>.png       grilla con las n imagenes
    segmentaciones_<split>_<orden>/<split>_<idx>.png   una figura por imagen (--individuales)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from cardiac_dataset import load_splits, CardiacDataset
from model import load_checkpoint
from losses import iou_score

# RGBA: verde = acierto, rojo = falso positivo, azul = falso negativo
COLOR_TP = (0.0, 1.0, 0.0, 0.6)
COLOR_FP = (1.0, 0.0, 0.0, 0.6)
COLOR_FN = (0.0, 0.4, 1.0, 0.6)
COLOR_REAL = (0.0, 0.8, 0.0, 0.6)
COLOR_PRED = (1.0, 0.1, 0.1, 0.6)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualiza segmentaciones sobre val o test")
    p.add_argument("data_dir", type=str)
    p.add_argument("--checkpoint", type=str, default="outputs/best_model.pt")
    p.add_argument("--split", choices=("val", "test"), default="test")
    p.add_argument("--n", type=int, default=8, help="Numero de imagenes a mostrar")
    p.add_argument("--orden", choices=("aleatorio", "peores", "mejores"), default="aleatorio",
                   help="peores/mejores elige las n imagenes de menor/mayor IoU del split")
    p.add_argument("--individuales", action="store_true",
                   help="Ademas de la grilla, guarda una figura por imagen")
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def capa_de_mascara(mask: np.ndarray, color) -> np.ndarray:
    """Capa RGBA con `color` donde mask == 1 y transparente en el resto.

    Color fijo en vez de un colormap: una mascara tiene un solo valor (1), y
    imshow estira el colormap a ese rango degenerado y la pinta casi blanca.
    """
    capa = np.zeros((*mask.shape, 4), dtype=np.float32)
    capa[mask == 1] = color
    return capa


def mapa_de_errores(mask: np.ndarray, pred: np.ndarray) -> np.ndarray:
    """Capa RGBA (H, W, 4) para superponer sobre la imagen.

    Los verdaderos negativos quedan transparentes (alpha 0): con una capa RGB
    opaca, el fondo negro oscurece toda la imagen y no solo los errores.
    """
    capa = np.zeros((*mask.shape, 4), dtype=np.float32)
    capa[(mask == 1) & (pred == 1)] = COLOR_TP
    capa[(mask == 0) & (pred == 1)] = COLOR_FP
    capa[(mask == 1) & (pred == 0)] = COLOR_FN
    return capa


@torch.no_grad()
def predecir(model, ds: CardiacDataset, idx: int, device):
    """Devuelve (imagen, mascara real, mascara predicha, IoU) de una muestra."""
    img_t, mask_t = ds[idx]
    logits = model(img_t.unsqueeze(0).to(device))
    pred = (torch.sigmoid(logits) > 0.5).float()
    iou = iou_score(logits, mask_t.unsqueeze(0).to(device)).item()
    return (img_t[0].numpy(), mask_t[0].numpy(), pred[0, 0].cpu().numpy(), iou)


@torch.no_grad()
def iou_por_imagen(model, ds: CardiacDataset, device, batch_size: int = 32) -> np.ndarray:
    """IoU de cada imagen del split (para elegir los peores/mejores casos)."""
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)
    ious = []
    for imgs, masks in loader:
        logits = model(imgs.to(device))
        for lg, m in zip(logits, masks.to(device)):
            ious.append(iou_score(lg.unsqueeze(0), m.unsqueeze(0)).item())
    return np.array(ious)


def dibujar(axes, idx: int, img, mask, pred, iou: float) -> None:
    """Dibuja una muestra en 4 ejes: original, real, prediccion, errores."""
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title(f"idx {idx}", fontsize=9)

    axes[1].imshow(img, cmap="gray")
    axes[1].imshow(capa_de_mascara(mask, COLOR_REAL))
    axes[1].set_title("real", fontsize=9)

    axes[2].imshow(img, cmap="gray")
    axes[2].imshow(capa_de_mascara(pred, COLOR_PRED))
    axes[2].set_title(f"prediccion\nIoU={iou:.2f}", fontsize=9)

    axes[3].imshow(img, cmap="gray")
    axes[3].imshow(mapa_de_errores(mask, pred))
    axes[3].set_title("verde=ok rojo=FP azul=FN", fontsize=8)

    for ax in axes:
        ax.axis("off")


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    img_size = checkpoint["img_size"]
    print(f"Checkpoint cargado: base_channels={checkpoint['base_channels']} "
          f"img_size={img_size} val_iou={checkpoint['val_iou']:.4f}")

    if "mean" in checkpoint:
        # misma normalizacion que en el entrenamiento, sin necesitar train en disco
        stats = {"mean": checkpoint["mean"], "std": checkpoint["std"]}
        data = load_splits(args.data_dir, stats=stats, splits=(args.split,))
    else:
        # checkpoint antiguo: se recalculan los stats desde train
        data = load_splits(args.data_dir)
    ds = CardiacDataset(data[args.split], augment=False, img_size=img_size)
    n = min(args.n, len(ds))

    if args.orden == "aleatorio":
        idxs = np.random.default_rng(args.seed).choice(len(ds), size=n, replace=False)
    else:
        todos = iou_por_imagen(model, ds, device)
        print(f"IoU promedio de todo {args.split}: {todos.mean():.4f} ({len(ds)} imagenes)")
        orden = np.argsort(todos)
        idxs = orden[:n] if args.orden == "peores" else orden[::-1][:n]

    muestras = [(int(idx), *predecir(model, ds, int(idx), device)) for idx in idxs]

    fig, axes = plt.subplots(4, n, figsize=(2.6 * n, 10.5), squeeze=False)
    for col, muestra in enumerate(muestras):
        dibujar(axes[:, col], *muestra)
    fig.tight_layout()
    grilla = out_dir / f"predicciones_{args.split}_{args.orden}.png"
    fig.savefig(grilla, dpi=110)
    plt.close(fig)
    print(f"IoU promedio de las {n} imagenes mostradas: {np.mean([m[-1] for m in muestras]):.4f}")
    print(f"Grilla guardada en: {grilla}")

    if args.individuales:
        carpeta = out_dir / f"segmentaciones_{args.split}_{args.orden}"
        carpeta.mkdir(exist_ok=True)
        for muestra in muestras:
            fig, axes = plt.subplots(1, 4, figsize=(10.4, 3.0))
            dibujar(axes, *muestra)
            fig.tight_layout()
            fig.savefig(carpeta / f"{args.split}_{muestra[0]:04d}.png", dpi=110)
            plt.close(fig)
        print(f"{n} figuras individuales en: {carpeta}")


if __name__ == "__main__":
    main()
