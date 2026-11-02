"""
Dibuja las segmentaciones de la etapa 1 sobre val o test.

    python scripts/visualize_predictions.py data/cardiac_data --split val --orden peores --individuales

Por imagen: original, mascara real, prediccion y errores (verde = acierto,
rojo = falso positivo, azul = falso negativo). Guarda en --out-dir la grilla
predicciones_<split>_<orden>.png y, con --individuales, una figura por imagen.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from miocardio.cardiac_dataset import CardiacDataset, load_splits
from miocardio.losses import iou_score
from miocardio.model import load_checkpoint

#RGBA
COLOR_TP=(0.0, 1.0, 0.0, 0.6)
COLOR_FP=(1.0, 0.0, 0.0, 0.6)
COLOR_FN=(0.0, 0.4, 1.0, 0.6)
COLOR_REAL=(0.0, 0.8, 0.0, 0.6)
COLOR_PRED=(1.0, 0.1, 0.1, 0.6)


def parse_args():
    parser=argparse.ArgumentParser(description="Dibuja segmentaciones sobre val o test")
    parser.add_argument("data_dir")
    parser.add_argument("--checkpoint", default="outputs/best_model.pt")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--n", type=int, default=8, help="Numero de imagenes")
    parser.add_argument("--orden", choices=("aleatorio", "peores", "mejores"), default="aleatorio")
    parser.add_argument("--individuales", action="store_true", help="Guarda tambien una figura por imagen")
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def capa_de_mascara(mask, color):
    """Capa RGBA: color fijo donde mask == 1, transparente en el resto."""
    capa=np.zeros((*mask.shape, 4), dtype=np.float32)
    capa[mask==1]=color
    return capa


def mapa_de_errores(mask, pred):
    """Capa RGBA con aciertos y errores. Los verdaderos negativos quedan transparentes."""
    capa=np.zeros((*mask.shape, 4), dtype=np.float32)
    capa[(mask==1)&(pred==1)]=COLOR_TP
    capa[(mask==0)&(pred==1)]=COLOR_FP
    capa[(mask==1)&(pred==0)]=COLOR_FN
    return capa


@torch.no_grad()
def predecir(model, dataset, idx, device):
    """Devuelve (imagen, mascara, prediccion, IoU) de una muestra."""
    img, mask=dataset[idx]
    logits=model(img.unsqueeze(0).to(device))
    pred=(torch.sigmoid(logits)>0.5).float()
    iou=iou_score(logits, mask.unsqueeze(0).to(device)).item()
    return img[0].numpy(), mask[0].numpy(), pred[0, 0].cpu().numpy(), iou


@torch.no_grad()
def iou_por_imagen(model, dataset, device, batch_size=32):
    ious=[]
    for imgs, masks in DataLoader(dataset, batch_size=batch_size, shuffle=False):
        logits=model(imgs.to(device))
        masks=masks.to(device)
        for i in range(len(imgs)):
            ious.append(iou_score(logits[i:i+1], masks[i:i+1]).item())
    return np.array(ious)


def dibujar(axes, idx, img, mask, pred, iou):
    """Dibuja una muestra en 4 ejes: original, real, prediccion y errores."""
    for ax in axes:
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    axes[0].set_title(f"idx {idx}", fontsize=9)
    axes[1].imshow(capa_de_mascara(mask, COLOR_REAL))
    axes[1].set_title("real", fontsize=9)
    axes[2].imshow(capa_de_mascara(pred, COLOR_PRED))
    axes[2].set_title(f"prediccion\nIoU={iou:.2f}", fontsize=9)
    axes[3].imshow(mapa_de_errores(mask, pred))
    axes[3].set_title("verde=ok rojo=FP azul=FN", fontsize=8)


def main():
    args=parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir=Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model, checkpoint=load_checkpoint(args.checkpoint, map_location=device)
    print(f"Checkpoint: img_size={checkpoint['img_size']} val_iou={checkpoint['val_iou']:.4f}")

    if "mean" in checkpoint:
        stats={"mean": checkpoint["mean"], "std": checkpoint["std"]}
        data=load_splits(args.data_dir, stats=stats, splits=(args.split,))
    else:
        #checkpoint antiguo, sin stats: se recalculan desde train
        data=load_splits(args.data_dir)
    dataset=CardiacDataset(data[args.split], augment=False, img_size=checkpoint["img_size"])
    n=min(args.n, len(dataset))

    if args.orden=="aleatorio":
        idxs=np.random.default_rng(args.seed).choice(len(dataset), size=n, replace=False)
    else:
        ious=iou_por_imagen(model, dataset, device)
        print(f"IoU promedio de {args.split}: {ious.mean():.4f} ({len(dataset)} imagenes)")
        orden=np.argsort(ious)
        idxs=orden[:n] if args.orden=="peores" else orden[::-1][:n]

    muestras=[]
    for idx in idxs:
        idx=int(idx)
        muestras.append((idx, *predecir(model, dataset, idx, device)))

    fig, axes=plt.subplots(4, n, figsize=(2.6*n, 10.5), squeeze=False)
    for col, muestra in enumerate(muestras):
        dibujar(axes[:, col], *muestra)
    fig.tight_layout()
    grilla=out_dir/f"predicciones_{args.split}_{args.orden}.png"
    fig.savefig(grilla, dpi=110)
    plt.close(fig)
    print(f"IoU promedio de las {n} imagenes: {np.mean([m[-1] for m in muestras]):.4f}")
    print(f"Grilla: {grilla}")

    if args.individuales:
        carpeta=out_dir/f"segmentaciones_{args.split}_{args.orden}"
        carpeta.mkdir(exist_ok=True)
        for muestra in muestras:
            fig, axes=plt.subplots(1, 4, figsize=(10.4, 3.0))
            dibujar(axes, *muestra)
            fig.tight_layout()
            fig.savefig(carpeta/f"{args.split}_{muestra[0]:04d}.png", dpi=110)
            plt.close(fig)
        print(f"Figuras individuales: {carpeta}")


if __name__=="__main__":
    main()
