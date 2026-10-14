"""
Entrenamiento de la U-Net para segmentacion binaria del miocardio.

Uso:
    python train.py <data_dir> [opciones]

Ejemplo (rapido, para probar que todo corre en CPU):
    python train.py cardiac_data --epochs 3 --img-size 128 --base-channels 16

Ejemplo (entrenamiento completo, mejor calidad, mas lento):
    python train.py cardiac_data --epochs 40 --img-size 256 --base-channels 32

El script:
  1. Carga y preprocesa los datos con cardiac_dataset.py
  2. Entrena la U-Net minimizando BCEDiceLoss
  3. En cada epoca evalua IoU en el set de validacion
  4. Guarda el mejor checkpoint (segun IoU de val) en <out_dir>/best_model.pt
  5. Al final, carga el mejor checkpoint y reporta el IoU en el set de test
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cardiac_dataset import load_splits, get_dataloaders
from model import UNet, count_parameters, load_checkpoint, save_checkpoint
from losses import BCEDiceLoss, iou_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Entrena U-Net para segmentar miocardio")
    p.add_argument("data_dir", type=str, help="Carpeta con train/val/test_imgs.npy y _annot.npy")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--img-size", type=int, default=128,
                    help="Redimensiona las imagenes a img_size x img_size. "
                         "Usa 256 para el tamano original (mas lento en CPU).")
    p.add_argument("--base-channels", type=int, default=16,
                    help="Canales base de la U-Net. 32 = arquitectura clasica (mas "
                         "capacidad, mas lento). 16 = version liviana, buena para CPU.")
    p.add_argument("--out-dir", type=str, default="outputs")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=0)
    return p.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_optimizer(model, lr: float):
    """Adam + ReduceLROnPlateau sobre el IoU de validacion.

    mode="max" porque el IoU se maximiza: con "min" el lr se reduciria justo
    cada vez que el modelo mejora, y el entrenamiento solo se estancaria.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=3
    )
    return optimizer, scheduler


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Corre una epoca de train (si optimizer no es None) o de eval.
    Devuelve (loss_promedio, iou_promedio) sobre el dataset completo."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss, total_iou, n_samples = 0.0, 0.0, 0
    context = torch.enable_grad() if is_train else torch.no_grad()

    with context:
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            bs = imgs.size(0)

            if is_train:
                optimizer.zero_grad()

            logits = model(imgs)
            loss = loss_fn(logits, masks)

            if is_train:
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * bs
            total_iou += iou_score(logits, masks).item() * bs
            n_samples += bs

    return total_loss / n_samples, total_iou / n_samples


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando device: {device}")
    if device.type == "cpu":
        print("Nota: entrenando en CPU. Si esta muy lento, baja --img-size (ej. 96) "
              "y/o --base-channels (ej. 8), o reduce --epochs para probar primero.")

    print("Cargando datos...")
    data = load_splits(args.data_dir)
    train_loader, val_loader, test_loader = get_dataloaders(
        data, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size,
    )
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
          f"test={len(test_loader.dataset)} imagenes | img_size={args.img_size}")

    model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    print(f"Parametros entrenables: {count_parameters(model):,}")

    optimizer, scheduler = build_optimizer(model, args.lr)
    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    best_val_iou = -1.0
    history = []  # lista de dicts, uno por epoca, para graficar curvas despues
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_iou = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, loss_fn, device, optimizer=None)
        scheduler.step(val_iou)
        dt = time.time() - t0

        marker = ""
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_checkpoint(best_path, model, args.base_channels, args.img_size,
                            val_iou, data["_stats"])
            marker = " <- mejor hasta ahora, guardado"

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_iou": train_iou,
            "val_loss": val_loss, "val_iou": val_iou, "lr": optimizer.param_groups[0]["lr"],
            "seconds": dt,
        })
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)  # se re-escribe cada epoca: sirve aunque se corte el entrenamiento

        print(f"[{epoch:03d}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_iou={train_iou:.4f} | "
              f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} | "
              f"{dt:.1f}s{marker}")

    print(f"\nMejor IoU en val: {best_val_iou:.4f} (checkpoint: {best_path})")

    print("\nEvaluando el mejor checkpoint en el set de TEST...")
    model, _ = load_checkpoint(best_path, map_location=device)
    test_loss, test_iou = run_epoch(model, test_loader, loss_fn, device, optimizer=None)
    print(f"Test: loss={test_loss:.4f} IoU={test_iou:.4f}")

    with open(out_dir / "test_results.json", "w") as f:
        json.dump({"test_loss": test_loss, "test_iou": test_iou,
                    "best_val_iou": best_val_iou, "img_size": args.img_size,
                    "base_channels": args.base_channels, "epochs": args.epochs}, f, indent=2)


if __name__ == "__main__":
    main()