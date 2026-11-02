"""
Entrena la U-Net de la etapa 1 para segmentar el miocardio.

    python train.py cardiac_data --epochs 25 --img-size 256 --base-channels 16

Guarda en --out-dir el mejor modelo segun el IoU de val y al final lo evalua en
test. Para una prueba rapida usa otro --out-dir: si no, sobrescribe outputs/best_model.pt.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cardiac_dataset import get_dataloaders, load_splits
from losses import BCEDiceLoss, iou_score
from model import UNet, count_parameters, load_checkpoint, save_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Entrena la U-Net para segmentar miocardio")
    parser.add_argument("data_dir", help="Carpeta con los .npy de train, val y test")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--img-size", type=int, default=256, help="256 es el tamano original")
    parser.add_argument("--base-channels", type=int, default=16)
    parser.add_argument("--out-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def build_optimizer(model, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # mode="max": el scheduler mira el IoU de val, que se maximiza
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
    return optimizer, scheduler


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Una epoca de train (con optimizer) o de evaluacion (sin optimizer). Devuelve (loss, iou)."""
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss, total_iou, n_samples = 0.0, 0.0, 0
    with torch.set_grad_enabled(is_train):
        for imgs, masks in loader:
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = loss_fn(logits, masks)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # se pondera por batch: el ultimo puede venir incompleto
            batch_size = imgs.size(0)
            total_loss += loss.item() * batch_size
            total_iou += iou_score(logits, masks).item() * batch_size
            n_samples += batch_size

    return total_loss / n_samples, total_iou / n_samples


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / "best_model.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    data = load_splits(args.data_dir)
    train_loader, val_loader, test_loader = get_dataloaders(
        data, batch_size=args.batch_size, num_workers=args.num_workers, img_size=args.img_size,
    )
    print(f"train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")

    model = UNet(in_channels=1, out_channels=1, base_channels=args.base_channels).to(device)
    print(f"parametros: {count_parameters(model):,}")
    optimizer, scheduler = build_optimizer(model, args.lr)
    loss_fn = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)

    best_val_iou = -1.0
    history = []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_loss, train_iou = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_iou = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_iou)
        seconds = time.time() - start

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            save_checkpoint(best_path, model, args.base_channels, args.img_size, val_iou, data["_stats"])

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_iou": train_iou,
            "val_loss": val_loss, "val_iou": val_iou, "lr": optimizer.param_groups[0]["lr"],
            "seconds": seconds,
        })
        # se reescribe cada epoca para no perder las curvas si se corta el entrenamiento
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        print(f"[{epoch:03d}/{args.epochs}] train_loss={train_loss:.4f} train_iou={train_iou:.4f} "
              f"val_loss={val_loss:.4f} val_iou={val_iou:.4f} {seconds:.1f}s")

    print(f"Mejor IoU en val: {best_val_iou:.4f}")
    model, _ = load_checkpoint(best_path, map_location=device)
    test_loss, test_iou = run_epoch(model, test_loader, loss_fn, device)
    print(f"Test: loss={test_loss:.4f} IoU={test_iou:.4f}")

    results = {"test_loss": test_loss, "test_iou": test_iou, "best_val_iou": best_val_iou,
               "img_size": args.img_size, "base_channels": args.base_channels, "epochs": args.epochs}
    with open(out_dir / "test_results.json", "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
