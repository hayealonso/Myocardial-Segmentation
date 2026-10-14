"""
Entrena la U-Net multiclase (fondo, VD, miocardio, VI) sobre un pliegue del ACDC.

Uso:
    python train_acdc.py --fold 0            # un pliegue (de 5)
    python train_acdc.py --fold 0 --epochs 2 # prueba rapida

Validacion cruzada de 5 pliegues por paciente sobre los 100 de `training`
(ver docs/plan-acdc.md). Se entrena un numero FIJO de epocas y se guarda el
ultimo checkpoint: el pliegue retenido solo se monitorea, nunca se usa para
elegir la epoca, porque sus predicciones alimentan despues al clasificador.

Salidas en outputs/acdc/fold<k>/: model.pt e history.json.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from acdc_data import (
    N_CLASES, NOMBRES_CLASES, ESPACIADO_OBJETIVO, TAMANO,
    CortesACDC, cargar_cache, listar_pacientes, pliegues_por_paciente,
)
from losses import CEDiceLoss
from model import UNet, count_parameters


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="U-Net multiclase sobre un pliegue del ACDC")
    p.add_argument("--fold", type=int, required=True)
    p.add_argument("--k", type=int, default=5)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--base-channels", type=int, default=32)
    p.add_argument("--val-every", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", type=str, default="outputs/acdc")
    p.add_argument("--no-amp", action="store_true", help="Desactiva precision mixta en GPU")
    return p.parse_args()


def guardar_modelo(path: Path, model: UNet, args, epoca: int) -> None:
    torch.save({
        "model_state": model.state_dict(), "base_channels": int(args.base_channels),
        "n_clases": N_CLASES, "fold": int(args.fold), "k": int(args.k), "seed": int(args.seed),
        "epocas": int(epoca), "espaciado": ESPACIADO_OBJETIVO, "tamano": TAMANO,
    }, path)


def cargar_modelo(path, map_location="cpu"):
    """Devuelve (modelo en modo eval, dict del checkpoint)."""
    ck = torch.load(path, map_location=map_location, weights_only=True)
    model = UNet(in_channels=1, out_channels=ck["n_clases"], base_channels=ck["base_channels"])
    model.load_state_dict(ck["model_state"])
    model.to(map_location).eval()
    return model, ck


@torch.no_grad()
def dice_por_volumen(model, imgs, labs, claves, device, batch_size: int = 32):
    """Dice 3D por clase, acumulado por (paciente, fase) en la grilla de 1.5 mm.

    Es solo para monitorear: la evaluacion final se hace en la grilla nativa.
    """
    model.eval()
    inter = defaultdict(lambda: np.zeros(N_CLASES))
    suma = defaultdict(lambda: np.zeros(N_CLASES))
    for i in range(0, len(imgs), batch_size):
        x = torch.from_numpy(imgs[i:i + batch_size]).unsqueeze(1).to(device)
        pred = model(x).argmax(1).cpu().numpy()
        for j, (pr, gt) in enumerate(zip(pred, labs[i:i + batch_size])):
            clave = claves[i + j]
            for c in range(1, N_CLASES):
                inter[clave][c] += ((pr == c) & (gt == c)).sum()
                suma[clave][c] += (pr == c).sum() + (gt == c).sum()
    dices = np.array([2 * inter[k][1:] / np.maximum(suma[k][1:], 1) for k in inter])
    return dices.mean(axis=0)  # (C-1,) promedio sobre volumenes


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usar_amp = device.type == "cuda" and not args.no_amp
    torch.backends.cudnn.benchmark = True  # entrada de tamano fijo: elige el algoritmo mas rapido

    out_dir = Path(args.out_dir) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pliegue = pliegues_por_paciente(listar_pacientes(split="training"), k=args.k)
    d = cargar_cache("training")
    en_val = np.array([pliegue[pid] == args.fold for pid in d["pids"]])
    claves_val = [f"{p}_{f}" for p, f in zip(d["pids"][en_val], d["fases"][en_val])]
    imgs_val, labs_val = d["imgs"][en_val], d["labs"][en_val]
    print(f"device={device} amp={usar_amp} | fold {args.fold}: "
          f"train {int((~en_val).sum())} cortes / val {int(en_val.sum())} cortes "
          f"({len(set(d['pids'][en_val]))} pacientes)")

    loader = torch.utils.data.DataLoader(
        CortesACDC(d["imgs"][~en_val], d["labs"][~en_val], augment=True),
        batch_size=args.batch_size, shuffle=True, drop_last=True,
    )
    model = UNet(in_channels=1, out_channels=N_CLASES, base_channels=args.base_channels).to(device)
    print(f"Parametros entrenables: {count_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)
    loss_fn = CEDiceLoss()

    history = []
    for epoca in range(1, args.epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for x, y in loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=usar_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total += loss.item() * len(x); n += len(x)
        scheduler.step()

        fila = {"epoca": epoca, "train_loss": total / n, "lr": optimizer.param_groups[0]["lr"],
                "segundos": time.time() - t0}
        if epoca % args.val_every == 0 or epoca == args.epochs:
            dice = dice_por_volumen(model, imgs_val, labs_val, claves_val, device)
            fila.update({f"val_dice_{NOMBRES_CLASES[c]}": float(dice[c - 1]) for c in range(1, N_CLASES)})
        history.append(fila)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        extra = " ".join(f"{k[9:]}={v:.3f}" for k, v in fila.items() if k.startswith("val_dice_"))
        print(f"[{epoca:03d}/{args.epochs}] loss={fila['train_loss']:.4f} {fila['segundos']:.1f}s {extra}")

    guardar_modelo(out_dir / "model.pt", model, args, args.epochs)
    print(f"Checkpoint guardado en {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
