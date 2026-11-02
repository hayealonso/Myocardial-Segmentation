"""
Entrena la U-Net de 4 clases (fondo, VD, miocardio, VI) sobre un pliegue del ACDC.

    python train_acdc.py --fold 0

Validacion cruzada de 5 pliegues por paciente sobre los 100 de training. Las epocas
son fijas y se guarda el ultimo modelo: el pliegue retenido solo se monitorea, porque
sus predicciones entrenan despues al clasificador.

Salidas en <out-dir>/fold<k>/: model.pt e history.json. Para una prueba rapida usa otro
--out-dir: si no, sobrescribe el modelo entrenado.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from acdc_data import (
    ESPACIADO_OBJETIVO, N_CLASES, NOMBRES_CLASES, TAMANO,
    CortesACDC, cargar_cache, listar_pacientes, pliegues_por_paciente,
)
from losses import CEDiceLoss
from model import UNet, count_parameters, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="U-Net de 4 clases sobre un pliegue del ACDC")
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--val-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="outputs/acdc")
    parser.add_argument("--no-amp", action="store_true", help="Desactiva la precision mixta")
    return parser.parse_args()


def guardar_modelo(path, model, args, epoca):
    torch.save({
        "model_state": model.state_dict(), "base_channels": int(args.base_channels),
        "n_clases": N_CLASES, "fold": int(args.fold), "k": int(args.k), "seed": int(args.seed),
        "epocas": int(epoca), "espaciado": ESPACIADO_OBJETIVO, "tamano": TAMANO,
    }, path)


def cargar_modelo(path, map_location="cpu"):
    """Devuelve (modelo en modo eval, checkpoint). Rechaza modelos de otra grilla."""
    model, checkpoint = load_checkpoint(path, map_location=map_location)
    if checkpoint["espaciado"] != ESPACIADO_OBJETIVO or checkpoint["tamano"] != TAMANO:
        raise ValueError(f"{path}: entrenado con {checkpoint['espaciado']} mm y {checkpoint['tamano']} px, "
                         f"la grilla actual es {ESPACIADO_OBJETIVO} mm y {TAMANO} px")
    return model, checkpoint


@torch.no_grad()
def dice_por_volumen(model, imgs, labs, claves, device, batch_size=32):
    """Dice 3D por clase (sin fondo), promedio sobre volumenes, en la grilla de 1.5 mm.

    Solo sirve para monitorear: la evaluacion final se hace en la grilla nativa.
    """
    model.eval()
    interseccion, suma = {}, {}
    for inicio in range(0, len(imgs), batch_size):
        x = torch.from_numpy(imgs[inicio:inicio + batch_size]).unsqueeze(1).to(device)
        preds = model(x).argmax(1).cpu().numpy()
        for i, pred in enumerate(preds):
            clave = claves[inicio + i]
            lab = labs[inicio + i]
            if clave not in interseccion:
                interseccion[clave] = np.zeros(N_CLASES)
                suma[clave] = np.zeros(N_CLASES)
            for clase in range(1, N_CLASES):
                interseccion[clave][clase] += ((pred == clase) & (lab == clase)).sum()
                suma[clave][clase] += (pred == clase).sum() + (lab == clase).sum()
    dices = [2 * interseccion[clave][1:] / np.maximum(suma[clave][1:], 1) for clave in interseccion]
    return np.mean(dices, axis=0)


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    usar_amp = device.type == "cuda" and not args.no_amp
    # la entrada siempre mide lo mismo, asi que puede elegir la convolucion mas rapida
    torch.backends.cudnn.benchmark = True

    out_dir = Path(args.out_dir) / f"fold{args.fold}"
    out_dir.mkdir(parents=True, exist_ok=True)

    pliegues = pliegues_por_paciente(listar_pacientes(split="training"), k=args.k)
    cache = cargar_cache("training")
    en_val = np.array([pliegues[pid] == args.fold for pid in cache["pids"]])
    claves_val = [f"{pid}_{fase}" for pid, fase in zip(cache["pids"][en_val], cache["fases"][en_val])]
    imgs_val, labs_val = cache["imgs"][en_val], cache["labs"][en_val]
    print(f"device={device} amp={usar_amp} fold={args.fold}: "
          f"train {(~en_val).sum()} cortes, val {en_val.sum()} cortes")

    train_ds = CortesACDC(cache["imgs"][~en_val], cache["labs"][~en_val], augment=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

    model = UNet(in_channels=1, out_channels=N_CLASES, base_channels=args.base_channels).to(device)
    print(f"parametros: {count_parameters(model):,}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)
    loss_fn = CEDiceLoss()

    history = []
    for epoca in range(1, args.epochs + 1):
        model.train()
        start = time.time()
        total_loss, n_samples = 0.0, 0
        for imgs, labs in train_loader:
            imgs, labs = imgs.to(device), labs.to(device)
            optimizer.zero_grad()
            with torch.autocast(device_type=device.type, enabled=usar_amp):
                loss = loss_fn(model(imgs), labs)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item() * len(imgs)
            n_samples += len(imgs)
        scheduler.step()

        fila = {"epoca": epoca, "train_loss": total_loss / n_samples,
                "lr": optimizer.param_groups[0]["lr"], "segundos": time.time() - start}
        linea = f"[{epoca:03d}/{args.epochs}] loss={fila['train_loss']:.4f} {fila['segundos']:.1f}s"
        if epoca % args.val_every == 0 or epoca == args.epochs:
            dices = dice_por_volumen(model, imgs_val, labs_val, claves_val, device)
            for nombre, dice in zip(NOMBRES_CLASES[1:], dices):
                fila[f"val_dice_{nombre}"] = float(dice)
                linea += f" {nombre}={dice:.3f}"
        history.append(fila)
        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)
        print(linea)

    guardar_modelo(out_dir / "model.pt", model, args, args.epochs)
    print(f"Modelo: {out_dir / 'model.pt'}")


if __name__ == "__main__":
    main()
