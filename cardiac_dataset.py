"""
Pipeline de datos para segmentacion de miocardio (ACDC + York).

Responsabilidades de este modulo:
  1. Cargar los .npy (imgs / annot) de train / val / test.
  2. Binarizar las mascaras: clase 1 (miocardio) -> 1, clases 0 y 2
     (fondo y endocardio del LV) -> 0.
  3. Normalizar las imagenes (uint8 [0,255] -> float32 [0,1], luego
     estandarizacion con media/std calculadas SOLO en train).
  4. Exponer un torch.utils.data.Dataset con aumentacion de datos
     opcional (flips / rotaciones de a 90) para el set de train.

Uso rapido:
    from cardiac_dataset import load_splits, CardiacDataset, get_dataloaders

    data = load_splits("cardiac_data")
    train_loader, val_loader, test_loader = get_dataloaders(data, batch_size=16)

    # solo test, con los stats guardados en el checkpoint (no necesita train)
    stats = {"mean": ckpt["mean"], "std": ckpt["std"]}
    data = load_splits("cardiac_data", stats=stats, splits=("test",))
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

# torch es opcional: el pipeline de numpy funciona sin el.
# Se importa "lazy" adentro de las clases/funciones que lo necesitan
# para que este modulo tambien sirva en un entorno sin torch instalado.


MYOCARDIUM_LABEL = 1  # clase de interes en <split>_annot.npy
SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- #
# 1) Carga + binarizacion + normalizacion (numpy puro)
# --------------------------------------------------------------------------- #

@dataclass
class SplitData:
    images: np.ndarray  # float32, (N, H, W), normalizado
    masks: np.ndarray   # uint8,   (N, H, W), valores en {0, 1} (1 = miocardio)


def _binarize_mask(annot: np.ndarray) -> np.ndarray:
    """Colapsa la mascara de 3 clases {0: fondo, 1: miocardio, 2: endocardio}
    a una mascara binaria {0: fondo/endocardio, 1: miocardio}."""
    return (annot == MYOCARDIUM_LABEL).astype(np.uint8)


def _normalize_images(imgs_uint8: np.ndarray, mean: float, std: float) -> np.ndarray:
    """uint8 [0,255] -> float32 [0,1] -> estandarizado con (mean, std) de train."""
    imgs = imgs_uint8.astype(np.float32) / 255.0
    return (imgs - mean) / (std + 1e-8)


def load_splits(
    data_dir: str,
    stats: Optional[Dict[str, float]] = None,
    splits: Sequence[str] = SPLITS,
) -> Dict[str, SplitData]:
    """Carga y preprocesa los splits pedidos (por defecto train/val/test).

    La media y el std usados para estandarizar se calculan UNICAMENTE
    sobre el split de train (buena practica: val/test no deben influir
    en el preprocesamiento) y se reutilizan para todos los splits.

    stats: {"mean", "std"} ya calculados en train, por ejemplo los que guarda
    el checkpoint. Si se dan, no se recalcula nada y train no hace falta: es lo
    que permite evaluar test con exactamente la normalizacion del entrenamiento.
    """
    if stats is None and "train" not in splits:
        raise ValueError("sin `stats` hace falta el split de train para calcularlos")

    raw = {}
    for split in splits:
        imgs = np.load(os.path.join(data_dir, f"{split}_imgs.npy"))
        annot = np.load(os.path.join(data_dir, f"{split}_annot.npy"))
        assert imgs.shape == annot.shape, f"{split}: shapes no calzan {imgs.shape} vs {annot.shape}"
        raw[split] = (imgs, annot)

    if stats is None:
        train_imgs_uint8, _ = raw["train"]
        # float de python y no np.float32: asi el checkpoint carga con weights_only=True
        mean = float(train_imgs_uint8.astype(np.float32).mean() / 255.0)
        std = float(train_imgs_uint8.astype(np.float32).std() / 255.0)
    else:
        mean, std = float(stats["mean"]), float(stats["std"])

    data: Dict[str, SplitData] = {}
    for split, (imgs_uint8, annot) in raw.items():
        norm_imgs = _normalize_images(imgs_uint8, mean, std)
        bin_masks = _binarize_mask(annot)
        data[split] = SplitData(images=norm_imgs, masks=bin_masks)

    data["_stats"] = {"mean": mean, "std": std}  # type: ignore[assignment]
    return data


def class_balance(masks: np.ndarray) -> Tuple[float, float]:
    """Fraccion de pixeles de fondo vs miocardio (util para saber si conviene
    ponderar la loss / usar Dice o Focal loss por el desbalance de clases)."""
    total = masks.size
    fg = masks.sum()
    return (total - fg) / total, fg / total


# --------------------------------------------------------------------------- #
# 2) Dataset de PyTorch (con augmentation opcional para train)
# --------------------------------------------------------------------------- #

class CardiacDataset:
    """torch.utils.data.Dataset para el problema de segmentacion binaria.

    Devuelve tensores:
        image: (1, H, W) float32
        mask:  (1, H, W) float32  (0/1)

    Augmentation (solo si augment=True, pensado para el split de train):
      - flip horizontal / vertical aleatorio
      - rotacion aleatoria en multiplos de 90 grados
    Estas transformaciones no distorsionan la imagen (no hay interpolacion),
    por lo que la mascara se transforma exactamente igual que la imagen.
    """

    def __init__(self, split_data: SplitData, augment: bool = False, img_size: int | None = None):
        import torch  # import lazy
        self._torch = torch
        self.images = split_data.images
        self.masks = split_data.masks
        self.augment = augment
        self.img_size = img_size  # si se da, redimensiona (ej. 128) para entrenar mas rapido en CPU

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int):
        torch = self._torch
        img = self.images[idx].copy()
        mask = self.masks[idx].copy()

        if self.augment:
            if np.random.rand() < 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
            if np.random.rand() < 0.5:
                img = np.flipud(img).copy()
                mask = np.flipud(mask).copy()
            k = np.random.randint(0, 4)
            if k:
                img = np.rot90(img, k).copy()
                mask = np.rot90(mask, k).copy()

        img_t = torch.from_numpy(img).float().unsqueeze(0)   # (1,H,W)
        mask_t = torch.from_numpy(mask).float().unsqueeze(0)  # (1,H,W)

        if self.img_size is not None and self.img_size != img_t.shape[-1]:
            import torch.nn.functional as F
            img_t = F.interpolate(
                img_t.unsqueeze(0), size=(self.img_size, self.img_size),
                mode="bilinear", align_corners=False,
            ).squeeze(0)
            # nearest para la mascara: no queremos valores intermedios entre 0 y 1
            mask_t = F.interpolate(
                mask_t.unsqueeze(0), size=(self.img_size, self.img_size), mode="nearest",
            ).squeeze(0)

        return img_t, mask_t


def get_dataloaders(
    data: Dict[str, SplitData],
    batch_size: int = 16,
    num_workers: int = 0,
    img_size: int | None = None,
):
    """Arma los 3 DataLoaders (train con augmentation + shuffle, val/test sin).

    img_size: si se especifica (ej. 128), redimensiona las imagenes/mascaras
    a ese tamano cuadrado. Util para entrenar mucho mas rapido en CPU;
    None mantiene el tamano original (256x256).
    """
    from torch.utils.data import DataLoader

    train_ds = CardiacDataset(data["train"], augment=True, img_size=img_size)
    val_ds = CardiacDataset(data["val"], augment=False, img_size=img_size)
    test_ds = CardiacDataset(data["test"], augment=False, img_size=img_size)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    import sys

    data_dir = sys.argv[1] if len(sys.argv) > 1 else "cardiac_data"
    data = load_splits(data_dir)
    for split in ("train", "val", "test"):
        sd = data[split]  # type: ignore[index]
        bg, fg = class_balance(sd.masks)
        print(f"{split:5s} -> imgs {sd.images.shape} {sd.images.dtype} | "
              f"masks {sd.masks.shape} {sd.masks.dtype} | "
              f"bg={bg:.3%} fg(miocardio)={fg:.3%}")
    print("stats (mean/std de train, usados para estandarizar todo):", data["_stats"])