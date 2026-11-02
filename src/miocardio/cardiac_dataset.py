"""
Datos de la etapa 1: carga los .npy de ACDC + York y arma los DataLoaders.

Las mascaras traen 0 fondo, 1 miocardio y 2 cavidad del VI; solo el miocardio
queda como 1. Las imagenes se normalizan con la media y el std de train.

    data=load_splits("data/cardiac_data")
    train_loader, val_loader, test_loader=get_dataloaders(data, batch_size=16)
"""

import os
import sys
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

MYOCARDIUM_LABEL=1
SPLITS=("train", "val", "test")


@dataclass
class SplitData:
    images: np.ndarray  #float32 (N, H, W), normalizado
    masks: np.ndarray   #uint8 (N, H, W), 1=miocardio


def _binarize_mask(annot):
    return (annot==MYOCARDIUM_LABEL).astype(np.uint8)


def _normalize_images(imgs_uint8, mean, std):
    imgs=imgs_uint8.astype(np.float32)/255.0
    return (imgs-mean)/(std+1e-8)


def load_splits(data_dir, stats=None, splits=SPLITS):
    """Carga los splits pedidos. Devuelve {split: SplitData, "_stats": {"mean", "std"}}.

    stats: media y std de train ya calculados (los guarda el checkpoint). Con
    ellos se puede evaluar test sin tener train en disco.
    """
    if stats is None and "train" not in splits:
        raise ValueError("sin `stats` hace falta el split de train para calcularlos")

    raw={}
    for split in splits:
        imgs=np.load(os.path.join(data_dir, f"{split}_imgs.npy"))
        annot=np.load(os.path.join(data_dir, f"{split}_annot.npy"))
        assert imgs.shape==annot.shape, f"{split}: shapes no calzan {imgs.shape} vs {annot.shape}"
        raw[split]=(imgs, annot)

    if stats is None:
        train_imgs=raw["train"][0].astype(np.float32)
        #float de python: np.float32 no carga con weights_only=True
        mean=float(train_imgs.mean()/255.0)
        std=float(train_imgs.std()/255.0)
    else:
        mean, std=float(stats["mean"]), float(stats["std"])

    data={}
    for split, (imgs, annot) in raw.items():
        data[split]=SplitData(images=_normalize_images(imgs, mean, std), masks=_binarize_mask(annot))
    data["_stats"]={"mean": mean, "std": std}
    return data


def class_balance(masks):
    """Fraccion de pixeles de fondo y de miocardio."""
    total=masks.size
    miocardio=masks.sum()
    return (total-miocardio)/total, miocardio/total


class CardiacDataset(Dataset):
    """Devuelve image (1, H, W) y mask (1, H, W), ambos float32.

    Con augment=True aplica flips y rotaciones de 90 grados, iguales en imagen y mascara.
    """

    def __init__(self, split_data, augment=False, img_size=None):
        self.images=split_data.images
        self.masks=split_data.masks
        self.augment=augment
        self.img_size=img_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img=self.images[idx].copy()
        mask=self.masks[idx].copy()

        if self.augment:
            if np.random.rand()<0.5:
                img=np.fliplr(img).copy()
                mask=np.fliplr(mask).copy()
            if np.random.rand()<0.5:
                img=np.flipud(img).copy()
                mask=np.flipud(mask).copy()
            k=np.random.randint(0, 4)
            if k:
                img=np.rot90(img, k).copy()
                mask=np.rot90(mask, k).copy()

        img_t=torch.from_numpy(img).float().unsqueeze(0)
        mask_t=torch.from_numpy(mask).float().unsqueeze(0)

        if self.img_size is not None and self.img_size!=img_t.shape[-1]:
            size=(self.img_size, self.img_size)
            img_t=F.interpolate(img_t.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
            #bilinear + umbral, igual que la imagen. Al reducir a la mitad, nearest y
            #nearest-exact corren la mascara medio pixel, cada uno hacia un lado.
            mask_t=F.interpolate(mask_t.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)
            mask_t=(mask_t>=0.5).float()

        return img_t, mask_t


def get_dataloaders(data, batch_size=16, num_workers=0, img_size=None):
    """Train con aumentacion y shuffle; val y test sin ninguno de los dos."""
    train_ds=CardiacDataset(data["train"], augment=True, img_size=img_size)
    val_ds=CardiacDataset(data["val"], augment=False, img_size=img_size)
    test_ds=CardiacDataset(data["test"], augment=False, img_size=img_size)

    train_loader=DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader=DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader=DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__=="__main__":
    data_dir=sys.argv[1] if len(sys.argv)>1 else "data/cardiac_data"
    data=load_splits(data_dir)
    for split in SPLITS:
        fondo, miocardio=class_balance(data[split].masks)
        print(f"{split:5s}: {data[split].images.shape} fondo={fondo:.3%} miocardio={miocardio:.3%}")
    print("media y std de train:", data["_stats"])
