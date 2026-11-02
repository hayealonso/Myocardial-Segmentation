"""
Datos de la etapa 2: ACDC original (150 pacientes, NIfTI).

Etiquetas del ACDC: 0 fondo, 1 VD, 2 miocardio, 3 VI. En cardiac_data/*.npy el 1 es
miocardio, asi que no se pueden mezclar.

Cada volumen se normaliza, se remuestrea a 1.5 mm por pixel y se recorta a 192x192.
`a_nativo` deshace ese camino para medir volumenes con el espaciado real del paciente.

    python acdc_data.py  #construye la cache y resume los datos
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

ACDC_ROOT=Path("data/ACDC/database")
CACHE_DIR=Path("data/cache_acdc")

FONDO, VD, MIOCARDIO, VI=0, 1, 2, 3
N_CLASES=4
NOMBRES_CLASES=("fondo", "VD", "miocardio", "VI")
GRUPOS=("NOR", "MINF", "DCM", "HCM", "RV")
FASES=("ED", "ES")

ESPACIADO_OBJETIVO=1.5  #mm por pixel
TAMANO=192              #288 mm; el corazon mas descentrado del ACDC necesita 179 px


@dataclass
class Paciente:
    pid: str       #"patient001"
    split: str     #"training" o "testing", la division oficial del ACDC
    grupo: str
    altura: float  #cm
    peso: float    #kg
    ed: int        #numero de frame, como en el nombre del archivo
    es: int
    carpeta: Path

    def archivo(self, fase, gt=False):
        frame=self.ed if fase=="ED" else self.es
        sufijo="_gt" if gt else ""
        return self.carpeta/f"{self.pid}_frame{frame:02d}{sufijo}.nii.gz"

    @property
    def superficie_corporal(self):
        """m^2, formula de Mosteller."""
        return float(np.sqrt(self.altura*self.peso/3600.0))


def leer_info(path):
    info={}
    for linea in Path(path).read_text().splitlines():
        if ":" in linea:
            clave, valor=linea.split(":", 1)
            info[clave.strip()]=valor.strip()
    return info


def listar_pacientes(root=ACDC_ROOT, split=None):
    splits=(split,) if split else ("training", "testing")
    pacientes=[]
    for nombre_split in splits:
        for carpeta in sorted(Path(root, nombre_split).glob("patient*")):
            info=leer_info(carpeta/"Info.cfg")
            pacientes.append(Paciente(
                pid=carpeta.name, split=nombre_split, grupo=info["Group"],
                altura=float(info["Height"]), peso=float(info["Weight"]),
                ed=int(info["ED"]), es=int(info["ES"]), carpeta=carpeta,
            ))
    return pacientes


def cargar_fase(paciente, fase):
    """Devuelve (img float32 (H, W, Z), gt uint8 (H, W, Z), espaciado (sx, sy, sz) en mm)."""
    img=nib.load(paciente.archivo(fase))
    gt=nib.load(paciente.archivo(fase, gt=True))
    espaciado=tuple(float(z) for z in img.header.get_zooms()[:3])
    return np.asarray(img.dataobj, dtype=np.float32), np.asarray(gt.dataobj).astype(np.uint8), espaciado


def normalizar_volumen(vol):
    """Recorta a los percentiles 1 y 99 y aplica z-score con los valores del propio volumen.

    Cada equipo guarda las intensidades en otra escala, asi que no sirve una media global.
    """
    lo, hi=np.percentile(vol, (1, 99))
    vol=np.clip(vol, lo, hi)
    return ((vol-vol.mean())/(vol.std()+1e-8)).astype(np.float32)


def _tamano_remuestreado(shape_hw, espaciado_xy):
    alto=max(1, int(round(shape_hw[0]*espaciado_xy[0]/ESPACIADO_OBJETIVO)))
    ancho=max(1, int(round(shape_hw[1]*espaciado_xy[1]/ESPACIADO_OBJETIVO)))
    return alto, ancho


def remuestrear(cortes, tamano, etiqueta):
    """(Z, H, W) -> (Z, h, w). Bilinear en la imagen, nearest-exact en la mascara.

    Con `nearest` la mascara queda corrida medio pixel respecto de la imagen.
    """
    x=torch.from_numpy(cortes.copy()).float().unsqueeze(1)
    if etiqueta:
        salida=F.interpolate(x, size=tamano, mode="nearest-exact")
        return salida.squeeze(1).round().numpy().astype(np.uint8)
    salida=F.interpolate(x, size=tamano, mode="bilinear", align_corners=False)
    return salida.squeeze(1).numpy().astype(np.float32)


def _inicios_centrados(largo, tamano):
    """Inicio en cada eje y largo de la parte comun, con los dos ejes centrados."""
    comun=min(largo, tamano)
    return (largo-comun)//2, (tamano-comun)//2, comun


def centrar(arr, tamano, relleno=0.0):
    """Recorta o rellena los dos ultimos ejes a (tamano, tamano), centrado."""
    salida=np.full(arr.shape[:-2]+(tamano, tamano), relleno, dtype=arr.dtype)
    y_arr, y_sal, alto=_inicios_centrados(arr.shape[-2], tamano)
    x_arr, x_sal, ancho=_inicios_centrados(arr.shape[-1], tamano)
    salida[..., y_sal:y_sal+alto, x_sal:x_sal+ancho]=arr[..., y_arr:y_arr+alto, x_arr:x_arr+ancho]
    return salida


def descentrar(arr, shape_hw, relleno=0.0):
    """Inversa de `centrar`: lleva los dos ultimos ejes a shape_hw."""
    tamano=arr.shape[-1]
    salida=np.full(arr.shape[:-2]+tuple(shape_hw), relleno, dtype=arr.dtype)
    y_sal, y_arr, alto=_inicios_centrados(shape_hw[0], tamano)
    x_sal, x_arr, ancho=_inicios_centrados(shape_hw[1], tamano)
    salida[..., y_sal:y_sal+alto, x_sal:x_sal+ancho]=arr[..., y_arr:y_arr+alto, x_arr:x_arr+ancho]
    return salida


@dataclass
class Geometria:
    """Lo necesario para devolver una prediccion a la grilla nativa."""
    shape_hw: tuple         #tamano nativo del corte
    remuestreado_hw: tuple  #tamano a 1.5 mm, antes de recortar
    espaciado: tuple


def preparar_volumen(img, gt, espaciado):
    """(H, W, Z) nativo -> (Z, TAMANO, TAMANO), mascara si la hay, y su Geometria."""
    shape_hw=img.shape[:2]
    remuestreado_hw=_tamano_remuestreado(shape_hw, espaciado[:2])
    normalizado=normalizar_volumen(img)
    cortes=remuestrear(np.moveaxis(normalizado, -1, 0), remuestreado_hw, etiqueta=False)
    cortes=centrar(cortes, TAMANO, relleno=float(normalizado.min()))
    etiquetas=None
    if gt is not None:
        etiquetas=remuestrear(np.moveaxis(gt, -1, 0), remuestreado_hw, etiqueta=True)
        etiquetas=centrar(etiquetas, TAMANO)
    return cortes, etiquetas, Geometria(tuple(shape_hw), remuestreado_hw, tuple(espaciado))


def a_nativo(probs, geometria):
    """Probabilidades (Z, C, TAMANO, TAMANO) -> (Z, C, H, W) en la grilla nativa.

    Se interpolan probabilidades y el argmax se toma al final, para que los bordes no queden escalonados.
    """
    n_cortes, n_clases=probs.shape[:2]
    sin_recorte=descentrar(probs, geometria.remuestreado_hw)
    #lo que quedo fuera del recorte vuelve como fondo
    fuera=sin_recorte.sum(axis=1)==0
    sin_recorte[:, FONDO][fuera]=1.0
    x=torch.from_numpy(sin_recorte.reshape(n_cortes*n_clases, 1, *geometria.remuestreado_hw))
    salida=F.interpolate(x, size=geometria.shape_hw, mode="bilinear", align_corners=False)
    return salida.reshape(n_cortes, n_clases, *geometria.shape_hw).numpy()


def construir_cache(root=ACDC_ROOT, destino=CACHE_DIR):
    """Preprocesa los 150 pacientes y guarda sus cortes en un .npz por split."""
    destino=Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    for split in ("training", "testing"):
        imgs, labs, pids, fases, zs=[], [], [], [], []
        for paciente in listar_pacientes(root, split):
            for fase in FASES:
                img, gt, espaciado=cargar_fase(paciente, fase)
                cortes, etiquetas, _=preparar_volumen(img, gt, espaciado)
                imgs.append(cortes)
                labs.append(etiquetas)
                pids+=[paciente.pid]*len(cortes)
                fases+=[fase]*len(cortes)
                zs+=list(range(len(cortes)))
        np.savez(destino/f"{split}.npz", imgs=np.concatenate(imgs), labs=np.concatenate(labs),
                 pids=np.array(pids), fases=np.array(fases), zs=np.array(zs),
                 espaciado=ESPACIADO_OBJETIVO, tamano=TAMANO)
    return destino


def _cache_vigente(path):
    """Una cache de otra grilla entrenaria la red con otro preprocesamiento que la inferencia."""
    with np.load(path) as cache:
        if "espaciado" not in cache.files:
            return False
        return float(cache["espaciado"])==ESPACIADO_OBJETIVO and int(cache["tamano"])==TAMANO


def cargar_cache(split, destino=CACHE_DIR, root=ACDC_ROOT):
    path=Path(destino)/f"{split}.npz"
    if not path.exists() or not _cache_vigente(path):
        construir_cache(root, destino)
    with np.load(path) as cache:
        return {clave: cache[clave] for clave in cache.files}


def pliegues_por_paciente(pacientes, k=5, seed=0):
    """Devuelve {pid: pliegue}, con la misma cantidad de cada grupo en cada pliegue.

    Se divide por paciente porque los cortes de un mismo corazon son casi iguales.
    """
    rng=np.random.default_rng(seed)
    pliegues={}
    for grupo in GRUPOS:
        pids=sorted(p.pid for p in pacientes if p.grupo==grupo)
        rng.shuffle(pids)
        for i, pid in enumerate(pids):
            pliegues[pid]=i%k
    return pliegues


class CortesACDC(Dataset):
    """Devuelve imagen (1, H, W) float32 y mascara (H, W) int64.

    Con augment=True aplica flips y rotaciones de 90 grados a imagen y mascara, y una
    variacion leve de contraste solo a la imagen.
    """

    def __init__(self, imgs, labs, augment=False):
        assert len(imgs)==len(labs)
        self.imgs=imgs
        self.labs=labs
        self.augment=augment

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, lab=self.imgs[idx], self.labs[idx]
        if self.augment:
            if np.random.rand()<0.5:
                img, lab=np.fliplr(img), np.fliplr(lab)
            if np.random.rand()<0.5:
                img, lab=np.flipud(img), np.flipud(lab)
            k=np.random.randint(4)
            img, lab=np.rot90(img, k), np.rot90(lab, k)
            img=img*np.random.uniform(0.9, 1.1)+np.random.uniform(-0.1, 0.1)
        return torch.from_numpy(img.copy()).float().unsqueeze(0), torch.from_numpy(lab.copy()).long()


if __name__=="__main__":
    root=Path(sys.argv[1]) if len(sys.argv)>1 else ACDC_ROOT
    print(f"{len(listar_pacientes(root))} pacientes")
    for split in ("training", "testing"):
        cache=cargar_cache(split, root=root)
        fracciones=np.bincount(cache["labs"].ravel(), minlength=N_CLASES)/cache["labs"].size
        resumen=" ".join(f"{nombre}={f:.2%}" for nombre, f in zip(NOMBRES_CLASES, fracciones))
        print(f"{split:8s}: {len(cache['imgs'])} cortes {cache['imgs'].shape[1:]} | {resumen}")
