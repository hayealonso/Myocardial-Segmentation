"""
Datos del ACDC original (150 pacientes, NIfTI) para la etapa 2 del proyecto:
segmentacion multiclase 2D -> biomarcadores -> diagnostico (ver docs/plan-acdc.md).

Convencion de etiquetas del ACDC, DISTINTA de cardiac_data/*.npy (alli 1 = miocardio):
    0 fondo, 1 cavidad del VD, 2 miocardio, 3 cavidad del VI

Cada corte se procesa por separado (enfoque 2D):
    normalizar el volumen -> remuestrear el corte a 1.5 mm -> recortar 192x192 al centro
y `a_nativo` deshace el recorte y el remuestreo, para medir volumenes con el
espaciado original de cada paciente.

Uso:
    python acdc_data.py                    # construye la cache y resume los datos
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

ACDC_ROOT = Path("data/ACDC/database")
CACHE_DIR = Path("data/cache_acdc")

FONDO, VD, MIOCARDIO, VI = 0, 1, 2, 3
N_CLASES = 4
NOMBRES_CLASES = ("fondo", "VD", "miocardio", "VI")
GRUPOS = ("NOR", "MINF", "DCM", "HCM", "RV")
FASES = ("ED", "ES")

ESPACIADO_OBJETIVO = 1.5  # mm por pixel en el plano
TAMANO = 192              # 288 mm; el peor caso del ACDC necesita 179 px


# --------------------------------------------------------------------------- #
# Lectura
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Paciente:
    pid: str          # "patient001"
    split: str        # "training" | "testing" (division oficial del ACDC)
    grupo: str        # uno de GRUPOS
    altura: float     # cm
    peso: float       # kg
    ed: int           # indice de frame (1-based, como en los nombres de archivo)
    es: int
    carpeta: Path

    def archivo(self, fase: str, gt: bool = False) -> Path:
        frame = self.ed if fase == "ED" else self.es
        sufijo = "_gt" if gt else ""
        return self.carpeta / f"{self.pid}_frame{frame:02d}{sufijo}.nii.gz"

    @property
    def superficie_corporal(self) -> float:
        """m^2, formula de Mosteller: los volumenes cardiacos se indexan por ella."""
        return float(np.sqrt(self.altura * self.peso / 3600.0))


def leer_info(path: Path) -> Dict[str, str]:
    info = {}
    for linea in Path(path).read_text().splitlines():
        if ":" in linea:
            k, v = linea.split(":", 1)
            info[k.strip()] = v.strip()
    return info


def listar_pacientes(root: Path = ACDC_ROOT, split: Optional[str] = None) -> List[Paciente]:
    splits = (split,) if split else ("training", "testing")
    pacientes = []
    for s in splits:
        for carpeta in sorted(Path(root, s).glob("patient*")):
            info = leer_info(carpeta / "Info.cfg")
            pacientes.append(Paciente(
                pid=carpeta.name, split=s, grupo=info["Group"],
                altura=float(info["Height"]), peso=float(info["Weight"]),
                ed=int(info["ED"]), es=int(info["ES"]), carpeta=carpeta,
            ))
    return pacientes


def cargar_fase(p: Paciente, fase: str):
    """Devuelve (img float32 (H, W, Z), gt uint8 (H, W, Z), espaciado (sx, sy, sz) en mm)."""
    import nibabel as nib

    img = nib.load(p.archivo(fase))
    gt = nib.load(p.archivo(fase, gt=True))
    espaciado = tuple(float(z) for z in img.header.get_zooms()[:3])
    return (np.asarray(img.dataobj, dtype=np.float32),
            np.asarray(gt.dataobj).astype(np.uint8), espaciado)


# --------------------------------------------------------------------------- #
# Preprocesamiento reversible
# --------------------------------------------------------------------------- #

def normalizar_volumen(vol: np.ndarray) -> np.ndarray:
    """Recorte a percentiles 1-99 y z-score, con estadisticos DEL VOLUMEN.

    Cada equipo guarda las intensidades en otra escala (float32, int16, uint8):
    una normalizacion global dejaria a cada paciente en un rango distinto.
    """
    lo, hi = np.percentile(vol, (1, 99))
    v = np.clip(vol, lo, hi)
    return ((v - v.mean()) / (v.std() + 1e-8)).astype(np.float32)


def _tamano_remuestreado(shape_hw, espaciado_xy, objetivo=ESPACIADO_OBJETIVO):
    return tuple(max(1, int(round(n * s / objetivo))) for n, s in zip(shape_hw, espaciado_xy))


def remuestrear(cortes: np.ndarray, tamano: Tuple[int, int], etiqueta: bool) -> np.ndarray:
    """(Z, H, W) -> (Z, h, w). Imagen bilinear; mascara `nearest-exact`.

    `nearest` a secas toma el pixel de la esquina y no el del centro: con una
    imagen bilinear (align_corners=False) la mascara queda corrida medio pixel.
    """
    t = torch.from_numpy(np.ascontiguousarray(cortes)).float().unsqueeze(1)
    if etiqueta:
        out = F.interpolate(t, size=tamano, mode="nearest-exact")
        return out.squeeze(1).round().numpy().astype(np.uint8)
    out = F.interpolate(t, size=tamano, mode="bilinear", align_corners=False)
    return out.squeeze(1).numpy().astype(np.float32)


def centrar(arr: np.ndarray, tamano: int, relleno: float = 0.0) -> np.ndarray:
    """Recorta o rellena los dos ultimos ejes a (tamano, tamano), centrado."""
    out = np.full(arr.shape[:-2] + (tamano, tamano), relleno, dtype=arr.dtype)
    src, dst = [], []
    for n in arr.shape[-2:]:
        if n >= tamano:
            a = (n - tamano) // 2
            src.append(slice(a, a + tamano)); dst.append(slice(0, tamano))
        else:
            a = (tamano - n) // 2
            src.append(slice(0, n)); dst.append(slice(a, a + n))
    out[(..., *dst)] = arr[(..., *src)]
    return out


def descentrar(arr: np.ndarray, shape_hw: Tuple[int, int], relleno: float = 0.0) -> np.ndarray:
    """Inversa de `centrar`: devuelve los dos ultimos ejes a `shape_hw`."""
    tamano = arr.shape[-1]
    out = np.full(arr.shape[:-2] + tuple(shape_hw), relleno, dtype=arr.dtype)
    src, dst = [], []
    for n in shape_hw:
        if n >= tamano:
            a = (n - tamano) // 2
            dst.append(slice(a, a + tamano)); src.append(slice(0, tamano))
        else:
            a = (tamano - n) // 2
            dst.append(slice(0, n)); src.append(slice(a, a + n))
    out[(..., *dst)] = arr[(..., *src)]
    return out


@dataclass(frozen=True)
class Geometria:
    """Lo necesario para devolver una prediccion a la grilla nativa."""
    shape_hw: Tuple[int, int]        # tamano nativo del corte
    remuestreado_hw: Tuple[int, int]  # tamano a ESPACIADO_OBJETIVO, antes de centrar
    espaciado: Tuple[float, float, float]


def preparar_volumen(img: np.ndarray, gt: Optional[np.ndarray], espaciado):
    """(H, W, Z) nativo -> (Z, TAMANO, TAMANO) listo para la red, mas su Geometria."""
    shape_hw = img.shape[:2]
    rem_hw = _tamano_remuestreado(shape_hw, espaciado[:2])
    norm = normalizar_volumen(img)
    cortes = remuestrear(np.moveaxis(norm, -1, 0), rem_hw, etiqueta=False)
    cortes = centrar(cortes, TAMANO, relleno=float(norm.min()))
    etiquetas = None
    if gt is not None:
        etiquetas = centrar(remuestrear(np.moveaxis(gt, -1, 0), rem_hw, etiqueta=True), TAMANO)
    return cortes, etiquetas, Geometria(tuple(shape_hw), rem_hw, tuple(espaciado))


def a_nativo(probs: np.ndarray, geo: Geometria) -> np.ndarray:
    """Probabilidades (Z, C, TAMANO, TAMANO) -> (Z, C, H, W) en la grilla nativa.

    Se interpolan probabilidades y no etiquetas: el argmax se toma recien en la
    resolucion final, asi los bordes no quedan escalonados. Fuera del recorte la
    probabilidad es 1 para el fondo.
    """
    z, c = probs.shape[:2]
    fuera = np.zeros((c,), dtype=np.float32); fuera[FONDO] = 1.0
    sin_centrar = descentrar(probs, geo.remuestreado_hw)
    # lo que quedo fuera del recorte (relleno 0) pasa a fondo
    vacio = sin_centrar.sum(axis=1, keepdims=True) == 0
    sin_centrar = np.where(vacio, fuera[None, :, None, None], sin_centrar)
    t = torch.from_numpy(sin_centrar.reshape(z * c, 1, *geo.remuestreado_hw))
    out = F.interpolate(t, size=geo.shape_hw, mode="bilinear", align_corners=False)
    return out.reshape(z, c, *geo.shape_hw).numpy()


# --------------------------------------------------------------------------- #
# Cache de cortes 2D
# --------------------------------------------------------------------------- #

def construir_cache(root: Path = ACDC_ROOT, destino: Path = CACHE_DIR) -> Path:
    """Preprocesa los 150 pacientes una vez y guarda los cortes en un .npz por split.

    Por corte: imagen, mascara, pid, fase e indice z. Se reconstruye si se borra.
    """
    destino = Path(destino)
    destino.mkdir(parents=True, exist_ok=True)
    for split in ("training", "testing"):
        imgs, labs, pids, fases, zs = [], [], [], [], []
        for p in listar_pacientes(root, split):
            for fase in FASES:
                img, gt, esp = cargar_fase(p, fase)
                cortes, etiquetas, _ = preparar_volumen(img, gt, esp)
                imgs.append(cortes); labs.append(etiquetas)
                pids += [p.pid] * len(cortes); fases += [fase] * len(cortes)
                zs += list(range(len(cortes)))
        np.savez(destino / f"{split}.npz", imgs=np.concatenate(imgs), labs=np.concatenate(labs),
                 pids=np.array(pids), fases=np.array(fases), zs=np.array(zs))
    return destino


def cargar_cache(split: str, destino: Path = CACHE_DIR, root: Path = ACDC_ROOT) -> Dict[str, np.ndarray]:
    path = Path(destino) / f"{split}.npz"
    if not path.exists():
        construir_cache(root, destino)
    with np.load(path) as d:
        return {k: d[k] for k in d.files}


# --------------------------------------------------------------------------- #
# Pliegues por paciente y Dataset
# --------------------------------------------------------------------------- #

def pliegues_por_paciente(pacientes: Sequence[Paciente], k: int = 5, seed: int = 0) -> Dict[str, int]:
    """pid -> pliegue, estratificado por grupo (20 por grupo -> 4 por pliegue).

    Por paciente y no por corte: los cortes vecinos de un mismo corazon son casi
    iguales, y repartirlos entre train y validacion infla la metrica sin error.
    """
    rng = np.random.default_rng(seed)
    asignacion = {}
    for grupo in GRUPOS:
        pids = sorted(p.pid for p in pacientes if p.grupo == grupo)
        rng.shuffle(pids)
        for i, pid in enumerate(pids):
            asignacion[pid] = i % k
    return asignacion


class CortesACDC(torch.utils.data.Dataset):
    """Cortes 2D: imagen (1, H, W) float32 y mascara (H, W) int64 en {0..3}.

    Aumentacion (solo train): flips y rotaciones de 90 grados, identicos en
    imagen y mascara, mas una variacion leve de contraste solo en la imagen.
    """

    def __init__(self, imgs: np.ndarray, labs: np.ndarray, augment: bool = False):
        assert len(imgs) == len(labs)
        self.imgs, self.labs, self.augment = imgs, labs, augment

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        img, lab = self.imgs[idx], self.labs[idx]
        if self.augment:
            if np.random.rand() < 0.5:
                img, lab = img[:, ::-1], lab[:, ::-1]
            if np.random.rand() < 0.5:
                img, lab = img[::-1, :], lab[::-1, :]
            k = np.random.randint(4)
            img, lab = np.rot90(img, k), np.rot90(lab, k)
            img = img * np.random.uniform(0.9, 1.1) + np.random.uniform(-0.1, 0.1)
        img_t = torch.from_numpy(np.ascontiguousarray(img, dtype=np.float32)).unsqueeze(0)
        lab_t = torch.from_numpy(np.ascontiguousarray(lab).astype(np.int64))
        return img_t, lab_t


if __name__ == "__main__":
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ACDC_ROOT
    pacientes = listar_pacientes(root)
    print(f"{len(pacientes)} pacientes")
    for split in ("training", "testing"):
        d = cargar_cache(split, root=root)
        frac = np.bincount(d["labs"].ravel(), minlength=N_CLASES) / d["labs"].size
        print(f"{split:8s}: {len(d['imgs'])} cortes {d['imgs'].shape[1:]} | "
              + " ".join(f"{n}={f:.2%}" for n, f in zip(NOMBRES_CLASES, frac)))
