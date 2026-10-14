"""
Biomarcadores cardiacos a partir de mascaras de segmentacion (etapa 2, ACDC).

Las mascaras van en la grilla NATIVA del paciente, (Z, H, W) con etiquetas
0 fondo, 1 VD, 2 miocardio, 3 VI, y el espaciado (sx, sy, sz) en mm del header
NIfTI (sx corresponde al eje H, sy al eje W, sz a la separacion entre cortes).

Son las medidas que usa un cardiologo para diagnosticar los 5 grupos del ACDC:
  - volumenes de fin de diastole (FD) y de sistole (FS) y fraccion de eyeccion (FE)
    del VI y del VD: DCM (VI dilatado, FE baja), RV (VD dilatado), MINF (FE baja)
  - masa del miocardio: HCM (alta)
  - grosor de pared y su engrosamiento sistolico: HCM (gruesa), MINF (adelgazada
    en una region y con poco engrosamiento)
"""

from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
from scipy import ndimage

VD, MIOCARDIO, VI = 1, 2, 3
DENSIDAD_MIOCARDIO = 1.05  # g/ml


def volumen_ml(mascara: np.ndarray, clase: int, espaciado: Sequence[float]) -> float:
    """Volumen de una clase en ml: voxeles x volumen del voxel (mm^3 / 1000)."""
    return float((mascara == clase).sum() * np.prod(espaciado[:3]) / 1000.0)


def fraccion_eyeccion(v_fd: float, v_fs: float) -> float:
    """FE en %: (FD - FS) / FD."""
    return float(100.0 * (v_fd - v_fs) / v_fd) if v_fd > 0 else float("nan")


def grosores_de_pared(mascara: np.ndarray, espaciado: Sequence[float]) -> np.ndarray:
    """Grosor del miocardio en mm, medido sobre su linea media, en todos los cortes.

    En cada corte: transformada de distancia del miocardio (en mm, con el
    espaciado real de cada eje) y sus maximos locales, que forman la "cresta" a
    mitad de la pared. El grosor ahi es el doble de la distancia al pixel de
    fondo mas cercano. Con anillos de grosor conocido el error queda por debajo
    de un pixel (tests/test_biomarcadores.py).
    """
    sx, sy = espaciado[0], espaciado[1]
    valores = []
    for corte in mascara:
        miocardio = corte == MIOCARDIO
        if miocardio.sum() < 10:
            continue
        dist = ndimage.distance_transform_edt(miocardio, sampling=(sx, sy))
        cresta = miocardio & (dist == ndimage.maximum_filter(dist, size=3)) & (dist > 0)
        valores.append(2 * dist[cresta])
    return np.concatenate(valores) if valores else np.array([np.nan])


def biomarcadores(fd: np.ndarray, fs: np.ndarray, espaciado: Sequence[float],
                  superficie_corporal: float) -> Dict[str, float]:
    """Todos los biomarcadores de un paciente a partir de sus mascaras FD y FS."""
    vi_fd, vi_fs = volumen_ml(fd, VI, espaciado), volumen_ml(fs, VI, espaciado)
    vd_fd, vd_fs = volumen_ml(fd, VD, espaciado), volumen_ml(fs, VD, espaciado)
    mio_fd = volumen_ml(fd, MIOCARDIO, espaciado)
    g_fd, g_fs = grosores_de_pared(fd, espaciado), grosores_de_pared(fs, espaciado)
    sc = superficie_corporal
    return {
        "vi_fd": vi_fd, "vi_fs": vi_fs, "fe_vi": fraccion_eyeccion(vi_fd, vi_fs),
        "vd_fd": vd_fd, "vd_fs": vd_fs, "fe_vd": fraccion_eyeccion(vd_fd, vd_fs),
        "masa": mio_fd * DENSIDAD_MIOCARDIO,
        "vi_fd_idx": vi_fd / sc, "vd_fd_idx": vd_fd / sc,
        "masa_idx": mio_fd * DENSIDAD_MIOCARDIO / sc,
        "vd_vi": vd_fd / vi_fd if vi_fd > 0 else float("nan"),
        "masa_vi": mio_fd * DENSIDAD_MIOCARDIO / vi_fd if vi_fd > 0 else float("nan"),
        "grosor_max": float(np.nanpercentile(g_fd, 95)),
        "grosor_medio": float(np.nanmean(g_fd)),
        "grosor_cv": float(np.nanstd(g_fd) / np.nanmean(g_fd)),
        "engrosamiento": float(100.0 * (np.nanmean(g_fs) - np.nanmean(g_fd)) / np.nanmean(g_fd)),
    }


NOMBRES = ("vi_fd", "vi_fs", "fe_vi", "vd_fd", "vd_fs", "fe_vd", "masa", "vi_fd_idx",
           "vd_fd_idx", "masa_idx", "vd_vi", "masa_vi", "grosor_max", "grosor_medio",
           "grosor_cv", "engrosamiento")
