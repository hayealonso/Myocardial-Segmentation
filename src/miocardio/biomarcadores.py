"""
Biomarcadores cardiacos a partir de mascaras de segmentacion.

Las mascaras van en la grilla nativa, (Z, H, W), con las etiquetas del ACDC. El
espaciado (sx, sy, sz) viene del header NIfTI: sx es el eje H, sy el eje W y sz la
separacion entre cortes.

Son las medidas con que se distinguen los 5 grupos: DCM tiene el VI dilatado y FE baja,
RV el VD dilatado, MINF FE baja y poco engrosamiento, HCM la pared gruesa y masa alta.
"""

import numpy as np
from scipy import ndimage

from miocardio.acdc_data import MIOCARDIO, VD, VI

DENSIDAD_MIOCARDIO=1.05  #g/ml

NOMBRES=("vi_fd", "vi_fs", "fe_vi", "vd_fd", "vd_fs", "fe_vd", "masa", "vi_fd_idx",
         "vd_fd_idx", "masa_idx", "vd_vi", "masa_vi", "grosor_max", "grosor_medio",
         "grosor_cv", "engrosamiento")


def volumen_ml(mascara, clase, espaciado):
    volumen_voxel_mm3=espaciado[0]*espaciado[1]*espaciado[2]
    return float((mascara==clase).sum()*volumen_voxel_mm3/1000.0)


def fraccion_eyeccion(v_fd, v_fs):
    """FE en %: (FD - FS) / FD."""
    if v_fd<=0:
        return float("nan")
    return float(100.0*(v_fd-v_fs)/v_fd)


def grosores_de_pared(mascara, espaciado):
    """Grosor del miocardio en mm, medido en la linea media de la pared, en todos los cortes.

    La transformada de distancia da, por pixel, los mm hasta el borde. Sus maximos locales
    estan a mitad de la pared, y ahi el grosor es el doble de esa distancia.
    """
    sx, sy=espaciado[0], espaciado[1]
    valores=[]
    for corte in mascara:
        miocardio=corte==MIOCARDIO
        if miocardio.sum()<10:
            continue
        distancia=ndimage.distance_transform_edt(miocardio, sampling=(sx, sy))
        linea_media=miocardio&(distancia==ndimage.maximum_filter(distancia, size=3))&(distancia>0)
        valores.append(2*distancia[linea_media])
    if not valores:
        return np.array([np.nan])
    return np.concatenate(valores)


def biomarcadores(fd, fs, espaciado, superficie_corporal):
    """Los 16 biomarcadores de un paciente a partir de sus mascaras de FD y FS."""
    vi_fd=volumen_ml(fd, VI, espaciado)
    vi_fs=volumen_ml(fs, VI, espaciado)
    vd_fd=volumen_ml(fd, VD, espaciado)
    vd_fs=volumen_ml(fs, VD, espaciado)
    masa=volumen_ml(fd, MIOCARDIO, espaciado)*DENSIDAD_MIOCARDIO
    grosor_fd=grosores_de_pared(fd, espaciado)
    grosor_fs=grosores_de_pared(fs, espaciado)
    grosor_medio=np.nanmean(grosor_fd)
    return {
        "vi_fd": vi_fd,
        "vi_fs": vi_fs,
        "fe_vi": fraccion_eyeccion(vi_fd, vi_fs),
        "vd_fd": vd_fd,
        "vd_fs": vd_fs,
        "fe_vd": fraccion_eyeccion(vd_fd, vd_fs),
        "masa": masa,
        "vi_fd_idx": vi_fd/superficie_corporal,
        "vd_fd_idx": vd_fd/superficie_corporal,
        "masa_idx": masa/superficie_corporal,
        "vd_vi": vd_fd/vi_fd if vi_fd>0 else float("nan"),
        "masa_vi": masa/vi_fd if vi_fd>0 else float("nan"),
        "grosor_max": float(np.nanpercentile(grosor_fd, 95)),
        "grosor_medio": float(grosor_medio),
        "grosor_cv": float(np.nanstd(grosor_fd)/grosor_medio),
        "engrosamiento": float(100.0*(np.nanmean(grosor_fs)-grosor_medio)/grosor_medio),
    }
