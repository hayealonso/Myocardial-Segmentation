"""Invariantes de biomarcadores e inferencia (etapa 2): fallas silenciosas.

Un biomarcador mal calculado no lanza nada: la FE sale corrida unos puntos o
el grosor duplicado, y el clasificador aprende igual sobre numeros falsos.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import biomarcadores as B
from inferir_acdc import deshacer_vista, vistas_d4


def _anillo(g_mm, sx, sy, r_cav=25.0, n=200):
    """Corte con cavidad del VI de radio r_cav y pared de grosor g_mm (en mm)."""
    yy, xx = np.mgrid[:n, :n]
    d = np.hypot((yy - n / 2) * sx, (xx - n / 2) * sy)
    return np.where(d < r_cav, B.VI, np.where(d < r_cav + g_mm, B.MIOCARDIO, 0)).astype(np.uint8)


def test_volumen_usa_el_espaciado_de_los_tres_ejes():
    """Un bloque de 10x20x4 voxeles de 1.5x1.2x10 mm = 14.4 ml. Olvidar la
    separacion entre cortes (5 o 10 mm segun el paciente) da volumenes que no
    se pueden comparar entre pacientes."""
    m = np.zeros((4, 30, 30), dtype=np.uint8)
    m[:, 5:15, 5:25] = B.VI
    assert B.volumen_ml(m, B.VI, (1.5, 1.2, 10.0)) == pytest.approx(10 * 20 * 4 * 18 / 1000)


def test_fraccion_de_eyeccion():
    assert B.fraccion_eyeccion(100.0, 40.0) == pytest.approx(60.0)
    assert np.isnan(B.fraccion_eyeccion(0.0, 0.0))


@pytest.mark.parametrize("sx,sy", [(1.0, 1.0), (1.37, 1.37), (1.68, 1.2)])
@pytest.mark.parametrize("grosor", [6.0, 10.0, 16.0])
def test_grosor_de_pared_con_anillos_conocidos(sx, sy, grosor):
    """Pared de 6 mm (normal), 10 mm y 16 mm (hipertrofia): el error tiene que
    quedar por debajo de un pixel con cualquier espaciado del ACDC."""
    g = B.grosores_de_pared(_anillo(grosor, sx, sy)[None], (sx, sy, 10.0))
    assert abs(np.median(g) - grosor) < max(sx, sy)


def test_grosor_respeta_que_eje_tiene_cada_espaciado():
    """sx corresponde al eje H y sy al eje W. Una banda vertical de 10 px de
    ancho mide 10 mm con sy = 1; con los espaciados intercambiados mediria
    20 mm, y ningun shape cambia."""
    m = np.zeros((1, 60, 60), dtype=np.uint8)
    m[0, 5:55, 20:30] = B.MIOCARDIO  # grosor a lo largo de W
    g = B.grosores_de_pared(m, (2.0, 1.0, 10.0))
    assert np.median(g) == pytest.approx(10.0, abs=1.0)


def test_fd_y_fs_no_se_pueden_intercambiar():
    """FD es el corazon lleno (cavidad grande, pared delgada) y FS el
    contraido. Con el orden invertido la FE sale negativa y el engrosamiento
    tambien: el clasificador recibiria los signos al reves."""
    fd = np.stack([_anillo(8.0, 1.4, 1.4, r_cav=28.0)] * 6)
    fs = np.stack([_anillo(12.0, 1.4, 1.4, r_cav=18.0)] * 6)
    b = B.biomarcadores(fd, fs, (1.4, 1.4, 10.0), superficie_corporal=1.9)
    assert b["vi_fd"] > b["vi_fs"] and 40 < b["fe_vi"] < 80
    assert b["engrosamiento"] > 20
    assert b["vi_fd_idx"] == pytest.approx(b["vi_fd"] / 1.9)
    assert set(b) == set(B.NOMBRES)


def test_tta_deshace_cada_vista():
    """Cada una de las 8 vistas, desrotada, tiene que volver a la imagen
    original. Con el orden de inversion equivocado (flip antes que la rotacion)
    las vistas con flip quedan giradas 180 grados y el promedio de TTA mezcla
    predicciones desalineadas: el Dice baja y la incertidumbre se dispara."""
    x = torch.arange(2 * 3 * 5 * 5, dtype=torch.float32).reshape(2, 3, 5, 5)
    vistas = list(vistas_d4(x))
    assert len({v.numpy().tobytes() for _, _, v in vistas}) == 8
    for k, flip, v in vistas:
        assert torch.equal(deshacer_vista(v, k, flip), x), f"vista k={k} flip={flip}"
