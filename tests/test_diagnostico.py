"""Invariantes del clasificador y del modulo de seguridad (etapa 2).

La falla silenciosa tipica aca es la fuga: que test influya, aunque sea un poco,
en la eleccion del modelo o en los umbrales de alerta. El numero de test sale
optimista y nada lo delata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import diagnostico as D
import seguridad as S
from acdc_data import GRUPOS
from biomarcadores import NOMBRES


def _df_sintetico(seed=0, n_tr=20, n_te=10):
    """Biomarcadores falsos con senal por grupo, mismo formato que pacientes.csv."""
    rng = np.random.default_rng(seed)
    filas = []
    for split, n in (("training", n_tr), ("testing", n_te)):
        for g_idx, g in enumerate(GRUPOS):
            for i in range(n):
                base = rng.normal(g_idx, 0.6, len(NOMBRES))
                filas.append({"pid": f"{split}{g}{i}", "split": split, "grupo": g,
                              **{f"pred_{k}": v for k, v in zip(NOMBRES, base)},
                              **{f"real_{k}": v for k, v in zip(NOMBRES, base)}})
    return pd.DataFrame(filas)


def test_la_seleccion_del_modelo_no_mira_test():
    """Permutar las etiquetas de test no puede cambiar ni los puntajes de CV
    ni el modelo elegido: si cambiaran, test estaria entrando en la seleccion."""
    df = _df_sintetico()
    r1 = D.evaluar(df, "pred", repeticiones=1, n_arboles=20)
    alterado = df.copy()
    te = alterado.split == "testing"
    alterado.loc[te, "grupo"] = np.random.default_rng(1).permutation(alterado.loc[te, "grupo"].to_numpy())
    r2 = D.evaluar(alterado, "pred", repeticiones=1, n_arboles=20)
    assert r1["cv_training"] == r2["cv_training"] and r1["elegido"] == r2["elegido"]
    assert r1["exactitud_test"] != r2["exactitud_test"]


def test_probabilidades_de_training_son_fuera_de_pliegue():
    """El modulo de seguridad calibra su umbral con las probabilidades de
    training. Si fueran del modelo entrenado sobre esos mismos pacientes, los
    margenes saldrian enormes y el umbral no marcaria nada en test."""
    df = _df_sintetico()
    df.loc[df.split == "training", [f"pred_{k}" for k in NOMBRES]] = \
        np.random.default_rng(0).normal(size=(100, len(NOMBRES)))  # training sin senal
    r = D.evaluar(df, "pred", repeticiones=1, n_arboles=20)
    exactitud_oof = (np.array(r["_clases"])[r["_prob_tr"].argmax(1)]
                     == df[df.split == "training"].grupo.to_numpy()).mean()
    assert exactitud_oof < 0.5, "las probabilidades de training no son fuera de pliegue"


def test_empate_elige_el_modelo_mas_simple():
    assert D.elegir({"logistica": 0.9301, "bosque": 0.9304}) == "logistica"
    assert D.elegir({"bosque": 0.9304, "logistica": 0.9301}) == "logistica"
    assert D.elegir({"logistica": 0.90, "bosque": 0.93}) == "bosque"


def test_riesgo_cobertura_ordena_de_menor_a_mayor_duda():
    """Si la duda es perfecta (los errores son los mas dudosos), la exactitud
    al conservar los casos menos dudosos es 1 hasta llegar a los errores."""
    acierto = np.array([1, 1, 1, 0, 0], dtype=float)
    duda = np.array([0.1, 0.2, 0.3, 0.8, 0.9])
    cob, exa = S.riesgo_cobertura(acierto, duda)
    assert np.allclose(cob, [0.2, 0.4, 0.6, 0.8, 1.0])
    assert np.allclose(exa[:3], 1.0) and exa[-1] == pytest.approx(0.6)
    # duda invertida: lo peor aparece primero
    _, exa_inv = S.riesgo_cobertura(acierto, -duda)
    assert exa_inv[0] == 0.0


def test_duda_de_segmentacion_se_escala_con_training():
    """Cada componente se divide por su mediana en training: el puntaje de
    test no puede depender de los otros pacientes de test."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "split": ["training"] * 20 + ["testing"] * 5,
        "std_fe_vi": rng.uniform(0.5, 2, 25), "std_fe_vd": rng.uniform(1, 4, 25),
        "std_vd_fd": rng.uniform(1, 5, 25), "pred_vd_fd": rng.uniform(100, 200, 25),
        "std_masa": rng.uniform(1, 5, 25), "pred_masa": rng.uniform(80, 160, 25),
    })
    ref = df[df.split == "training"]
    completo = S.duda_segmentacion(df, ref)
    solo_uno = S.duda_segmentacion(df.iloc[[22]], ref)
    assert completo.iloc[22] == pytest.approx(solo_uno.iloc[0])
