"""
Modulo de seguridad (etapa 2): marca los pacientes cuya prediccion es dudosa.

Uso:
    python seguridad.py                 # despues de inferir_acdc.py y diagnostico.py

Dos fuentes de duda, ambas disponibles sin conocer la respuesta correcta:
  - segmentacion: dispersion de los biomarcadores entre las 8 vistas de TTA
    (FE del VI y del VD, y volumen del VD y masa en relativo). Cada una se
    divide por su mediana en training y se toma la mayor.
  - diagnostico: margen del clasificador (probabilidad del primero menos la del
    segundo). Margen chico = dos diagnosticos casi empatados.

Regla fijada antes de mirar test: se marca un paciente si su duda de
segmentacion supera el percentil 90 de training, o si su margen queda bajo el
percentil 10 de training. Los umbrales salen solo de training (fuera de pliegue)
y se aplican tal cual a test.

Salidas en outputs/acdc/: seguridad.json y seguridad_pacientes.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PERCENTIL_SEG = 90
PERCENTIL_MARGEN = 10


def duda_segmentacion(df: pd.DataFrame, referencia: pd.DataFrame) -> pd.Series:
    """Mayor dispersion relativa entre vistas, en unidades de la mediana de training."""
    def componentes(d):
        return pd.DataFrame({
            "fe_vi": d.std_fe_vi,
            "fe_vd": d.std_fe_vd,
            "vd_fd": d.std_vd_fd / d.pred_vd_fd,
            "masa": d.std_masa / d.pred_masa,
        })
    ref = componentes(referencia).median()
    return (componentes(df) / ref).max(axis=1)


def riesgo_cobertura(acierto: np.ndarray, duda: np.ndarray):
    """Exactitud cuando se conservan solo los `cobertura` casos menos dudosos."""
    orden = np.argsort(duda, kind="stable")
    acierto = acierto[orden]
    n = np.arange(1, len(acierto) + 1)
    return n / len(acierto), np.cumsum(acierto) / n


def main() -> None:
    ap = argparse.ArgumentParser(description="Modulo de seguridad ACDC")
    ap.add_argument("--dir", type=str, default="outputs/acdc")
    args = ap.parse_args()
    out = Path(args.dir)

    df = pd.read_csv(out / "pacientes.csv").merge(
        pd.read_csv(out / "diagnostico_pacientes.csv")[["pid", "pred", "margen"]], on="pid")
    tr = df[df.split == "training"]

    df["duda_seg"] = duda_segmentacion(df, tr)
    umbral_seg = float(np.percentile(df.loc[df.split == "training", "duda_seg"], PERCENTIL_SEG))
    umbral_margen = float(np.percentile(tr.margen, PERCENTIL_MARGEN))
    df["alerta_seg"] = df.duda_seg > umbral_seg
    df["alerta_dx"] = df.margen < umbral_margen
    df["alerta"] = df.alerta_seg | df.alerta_dx
    df["acierto"] = df.pred == df.grupo
    df["error_fe_vi"] = (df.pred_fe_vi - df.real_fe_vi).abs()
    df["error_fe_vd"] = (df.pred_fe_vd - df.real_fe_vd).abs()
    # puntaje unico para ordenar: el peor de los dos, cada uno como percentil de training
    pct = lambda x, ref: np.searchsorted(np.sort(ref), x, side="right") / len(ref)
    df["duda"] = np.maximum(pct(df.duda_seg, df.loc[df.split == "training", "duda_seg"]),
                            pct(-df.margen, -tr.margen))

    resumen = {"umbral_duda_seg": umbral_seg, "umbral_margen": umbral_margen}
    for split in ("training", "testing"):
        d = df[df.split == split]
        m, nm = d[d.alerta], d[~d.alerta]
        cob, exa = riesgo_cobertura(d.acierto.to_numpy(), d.duda.to_numpy())
        resumen[split] = {
            "n": int(len(d)), "marcados": int(len(m)),
            "exactitud_todos": float(d.acierto.mean()),
            "exactitud_no_marcados": float(nm.acierto.mean()),
            "exactitud_marcados": float(m.acierto.mean()) if len(m) else None,
            "errores_atrapados": f"{int((~m.acierto).sum())} de {int((~d.acierto).sum())}",
            "error_fe_vi_medio_no_marcados": float(nm.error_fe_vi.mean()),
            "error_fe_vi_medio_marcados": float(m.error_fe_vi.mean()) if len(m) else None,
            "spearman_duda_vs_error_fe_vi": float(spearmanr(d.std_fe_vi, d.error_fe_vi).statistic),
            "spearman_duda_vs_error_fe_vd": float(spearmanr(d.std_fe_vd, d.error_fe_vd).statistic),
            "riesgo_cobertura": {"cobertura": cob.round(4).tolist(), "exactitud": exa.round(4).tolist()},
        }
        r = resumen[split]
        print(f"{split:8s}: {r['marcados']}/{r['n']} marcados | exactitud: todos {r['exactitud_todos']:.1%}, "
              f"no marcados {r['exactitud_no_marcados']:.1%} | errores atrapados {r['errores_atrapados']} | "
              f"error FE VI no marcados {r['error_fe_vi_medio_no_marcados']:.1f} vs marcados "
              f"{r['error_fe_vi_medio_marcados'] or float('nan'):.1f} pts")

    (out / "seguridad.json").write_text(json.dumps(resumen, indent=2))
    cols = ["pid", "split", "grupo", "pred", "acierto", "margen", "duda_seg", "duda",
            "alerta_seg", "alerta_dx", "alerta", "pred_fe_vi", "real_fe_vi", "std_fe_vi",
            "pred_fe_vd", "real_fe_vd", "std_fe_vd"]
    df[cols].to_csv(out / "seguridad_pacientes.csv", index=False)
    print(f"Guardado {out / 'seguridad.json'} y seguridad_pacientes.csv")


if __name__ == "__main__":
    main()
