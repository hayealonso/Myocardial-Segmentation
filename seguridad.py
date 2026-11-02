"""
Marca a los pacientes cuya prediccion es dudosa.

    python seguridad.py  #despues de inferir_acdc.py y diagnostico.py

Dos senales que no necesitan conocer la respuesta:
  - duda de segmentacion: cuanto cambian los biomarcadores entre las 8 vistas del TTA
  - margen del clasificador: probabilidad del primer grupo menos la del segundo

Se marca un paciente si la duda supera el percentil 90 de training o el margen queda
bajo el percentil 10 de training. Los umbrales salen solo de training.

Salidas en outputs/acdc/: seguridad.json y seguridad_pacientes.csv
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PERCENTIL_DUDA=90
PERCENTIL_MARGEN=10


def dispersiones(df):
    return pd.DataFrame({
        "fe_vi": df.std_fe_vi,
        "fe_vd": df.std_fe_vd,
        "vd_fd": df.std_vd_fd/df.pred_vd_fd,
        "masa": df.std_masa/df.pred_masa,
    })


def duda_segmentacion(df, referencia):
    """Mayor dispersion entre vistas, medida en medianas de training."""
    medianas=dispersiones(referencia).median()
    return (dispersiones(df)/medianas).max(axis=1)


def percentil_en_referencia(valores, referencia):
    """Fraccion de la referencia que queda en o bajo cada valor."""
    referencia=np.asarray(referencia)
    return np.array([(referencia<=valor).mean() for valor in valores])


def riesgo_cobertura(acierto, duda):
    """Exactitud al conservar los casos menos dudosos, con un punto por nivel de duda.

    Los empates entran juntos para que la curva no dependa del orden de las filas.
    """
    orden=np.argsort(duda, kind="stable")
    acierto, duda=acierto[orden], duda[orden]
    n=np.arange(1, len(acierto)+1)
    fin_de_nivel=np.append(duda[1:]!=duda[:-1], True)
    return (n/len(acierto))[fin_de_nivel], (np.cumsum(acierto)/n)[fin_de_nivel]


def resumir(df):
    marcados=df[df.alerta]
    no_marcados=df[~df.alerta]
    cobertura, exactitud=riesgo_cobertura(df.acierto.to_numpy(), df.duda.to_numpy())
    return {
        "n": len(df),
        "marcados": len(marcados),
        "exactitud_todos": float(df.acierto.mean()),
        "exactitud_no_marcados": float(no_marcados.acierto.mean()),
        "exactitud_marcados": float(marcados.acierto.mean()) if len(marcados) else None,
        "errores_atrapados": f"{(~marcados.acierto).sum()} de {(~df.acierto).sum()}",
        "error_fe_vi_medio_no_marcados": float(no_marcados.error_fe_vi.mean()),
        "error_fe_vi_medio_marcados": float(marcados.error_fe_vi.mean()) if len(marcados) else None,
        "spearman_std_vs_error_fe_vi": float(spearmanr(df.std_fe_vi, df.error_fe_vi).statistic),
        "spearman_std_vs_error_fe_vd": float(spearmanr(df.std_fe_vd, df.error_fe_vd).statistic),
        "riesgo_cobertura": {"cobertura": cobertura.round(4).tolist(), "exactitud": exactitud.round(4).tolist()},
    }


def main():
    parser=argparse.ArgumentParser(description="Modulo de seguridad del ACDC")
    parser.add_argument("--dir", default="outputs/acdc")
    args=parser.parse_args()
    out=Path(args.dir)

    diagnostico=pd.read_csv(out/"diagnostico_pacientes.csv")
    df=pd.read_csv(out/"pacientes.csv").merge(diagnostico[["pid", "pred", "margen"]], on="pid")

    es_train=df.split=="training"
    df["duda_seg"]=duda_segmentacion(df, df[es_train])
    umbral_duda=float(np.percentile(df.loc[es_train, "duda_seg"], PERCENTIL_DUDA))
    umbral_margen=float(np.percentile(df.loc[es_train, "margen"], PERCENTIL_MARGEN))

    df["alerta_seg"]=df.duda_seg>umbral_duda
    df["alerta_dx"]=df.margen<umbral_margen
    df["alerta"]=df.alerta_seg|df.alerta_dx
    df["acierto"]=df.pred==df.grupo
    df["error_fe_vi"]=(df.pred_fe_vi-df.real_fe_vi).abs()
    df["error_fe_vd"]=(df.pred_fe_vd-df.real_fe_vd).abs()
    #puntaje para ordenar: la peor de las dos senales, como percentil de training
    df["duda"]=np.maximum(percentil_en_referencia(df.duda_seg, df.loc[es_train, "duda_seg"]),
                          percentil_en_referencia(-df.margen, -df.loc[es_train, "margen"]))

    resumen={"umbral_duda_seg": umbral_duda, "umbral_margen": umbral_margen}
    for split in ("training", "testing"):
        resumen[split]=resumir(df[df.split==split])
        r=resumen[split]
        print(f"{split:8s}: {r['marcados']}/{r['n']} marcados, errores atrapados {r['errores_atrapados']}, "
              f"exactitud {r['exactitud_todos']:.1%} (no marcados {r['exactitud_no_marcados']:.1%})")

    (out/"seguridad.json").write_text(json.dumps(resumen, indent=2))
    columnas=["pid", "split", "grupo", "pred", "acierto", "margen", "duda_seg", "duda",
              "alerta_seg", "alerta_dx", "alerta", "pred_fe_vi", "real_fe_vi", "std_fe_vi",
              "pred_fe_vd", "real_fe_vd", "std_fe_vd"]
    df[columnas].to_csv(out/"seguridad_pacientes.csv", index=False)
    print(f"Guardado en {out}")


if __name__=="__main__":
    main()
