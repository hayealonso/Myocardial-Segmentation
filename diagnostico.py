"""
Diagnostico de los 5 grupos del ACDC a partir de biomarcadores (etapa 2).

Uso:
    python diagnostico.py               # despues de inferir_acdc.py

Lee outputs/acdc/pacientes.csv y compara dos fuentes de biomarcadores:
  - pipeline: medidos sobre las mascaras PREDICHAS (fuera de pliegue en training)
  - oraculo:  medidos sobre las mascaras reales (cota superior)

Regla de seleccion fijada antes de mirar test: el modelo con mejor exactitud
media en validacion cruzada repetida sobre training (5 pliegues x 10); en empate
gana el mas simple (regresion logistica). Test se evalua una sola vez.

Salidas en outputs/acdc/:
    diagnostico.json            exactitudes, matrices de confusion, modelo elegido
    diagnostico_pacientes.csv   prediccion y probabilidades por paciente
                                (en training, fuera de pliegue: insumo del modulo de seguridad)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from acdc_data import GRUPOS
from biomarcadores import NOMBRES

SEMILLA = 0


def candidatos(n_arboles: int = 500):
    """Orden = preferencia en empate (el primero es el mas simple)."""
    return {
        "logistica": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "bosque": RandomForestClassifier(n_estimators=n_arboles, min_samples_leaf=2,
                                         random_state=SEMILLA, n_jobs=-1),
    }


def elegir(puntajes: dict) -> str:
    """Mejor exactitud de CV redondeada a 3 decimales; en empate, el primero
    de `candidatos()` (el mas simple). Diferencias menores a 0.001 con 100
    pacientes son ruido, no razon para preferir el modelo mas complejo."""
    orden = list(candidatos())
    return max(puntajes, key=lambda n: (round(puntajes[n], 3), -orden.index(n)))


def matriz(fuente: str, df: pd.DataFrame) -> np.ndarray:
    return df[[f"{fuente}_{n}" for n in NOMBRES]].to_numpy(dtype=float)


def evaluar(df: pd.DataFrame, fuente: str, repeticiones: int = 10, n_arboles: int = 500) -> dict:
    """Selecciona el modelo con CV en training, entrena en training y evalua en test."""
    tr, te = df[df.split == "training"], df[df.split == "testing"]
    X_tr, y_tr = matriz(fuente, tr), tr.grupo.to_numpy()
    X_te, y_te = matriz(fuente, te), te.grupo.to_numpy()

    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeticiones, random_state=SEMILLA)
    puntajes = {nombre: cross_val_score(m, X_tr, y_tr, cv=cv).mean()
                for nombre, m in candidatos(n_arboles).items()}
    elegido = elegir(puntajes)

    modelo = candidatos(n_arboles)[elegido]
    oof = cross_val_predict(candidatos(n_arboles)[elegido], X_tr, y_tr, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA))
    modelo.fit(X_tr, y_tr)
    clases = list(modelo.classes_)
    prob_te = modelo.predict_proba(X_te)
    pred_te = np.array(clases)[prob_te.argmax(1)]
    return {
        "cv_training": {k: float(v) for k, v in puntajes.items()},
        "elegido": elegido,
        "exactitud_test": float(accuracy_score(y_te, pred_te)),
        "confusion_test": confusion_matrix(y_te, pred_te, labels=list(GRUPOS)).tolist(),
        "_clases": clases, "_prob_tr": oof, "_prob_te": prob_te,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Clasificador de diagnostico ACDC")
    ap.add_argument("--dir", type=str, default="outputs/acdc")
    args = ap.parse_args()
    out = Path(args.dir)
    df = pd.read_csv(out / "pacientes.csv")
    assert df[[f"pred_{n}" for n in NOMBRES]].notna().all().all(), "hay biomarcadores NaN"

    resultados, filas = {}, []
    for fuente in ("pred", "real"):
        r = evaluar(df, fuente)
        clave = "pipeline" if fuente == "pred" else "oraculo"
        resultados[clave] = {k: v for k, v in r.items() if not k.startswith("_")}
        print(f"{clave:8s}: CV training {r['cv_training']} -> {r['elegido']} | "
              f"exactitud test = {r['exactitud_test']:.2%}")
        if fuente == "pred":
            for split, probs in (("training", r["_prob_tr"]), ("testing", r["_prob_te"])):
                sub = df[df.split == split]
                for (_, fila), pr in zip(sub.iterrows(), probs):
                    orden = np.sort(pr)[::-1]
                    filas.append({"pid": fila.pid, "split": split, "grupo": fila.grupo,
                                  "pred": r["_clases"][int(pr.argmax())],
                                  "margen": float(orden[0] - orden[1]),
                                  **{f"p_{c}": float(v) for c, v in zip(r["_clases"], pr)}})

    resultados["grupos"] = list(GRUPOS)
    (out / "diagnostico.json").write_text(json.dumps(resultados, indent=2))
    pd.DataFrame(filas).to_csv(out / "diagnostico_pacientes.csv", index=False)
    print(f"Guardado {out / 'diagnostico.json'} y diagnostico_pacientes.csv")


if __name__ == "__main__":
    main()
