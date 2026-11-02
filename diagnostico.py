"""
Predice el grupo de cada paciente del ACDC a partir de sus biomarcadores.

    python diagnostico.py  #despues de inferir_acdc.py

Se evalua con dos fuentes: las mascaras predichas (el sistema real) y las de los expertos
(el oraculo, la cota superior). El modelo se elige con validacion cruzada en training y
test se usa una sola vez.

Salidas en outputs/acdc/: diagnostico.json y diagnostico_pacientes.csv. En training las
probabilidades son fuera de pliegue, porque de ahi salen los umbrales de seguridad.
"""

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

SEMILLA=0


def candidatos(n_arboles=500):
    return {
        "logistica": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
        "bosque": RandomForestClassifier(n_estimators=n_arboles, min_samples_leaf=2,
                                         random_state=SEMILLA, n_jobs=-1),
    }


def elegir(puntajes):
    """El bosque solo gana si supera a la logistica al redondear a 3 decimales: con 100
    pacientes, una diferencia menor es ruido."""
    if round(puntajes["bosque"], 3)>round(puntajes["logistica"], 3):
        return "bosque"
    return "logistica"


def matriz(fuente, df):
    columnas=[f"{fuente}_{nombre}" for nombre in NOMBRES]
    return df[columnas].to_numpy(dtype=float)


def evaluar(df, fuente, repeticiones=10, n_arboles=500):
    """Elige el modelo con CV en training, lo entrena en training y lo evalua en test."""
    train=df[df.split=="training"]
    test=df[df.split=="testing"]
    X_train, y_train=matriz(fuente, train), train.grupo.to_numpy()
    X_test, y_test=matriz(fuente, test), test.grupo.to_numpy()

    cv=RepeatedStratifiedKFold(n_splits=5, n_repeats=repeticiones, random_state=SEMILLA)
    puntajes={}
    for nombre, modelo in candidatos(n_arboles).items():
        puntajes[nombre]=float(cross_val_score(modelo, X_train, y_train, cv=cv).mean())
    elegido=elegir(puntajes)

    modelo=candidatos(n_arboles)[elegido]
    probs_train=cross_val_predict(modelo, X_train, y_train, method="predict_proba",
                                  cv=StratifiedKFold(5, shuffle=True, random_state=SEMILLA))
    modelo.fit(X_train, y_train)
    clases=list(modelo.classes_)
    probs_test=modelo.predict_proba(X_test)
    pred_test=np.array(clases)[probs_test.argmax(axis=1)]
    return {
        "cv_training": puntajes,
        "elegido": elegido,
        "exactitud_test": float(accuracy_score(y_test, pred_test)),
        "confusion_test": confusion_matrix(y_test, pred_test, labels=list(GRUPOS)).tolist(),
        "_clases": clases,
        "_prob_tr": probs_train,
        "_prob_te": probs_test,
    }


def filas_por_paciente(df, split, probs, clases):
    filas=[]
    for (_, paciente), probs_paciente in zip(df[df.split==split].iterrows(), probs):
        ordenadas=np.sort(probs_paciente)
        fila={"pid": paciente.pid, "split": split, "grupo": paciente.grupo,
              "pred": clases[int(probs_paciente.argmax())],
              "margen": float(ordenadas[-1]-ordenadas[-2])}
        for clase, prob in zip(clases, probs_paciente):
            fila[f"p_{clase}"]=float(prob)
        filas.append(fila)
    return filas


def main():
    parser=argparse.ArgumentParser(description="Clasificador de diagnostico del ACDC")
    parser.add_argument("--dir", default="outputs/acdc")
    args=parser.parse_args()
    out=Path(args.dir)
    df=pd.read_csv(out/"pacientes.csv")
    columnas_pred=[f"pred_{nombre}" for nombre in NOMBRES]
    assert df[columnas_pred].notna().all().all(), "hay biomarcadores NaN"

    sistema=evaluar(df, "pred")
    oraculo=evaluar(df, "real")
    print(f"sistema: CV {sistema['cv_training']} -> {sistema['elegido']} | test {sistema['exactitud_test']:.2%}")
    print(f"oraculo: CV {oraculo['cv_training']} -> {oraculo['elegido']} | test {oraculo['exactitud_test']:.2%}")

    claves=("cv_training", "elegido", "exactitud_test", "confusion_test")
    resultados={
        "pipeline": {clave: sistema[clave] for clave in claves},
        "oraculo": {clave: oraculo[clave] for clave in claves},
        "grupos": list(GRUPOS),
    }
    (out/"diagnostico.json").write_text(json.dumps(resultados, indent=2))

    filas=filas_por_paciente(df, "training", sistema["_prob_tr"], sistema["_clases"])
    filas+=filas_por_paciente(df, "testing", sistema["_prob_te"], sistema["_clases"])
    pd.DataFrame(filas).to_csv(out/"diagnostico_pacientes.csv", index=False)
    print(f"Guardado en {out}")


if __name__=="__main__":
    main()
