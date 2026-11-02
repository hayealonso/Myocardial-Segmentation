"""
Figuras de la etapa 2 en outputs/acdc/figuras/ (despues de seguridad.py).

    bland_altman.png        FE del VI, FE del VD y masa contra las mascaras de expertos (test)
    confusion.png           matriz de confusion en test: sistema y oraculo
    riesgo_cobertura.png    exactitud al conservar los casos menos dudosos
    segmentaciones_test.png los pacientes de test con menos y con mas duda
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_rgb
from matplotlib.patches import Patch

import acdc_data as A

AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"
COLOR_CLASE = {A.VD: AZUL, A.MIOCARDIO: NARANJA, A.VI: AQUA}


def _estilo(ax):
    ax.grid(color=GRILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(TINTA_2)
    ax.spines["bottom"].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=9)


def bland_altman(df, destino):
    test = df[df.split == "testing"]
    con_alerta = test.alerta.to_numpy()
    medidas = [("fe_vi", "FE del VI (%)"), ("fe_vd", "FE del VD (%)"), ("masa", "Masa del miocardio (g)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (nombre, titulo) in zip(axes, medidas):
        pred, real = test[f"pred_{nombre}"], test[f"real_{nombre}"]
        promedio, diferencia = (pred + real) / 2, pred - real
        sesgo, desvio = diferencia.mean(), diferencia.std()
        ax.scatter(promedio[~con_alerta], diferencia[~con_alerta], s=28, color=AZUL, edgecolor="white",
                   linewidth=1, label="sin alerta", zorder=3)
        ax.scatter(promedio[con_alerta], diferencia[con_alerta], s=28, color=NARANJA, edgecolor="white",
                   linewidth=1, label="con alerta", zorder=3)
        lineas = [(sesgo + 1.96 * desvio, "--", "bottom", "+1.96 DE"),
                  (sesgo, "-", "bottom", "sesgo"),
                  (sesgo - 1.96 * desvio, "--", "top", "-1.96 DE")]
        for y, estilo, alineacion, etiqueta in lineas:
            ax.axhline(y, color=TINTA_2, linewidth=1, linestyle=estilo, zorder=2)
            ax.text(1.0, y, f"{etiqueta} = {y:.1f}", transform=ax.get_yaxis_transform(),
                    va=alineacion, ha="right", fontsize=8, color=TINTA_2)
        ax.set_title(titulo, fontsize=11, color=TINTA, loc="left")
        ax.set_xlabel("promedio (prediccion, real)", fontsize=9, color=TINTA_2)
        ax.set_ylabel("prediccion - real", fontsize=9, color=TINTA_2)
        _estilo(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Biomarcadores en test (50 pacientes): acuerdo con las mascaras de expertos",
                 x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def confusion(resultados, destino):
    grupos = resultados["grupos"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    paneles = [(axes[0], "pipeline", "Sistema (mascaras predichas)"),
               (axes[1], "oraculo", "Oraculo (mascaras de expertos)")]
    for ax, clave, titulo in paneles:
        matriz = np.array(resultados[clave]["confusion_test"])
        ax.imshow(matriz, cmap="Blues", vmin=0, vmax=matriz.sum(axis=1).max())
        for i in range(len(grupos)):
            for j in range(len(grupos)):
                if matriz[i, j]:
                    color = "white" if matriz[i, j] > 5 else TINTA
                    ax.text(j, i, matriz[i, j], ha="center", va="center", fontsize=11, color=color)
        ax.set_xticks(range(len(grupos)), grupos, fontsize=9)
        ax.set_yticks(range(len(grupos)), grupos, fontsize=9)
        ax.set_xlabel("prediccion", fontsize=9, color=TINTA_2)
        ax.set_ylabel("grupo real", fontsize=9, color=TINTA_2)
        ax.set_title(f"{titulo}\nexactitud {resultados[clave]['exactitud_test']:.0%}",
                     fontsize=11, color=TINTA, loc="left")
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def riesgo_cobertura(seguridad, destino):
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    curvas = [("training", AZUL, "training (fuera de pliegue)"), ("testing", NARANJA, "test")]
    for split, color, nombre in curvas:
        curva = seguridad[split]["riesgo_cobertura"]
        cobertura = np.array(curva["cobertura"]) * 100
        exactitud = np.array(curva["exactitud"]) * 100
        ax.plot(cobertura, exactitud, color=color, linewidth=2, label=nombre)
        ax.text(101, exactitud[-1], nombre, color=TINTA_2, fontsize=9, va="center")
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("pacientes conservados, de menor a mayor duda (%)", fontsize=9, color=TINTA_2)
    ax.set_ylabel("exactitud del diagnostico (%)", fontsize=9, color=TINTA_2)
    ax.set_title("Exactitud al revisar primero los casos dudosos", fontsize=11, color=TINTA, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def segmentaciones(df, carpeta_mascaras, destino, n=3):
    """Corte con mas miocardio en FD de los n pacientes con menos duda y los n con mas duda."""
    test = df[df.split == "testing"].sort_values("duda")
    elegidos = list(test.head(n).itertuples()) + list(test.tail(n).itertuples())
    pacientes = {paciente.pid: paciente for paciente in A.listar_pacientes(split="testing")}
    fig, axes = plt.subplots(3, len(elegidos), figsize=(2.6 * len(elegidos), 8.4))

    for col, fila in enumerate(elegidos):
        img, gt, _ = A.cargar_fase(pacientes[fila.pid], "ED")
        pred = np.load(carpeta_mascaras / f"{fila.pid}.npz")["ED"]
        z = int(np.argmax((gt == A.MIOCARDIO).sum(axis=(0, 1))))
        for i, (mascara, titulo) in enumerate(((None, ""), (gt[..., z], "real"), (pred[z], "prediccion"))):
            ax = axes[i, col]
            ax.imshow(img[..., z], cmap="gray")
            ax.axis("off")
            ax.set_title(titulo, fontsize=9, color=TINTA_2)
            if mascara is not None:
                capa = np.zeros((*mascara.shape, 4))
                for clase, color in COLOR_CLASE.items():
                    capa[mascara == clase] = (*to_rgb(color), 0.55)
                ax.imshow(capa)
        marca = "ALERTA" if fila.alerta else "sin alerta"
        axes[0, col].set_title(f"{fila.pid}\nreal {fila.grupo} / pred {fila.pred}\n{marca}",
                               fontsize=9, color=NARANJA if fila.alerta else TINTA)

    leyenda = [Patch(color=AZUL, label="VD"), Patch(color=NARANJA, label="miocardio"), Patch(color=AQUA, label="VI")]
    fig.legend(handles=leyenda, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Test: los {n} pacientes con menos duda (izquierda) y con mas duda (derecha)",
                 x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Figuras de la etapa 2")
    parser.add_argument("--dir", default="outputs/acdc")
    args = parser.parse_args()
    # Agg solo como script: al importar apagaria las figuras en linea del notebook
    plt.switch_backend("Agg")
    out = Path(args.dir)
    carpeta = out / "figuras"
    carpeta.mkdir(exist_ok=True)

    seguridad = pd.read_csv(out / "seguridad_pacientes.csv")
    df = pd.read_csv(out / "pacientes.csv").merge(seguridad[["pid", "pred", "alerta", "duda"]], on="pid")
    bland_altman(df, carpeta / "bland_altman.png")
    confusion(json.loads((out / "diagnostico.json").read_text()), carpeta / "confusion.png")
    riesgo_cobertura(json.loads((out / "seguridad.json").read_text()), carpeta / "riesgo_cobertura.png")
    segmentaciones(df, out / "mascaras", carpeta / "segmentaciones_test.png")
    print(f"Figuras en {carpeta}")


if __name__ == "__main__":
    main()
