"""
Figuras de la etapa 2 en outputs/acdc/figuras/ (despues de seguridad.py).

    bland_altman.png        FE del VI, FE del VD y masa: prediccion contra mascara real (test)
    confusion.png           matriz de confusion en test: pipeline y oraculo
    riesgo_cobertura.png    exactitud al conservar solo los casos menos dudosos
    segmentaciones_test.png corte medio en FD de los pacientes mas y menos dudosos
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors
import matplotlib.patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import acdc_data as A

# paleta de referencia (skill dataviz), en orden fijo
AZUL, NARANJA, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
TINTA, TINTA_2, GRILLA = "#0b0b0b", "#52514e", "#e4e3df"
COLOR_CLASE = {A.VD: AZUL, A.MIOCARDIO: NARANJA, A.VI: AQUA}


def _estilo(ax):
    ax.grid(color=GRILLA, linewidth=0.8)
    ax.set_axisbelow(True)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(TINTA_2)
    ax.tick_params(colors=TINTA_2, labelsize=9)


def bland_altman(df: pd.DataFrame, destino: Path) -> None:
    te = df[df.split == "testing"]
    medidas = [("fe_vi", "FE del VI (%)"), ("fe_vd", "FE del VD (%)"), ("masa", "Masa del miocardio (g)")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (n, titulo) in zip(axes, medidas):
        pred, real = te[f"pred_{n}"], te[f"real_{n}"]
        media, dif = (pred + real) / 2, pred - real
        sesgo, ds = dif.mean(), dif.std()
        marcado = te.alerta.to_numpy()
        ax.scatter(media[~marcado], dif[~marcado], s=28, color=AZUL, edgecolor="white",
                   linewidth=1, label="sin alerta", zorder=3)
        ax.scatter(media[marcado], dif[marcado], s=28, color=NARANJA, edgecolor="white",
                   linewidth=1, label="con alerta", zorder=3)
        for y, estilo in ((sesgo, "-"), (sesgo + 1.96 * ds, "--"), (sesgo - 1.96 * ds, "--")):
            ax.axhline(y, color=TINTA_2, linewidth=1, linestyle=estilo, zorder=2)
        ax.text(1.0, sesgo + 1.96 * ds, f" +1.96 DE = {sesgo + 1.96 * ds:.1f}", transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", fontsize=8, color=TINTA_2)
        ax.text(1.0, sesgo, f" sesgo = {sesgo:.1f}", transform=ax.get_yaxis_transform(),
                va="bottom", ha="right", fontsize=8, color=TINTA_2)
        ax.text(1.0, sesgo - 1.96 * ds, f" -1.96 DE = {sesgo - 1.96 * ds:.1f}", transform=ax.get_yaxis_transform(),
                va="top", ha="right", fontsize=8, color=TINTA_2)
        ax.set_title(titulo, fontsize=11, color=TINTA, loc="left")
        ax.set_xlabel("promedio (prediccion, real)", fontsize=9, color=TINTA_2)
        ax.set_ylabel("prediccion - real", fontsize=9, color=TINTA_2)
        _estilo(ax)
    axes[0].legend(frameon=False, fontsize=9, loc="lower left")
    fig.suptitle("Biomarcadores en test (50 pacientes): acuerdo con las mascaras reales",
                 x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def confusion(resultados: dict, destino: Path) -> None:
    grupos = resultados["grupos"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, clave, titulo in ((axes[0], "pipeline", "Pipeline (mascaras predichas)"),
                              (axes[1], "oraculo", "Oraculo (mascaras reales)")):
        m = np.array(resultados[clave]["confusion_test"])
        ax.imshow(m, cmap="Blues", vmin=0, vmax=m.sum(axis=1).max())
        for i in range(len(grupos)):
            for j in range(len(grupos)):
                if m[i, j]:
                    ax.text(j, i, m[i, j], ha="center", va="center", fontsize=11,
                            color="white" if m[i, j] > 5 else TINTA)
        ax.set_xticks(range(len(grupos)), grupos, fontsize=9)
        ax.set_yticks(range(len(grupos)), grupos, fontsize=9)
        ax.set_xlabel("prediccion", fontsize=9, color=TINTA_2)
        ax.set_ylabel("grupo real", fontsize=9, color=TINTA_2)
        ax.set_title(f"{titulo}\nexactitud {resultados[clave]['exactitud_test']:.0%}",
                     fontsize=11, color=TINTA, loc="left")
        for s in ax.spines.values():
            s.set_visible(False)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def riesgo_cobertura(seg: dict, destino: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for split, color, nombre in (("training", AZUL, "training (fuera de pliegue)"), ("testing", NARANJA, "test")):
        rc = seg[split]["riesgo_cobertura"]
        ax.plot(np.array(rc["cobertura"]) * 100, np.array(rc["exactitud"]) * 100, color=color,
                linewidth=2, label=nombre)
        ax.text(101, rc["exactitud"][-1] * 100, nombre, color=TINTA_2, fontsize=9, va="center")
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 20, 40, 60, 80, 100])
    ax.set_xlabel("pacientes conservados, de menor a mayor duda (%)", fontsize=9, color=TINTA_2)
    ax.set_ylabel("exactitud del diagnostico (%)", fontsize=9, color=TINTA_2)
    ax.set_title("Si se revisan primero los casos dudosos, los que quedan aciertan mas",
                 fontsize=11, color=TINTA, loc="left")
    ax.legend(frameon=False, fontsize=9, loc="lower left")
    _estilo(ax)
    fig.tight_layout()
    fig.savefig(destino, dpi=130)
    plt.close(fig)


def segmentaciones(df: pd.DataFrame, mascaras: Path, destino: Path, n: int = 3) -> None:
    te = df[df.split == "testing"].sort_values("duda")
    elegidos = list(te.head(n).itertuples()) + list(te.tail(n).itertuples())
    pacientes = {p.pid: p for p in A.listar_pacientes(split="testing")}
    fig, axes = plt.subplots(3, len(elegidos), figsize=(2.6 * len(elegidos), 8.4))
    for col, fila in enumerate(elegidos):
        img, gt, _ = A.cargar_fase(pacientes[fila.pid], "ED")
        pred = np.load(mascaras / f"{fila.pid}.npz")["ED"]
        z = int(np.argmax((gt == A.MIOCARDIO).sum(axis=(0, 1))))  # corte con mas miocardio
        corte = img[..., z]
        for fila_ax, (mascara, titulo) in enumerate(((None, ""), (gt[..., z], "real"), (pred[z], "prediccion"))):
            ax = axes[fila_ax, col]
            ax.imshow(corte, cmap="gray")
            if mascara is not None:
                capa = np.zeros((*mascara.shape, 4))
                for c, color in COLOR_CLASE.items():
                    capa[mascara == c] = (*matplotlib.colors.to_rgb(color), 0.55)
                ax.imshow(capa)
            ax.axis("off")
            if fila_ax == 0:
                marca = "ALERTA" if fila.alerta else "sin alerta"
                ax.set_title(f"{fila.pid}\nreal {fila.grupo} / pred {fila.pred}\n{marca}",
                             fontsize=9, color=NARANJA if fila.alerta else TINTA)
            else:
                ax.set_title(titulo, fontsize=9, color=TINTA_2)
    handles = [matplotlib.patches.Patch(color=c, label=n) for n, c in
               (("VD", AZUL), ("miocardio", NARANJA), ("VI", AQUA))]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"Test: los {n} pacientes con menos duda (izquierda) y con mas duda (derecha)",
                 x=0.01, ha="left", fontsize=12, color=TINTA)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(destino, dpi=120)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="Figuras de la etapa 2")
    ap.add_argument("--dir", type=str, default="outputs/acdc")
    args = ap.parse_args()
    out = Path(args.dir)
    figs = out / "figuras"
    figs.mkdir(exist_ok=True)

    df = pd.read_csv(out / "pacientes.csv").merge(
        pd.read_csv(out / "seguridad_pacientes.csv")[["pid", "pred", "alerta", "duda"]], on="pid")
    bland_altman(df, figs / "bland_altman.png")
    confusion(json.loads((out / "diagnostico.json").read_text()), figs / "confusion.png")
    riesgo_cobertura(json.loads((out / "seguridad.json").read_text()), figs / "riesgo_cobertura.png")
    segmentaciones(df, out / "mascaras", figs / "segmentaciones_test.png")
    print(f"Figuras en {figs}")


if __name__ == "__main__":
    main()
