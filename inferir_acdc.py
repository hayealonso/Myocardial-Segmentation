"""
Segmenta a los 150 pacientes del ACDC, vuelve a la grilla nativa y mide biomarcadores.

    python inferir_acdc.py  #despues de entrenar los 5 pliegues

Cada paciente de training lo segmenta el modelo que no lo vio. Los de test, los 5
modelos promediados. Siempre se promedian las 8 simetrias del cuadrado (TTA), y la
dispersion de los biomarcadores entre esas vistas es la incertidumbre del paciente.

En test esa dispersion se calcula por modelo y se promedia. El promedio recorta las
colas, asi que el umbral calibrado en training tiende a marcar menos pacientes en test.

Salidas en outputs/acdc/: pacientes.csv y mascaras/<pid>.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

import acdc_data as A
from biomarcadores import NOMBRES, biomarcadores
from losses import dice_y_iou
from train_acdc import cargar_modelo

CLASES={"vd": A.VD, "mio": A.MIOCARDIO, "vi": A.VI}


def vistas_d4(x):
    """Las 8 simetrias del cuadrado sobre los dos ultimos ejes: lista de (k, flip, tensor)."""
    vistas=[]
    for flip in (False, True):
        x_flip=torch.flip(x, dims=(-1,)) if flip else x
        for k in range(4):
            vistas.append((k, flip, torch.rot90(x_flip, k, dims=(-2, -1))))
    return vistas


def deshacer_vista(y, k, flip):
    """Primero deshace la rotacion y despues el flip. En otro orden las vistas quedan desalineadas."""
    y=torch.rot90(y, -k, dims=(-2, -1))
    if flip:
        y=torch.flip(y, dims=(-1,))
    return y


@torch.no_grad()
def probabilidades_tta(model, cortes, device):
    """(Z, H, W) -> (vistas, Z, clases, H, W)."""
    x=torch.from_numpy(cortes).unsqueeze(1).to(device)
    salida=[]
    for k, flip, x_vista in vistas_d4(x):
        probs=torch.softmax(model(x_vista).float(), dim=1)
        salida.append(deshacer_vista(probs, k, flip).cpu().numpy())
    return np.stack(salida)


def mascara_nativa(probs, geometria):
    """(Z, clases, TAMANO, TAMANO) -> etiquetas (Z, H, W) en la grilla nativa."""
    return A.a_nativo(probs, geometria).argmax(1).astype(np.uint8)


def procesar_paciente(paciente, modelos, device):
    """Devuelve (fila para el csv, mascaras finales por fase)."""
    sc=paciente.superficie_corporal
    fila={"pid": paciente.pid, "split": paciente.split, "grupo": paciente.grupo, "sc": sc}
    finales, gts, geometrias={}, {}, {}
    mascaras_por_vista={}  #mascaras_por_vista[fase][modelo][vista]

    for fase in A.FASES:
        img, gt, espaciado=A.cargar_fase(paciente, fase)
        cortes, _, geometria=A.preparar_volumen(img, None, espaciado)
        probs_por_modelo=[probabilidades_tta(model, cortes, device) for model in modelos]

        promedio=np.mean([probs.mean(axis=0) for probs in probs_por_modelo], axis=0)
        finales[fase]=mascara_nativa(promedio, geometria)
        gts[fase]=np.moveaxis(gt, -1, 0)
        geometrias[fase]=geometria

        mascaras_por_vista[fase]=[]
        for probs in probs_por_modelo:
            mascaras_por_vista[fase].append([mascara_nativa(probs_vista, geometria) for probs_vista in probs])

    espaciado=geometrias["ED"].espaciado
    pred=biomarcadores(finales["ED"], finales["ES"], espaciado, sc)
    real=biomarcadores(gts["ED"], gts["ES"], espaciado, sc)
    for nombre in NOMBRES:
        fila[f"pred_{nombre}"]=pred[nombre]
    for nombre in NOMBRES:
        fila[f"real_{nombre}"]=real[nombre]

    #incertidumbre: desviacion entre vistas de cada modelo, promediada entre modelos
    stds={nombre: [] for nombre in NOMBRES}
    for i_modelo in range(len(modelos)):
        por_vista=[]
        for fd, fs in zip(mascaras_por_vista["ED"][i_modelo], mascaras_por_vista["ES"][i_modelo]):
            por_vista.append(biomarcadores(fd, fs, espaciado, sc))
        for nombre in NOMBRES:
            stds[nombre].append(np.nanstd([vista[nombre] for vista in por_vista]))
    for nombre in NOMBRES:
        fila[f"std_{nombre}"]=float(np.mean(stds[nombre]))

    for fase in A.FASES:
        for nombre, clase in CLASES.items():
            dice, iou=dice_y_iou(finales[fase], gts[fase], clase)
            fila[f"dice_{nombre}_{fase}"]=dice
            fila[f"iou_{nombre}_{fase}"]=iou
        #metrica del curso: IoU 2D del miocardio, promedio por corte
        ious=[dice_y_iou(pred_corte, gt_corte, A.MIOCARDIO)[1]
              for pred_corte, gt_corte in zip(finales[fase], gts[fase])]
        fila[f"iou2d_mio_{fase}"]=float(np.mean(ious))
    return fila, finales


def main():
    parser=argparse.ArgumentParser(description="Inferencia del ACDC con TTA y biomarcadores")
    parser.add_argument("--modelos", default="outputs/acdc")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--out", default="outputs/acdc")
    args=parser.parse_args()
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelos=[]
    for fold in range(args.k):
        model, checkpoint=cargar_modelo(Path(args.modelos)/f"fold{fold}"/"model.pt", device)
        #otro k u otro pliegue le daria a un paciente un modelo que si lo vio
        if checkpoint["fold"]!=fold or checkpoint["k"]!=args.k:
            raise ValueError(f"fold{fold}/model.pt es del pliegue {checkpoint['fold']} de {checkpoint['k']}")
        modelos.append(model)

    pliegues=A.pliegues_por_paciente(A.listar_pacientes(split="training"), k=args.k)
    out=Path(args.out)
    (out/"mascaras").mkdir(parents=True, exist_ok=True)

    filas=[]
    for paciente in A.listar_pacientes():
        if paciente.split=="training":
            fold=pliegues[paciente.pid]
            fila, finales=procesar_paciente(paciente, [modelos[fold]], device)
        else:
            fold=-1
            fila, finales=procesar_paciente(paciente, modelos, device)
        fila["fold"]=fold
        np.savez_compressed(out/"mascaras"/f"{paciente.pid}.npz", ED=finales["ED"], ES=finales["ES"])
        filas.append(fila)
        print(f"{paciente.pid} {paciente.split:8s} {paciente.grupo:4s} "
              f"dice mio={fila['dice_mio_ED']:.3f}/{fila['dice_mio_ES']:.3f} "
              f"FE VI pred={fila['pred_fe_vi']:5.1f} real={fila['real_fe_vi']:5.1f} (+-{fila['std_fe_vi']:.1f})")

    pd.DataFrame(filas).to_csv(out/"pacientes.csv", index=False)
    print(f"Guardado {out/'pacientes.csv'} ({len(filas)} pacientes)")


if __name__=="__main__":
    main()
