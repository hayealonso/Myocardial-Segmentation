"""
Metricas y perdidas de segmentacion: binaria (etapa 1) y multiclase (etapa 2).

En la etapa 1 el dataset esta fuertemente desbalanceado (~1-1.5% de los pixeles son
miocardio, ver EDA_miocardio.ipynb), asi que usar solo binary cross-entropy
tiende a producir un modelo que predice "todo fondo". Por eso se combina:

  - BCEWithLogitsLoss: senal pixel a pixel, estable de optimizar.
  - Dice loss: optimiza directamente el solapamiento entre prediccion y
    mascara real, es mucho menos sensible al desbalance de clases porque
    no promedia sobre todos los pixeles sino sobre la interseccion/union.

loss_total = bce_weight * BCE + dice_weight * DiceLoss
"""

from __future__ import annotations

import torch
import torch.nn as nn


def iou_score(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5, eps: float = 1e-6) -> torch.Tensor:
    """IoU (Jaccard index) promedio del batch, sobre predicciones binarizadas.

    logits:  (N, 1, H, W) salida cruda del modelo (antes de sigmoid).
    targets: (N, 1, H, W) mascara real en {0, 1}.

    Devuelve un tensor escalar con el IoU promediado sobre las N imagenes del
    batch. Si una imagen no tiene miocardio ni prediccion (union == 0), se
    cuenta como IoU = 1.0 (prediccion perfecta de "no hay miocardio").
    """
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = (1, 2, 3)  # todo menos el batch
    intersection = (preds * targets).sum(dim=dims)
    union = (preds + targets).clamp(0, 1).sum(dim=dims)

    iou = torch.where(union > 0, (intersection + eps) / (union + eps), torch.ones_like(union))
    return iou.mean()


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """1 - Dice coefficient (version diferenciable, usa probabilidades no
    binarizadas para poder hacer backprop)."""
    probs = torch.sigmoid(logits)
    dims = (1, 2, 3)
    intersection = (probs * targets).sum(dim=dims)
    denom = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2 * intersection + eps) / (denom + eps)
    return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Loss combinada: bce_weight * BCEWithLogits + dice_weight * DiceLoss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        d_loss = dice_loss(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * d_loss



# Multiclase (etapa 2, ACDC): fondo, VD, miocardio, VI

def dice_multiclase_loss(logits: torch.Tensor, targets: torch.Tensor,
                         ignorar_fondo: bool = True, suavizado: float = 1.0) -> torch.Tensor:
    """1 - Dice soft promedio sobre clases, acumulado sobre TODO el batch.

    logits: (N, C, H, W); targets: (N, H, W) enteros en [0, C).

    Por batch y no por imagen: en cortes apicales y basales falta el VD o el VI, y
    un Dice por imagen castigaria la probabilidad residual de esa clase ausente.

    Suavizado 1 y no un eps diminuto: si la clase falta en todo el batch, eps=1e-6
    da Dice 0.5 aun con la prediccion perfecta.
    """
    n_clases = logits.shape[1]
    probs = torch.softmax(logits, dim=1)
    onehot = torch.nn.functional.one_hot(targets, n_clases).permute(0, 3, 1, 2).to(probs.dtype)
    dims = (0, 2, 3)
    inter = (probs * onehot).sum(dims)
    denom = probs.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + suavizado) / (denom + suavizado)
    if ignorar_fondo:
        dice = dice[1:]
    return 1 - dice.mean()


class CEDiceLoss(nn.Module):
    """ce_weight * CrossEntropy + dice_weight * Dice multiclase (sin fondo)."""

    def __init__(self, ce_weight: float = 1.0, dice_weight: float = 1.0):
        super().__init__()
        self.ce_weight, self.dice_weight = ce_weight, dice_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (self.ce_weight * self.ce(logits, targets)
                + self.dice_weight * dice_multiclase_loss(logits, targets))


def dice_y_iou(pred, target, clase: int):
    """Dice e IoU duros de una clase entre dos mascaras de etiquetas (cualquier forma).

    Si la clase no esta ni en la prediccion ni en la mascara real, ambos valen 1
    (misma convencion que `iou_score`).
    """
    p, t = (pred == clase), (target == clase)
    inter = float((p & t).sum())
    sp, st = float(p.sum()), float(t.sum())
    if sp + st == 0:
        return 1.0, 1.0
    return 2 * inter / (sp + st), inter / (sp + st - inter)

if __name__ == "__main__":
    # sanity checks sin datos reales
    torch.manual_seed(0)

    # prediccion perfecta -> IoU 1.0, dice_loss ~0
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:5, 2:5] = 1.0
    perfect_logits = (target * 2 - 1) * 10  # logits grandes y del signo correcto
    print("IoU (prediccion perfecta):", iou_score(perfect_logits, target).item())
    print("Dice loss (prediccion perfecta):", dice_loss(perfect_logits, target).item())

    # prediccion opuesta -> IoU 0.0
    wrong_logits = -perfect_logits
    print("IoU (prediccion opuesta):", iou_score(wrong_logits, target).item())

    # sin miocardio y bien predicho (todo fondo) -> IoU 1.0
    empty_target = torch.zeros(1, 1, 8, 8)
    empty_logits = torch.full((1, 1, 8, 8), -10.0)
    print("IoU (sin miocardio, bien predicho):", iou_score(empty_logits, empty_target).item())

    loss_fn = BCEDiceLoss()
    print("BCEDiceLoss (prediccion perfecta):", loss_fn(perfect_logits, target).item())
