"""
Metrica (IoU) y funcion de perdida para segmentacion binaria del miocardio.

El dataset esta fuertemente desbalanceado (~1-1.5% de los pixeles son
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


if __name__ == "__main__":
    # sanity checks sin datos reales
    torch.manual_seed(0)

    # caso 1: prediccion perfecta -> IoU debe ser 1.0, dice_loss cercano a 0
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:5, 2:5] = 1.0
    perfect_logits = (target * 2 - 1) * 10  # logits grandes y del signo correcto
    print("IoU (prediccion perfecta):", iou_score(perfect_logits, target).item())
    print("Dice loss (prediccion perfecta):", dice_loss(perfect_logits, target).item())

    # caso 2: prediccion totalmente opuesta -> IoU debe ser 0.0
    wrong_logits = -perfect_logits
    print("IoU (prediccion opuesta):", iou_score(wrong_logits, target).item())

    # caso 3: imagen sin miocardio, prediccion correcta (todo fondo) -> IoU = 1.0
    empty_target = torch.zeros(1, 1, 8, 8)
    empty_logits = torch.full((1, 1, 8, 8), -10.0)
    print("IoU (sin miocardio, bien predicho):", iou_score(empty_logits, empty_target).item())

    loss_fn = BCEDiceLoss()
    print("BCEDiceLoss (prediccion perfecta):", loss_fn(perfect_logits, target).item())
