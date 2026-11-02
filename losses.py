"""
Metricas y perdidas de segmentacion. Todas reciben logits.

El miocardio ocupa cerca del 1 % de los pixeles, asi que la cross-entropy sola
tiende a predecir todo fondo. Por eso se combina con Dice.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def iou_score(logits, targets, threshold=0.5, eps=1e-6):
    """IoU binario promedio por imagen. Sin miocardio real ni predicho, vale 1."""
    preds=(torch.sigmoid(logits)>threshold).float()
    dims=(1, 2, 3)
    intersection=(preds*targets).sum(dim=dims)
    union=(preds+targets).clamp(0, 1).sum(dim=dims)
    iou=torch.where(union>0, (intersection+eps)/(union+eps), torch.ones_like(union))
    return iou.mean()


def dice_loss(logits, targets, eps=1e-6):
    """1 - Dice binario, promedio por imagen. Usa probabilidades para poder derivar."""
    probs=torch.sigmoid(logits)
    dims=(1, 2, 3)
    intersection=(probs*targets).sum(dim=dims)
    denom=probs.sum(dim=dims)+targets.sum(dim=dims)
    dice=(2*intersection+eps)/(denom+eps)
    return 1-dice.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight=bce_weight
        self.dice_weight=dice_weight
        self.bce=nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.bce_weight*self.bce(logits, targets)+self.dice_weight*dice_loss(logits, targets)


def dice_multiclase_loss(logits, targets, ignorar_fondo=True, suavizado=1.0):
    """1 - Dice promedio por clase, acumulado sobre todo el batch.

    logits (N, C, H, W), targets (N, H, W). Se acumula por batch porque en algunos
    cortes falta el VD o el VI. El suavizado de 1 evita castigar una clase ausente
    bien predicha: con 1e-6 el Dice cae a 0.5 aunque la prediccion sea perfecta.
    """
    n_clases=logits.shape[1]
    probs=torch.softmax(logits, dim=1)
    onehot=F.one_hot(targets, n_clases).permute(0, 3, 1, 2).to(probs.dtype)
    dims=(0, 2, 3)
    intersection=(probs*onehot).sum(dims)
    denom=probs.sum(dims)+onehot.sum(dims)
    dice=(2*intersection+suavizado)/(denom+suavizado)
    if ignorar_fondo:
        dice=dice[1:]
    return 1-dice.mean()


class CEDiceLoss(nn.Module):
    def __init__(self, ce_weight=1.0, dice_weight=1.0):
        super().__init__()
        self.ce_weight=ce_weight
        self.dice_weight=dice_weight
        self.ce=nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        return self.ce_weight*self.ce(logits, targets)+self.dice_weight*dice_multiclase_loss(logits, targets)


def dice_y_iou(pred, target, clase):
    """Dice e IoU de una clase entre dos mascaras de etiquetas. Si la clase no esta en ninguna, 1 y 1."""
    pred_clase=pred==clase
    target_clase=target==clase
    intersection=float((pred_clase&target_clase).sum())
    area_pred=float(pred_clase.sum())
    area_target=float(target_clase.sum())
    if area_pred+area_target==0:
        return 1.0, 1.0
    return 2*intersection/(area_pred+area_target), intersection/(area_pred+area_target-intersection)


if __name__=="__main__":
    target=torch.zeros(2, 1, 8, 8)
    target[:, :, 2:5, 2:5]=1.0
    perfect_logits=(target*2-1)*10
    print("IoU perfecto (1.0):", iou_score(perfect_logits, target).item())
    print("IoU opuesto (0.0):", iou_score(-perfect_logits, target).item())
    print("Dice loss perfecto (~0):", dice_loss(perfect_logits, target).item())
