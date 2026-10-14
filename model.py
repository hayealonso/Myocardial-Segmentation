"""
Arquitectura U-Net (encoder-decoder con skip connections) para segmentacion
binaria del miocardio a partir de una sola imagen 2D (enfoque 2D, sin usar
cortes vecinos ni la secuencia temporal).

Referencia: Ronneberger et al. "U-Net: Convolutional Networks for Biomedical
Image Segmentation" (2015); Bernard et al. 2018 (revision de metodos ACDC).

El encoder reduce la resolucion espacial (MaxPool) mientras aumenta el
numero de canales, para extraer features cada vez mas abstractos con mayor
campo receptivo. El decoder hace el camino inverso (ConvTranspose2d) y
concatena ("skip connection") los features del encoder de la misma
resolucion, para recuperar detalle espacial fino que se pierde en el
downsampling.

Salida: logits de forma (N, 1, H, W). Se debe aplicar sigmoid + threshold
(o usar BCEWithLogitsLoss directamente, que ya incluye el sigmoid).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x2, sin cambiar la resolucion espacial."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Down(nn.Module):
    """MaxPool(2) seguido de DoubleConv. Reduce H,W a la mitad."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class Up(nn.Module):
    """ConvTranspose2d (duplica H,W) + concatenacion con el skip del encoder
    + DoubleConv."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.upconv = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels // 2 + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upconv(x)
        # por si hay un desfase de 1 pixel (tamanos impares), recortamos/rellenamos
        diff_h = skip.shape[-2] - x.shape[-2]
        diff_w = skip.shape[-1] - x.shape[-1]
        if diff_h != 0 or diff_w != 0:
            x = nn.functional.pad(
                x, [diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2]
            )
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """U-Net configurable en profundidad (numero de canales base).

    base_channels=32 (default) da la arquitectura clasica 32-64-128-256-512.
    Para CPU / prototipado rapido, usar base_channels=16 reduce el numero de
    parametros (~4x menos que con 32) a costa de algo de capacidad.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32):
        super().__init__()
        c = base_channels

        self.in_conv = DoubleConv(in_channels, c)          # H, W
        self.down1 = Down(c, c * 2)                         # H/2
        self.down2 = Down(c * 2, c * 4)                     # H/4
        self.down3 = Down(c * 4, c * 8)                     # H/8
        self.down4 = Down(c * 8, c * 16)                    # H/16 (bottleneck)

        self.up1 = Up(c * 16, skip_channels=c * 8, out_channels=c * 8)
        self.up2 = Up(c * 8, skip_channels=c * 4, out_channels=c * 4)
        self.up3 = Up(c * 4, skip_channels=c * 2, out_channels=c * 2)
        self.up4 = Up(c * 2, skip_channels=c, out_channels=c)

        self.out_conv = nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.in_conv(x)   # (N, c,    H,    W)
        x2 = self.down1(x1)    # (N, 2c,   H/2,  W/2)
        x3 = self.down2(x2)    # (N, 4c,   H/4,  W/4)
        x4 = self.down3(x3)    # (N, 8c,   H/8,  W/8)
        x5 = self.down4(x4)    # (N, 16c,  H/16, W/16)

        x = self.up1(x5, x4)   # (N, 8c,   H/8,  W/8)
        x = self.up2(x, x3)    # (N, 4c,   H/4,  W/4)
        x = self.up3(x, x2)    # (N, 2c,   H/2,  W/2)
        x = self.up4(x, x1)    # (N, c,    H,    W)

        return self.out_conv(x)  # (N, 1, H, W) logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model: UNet, base_channels: int, img_size: int,
                    val_iou: float, stats: dict) -> None:
    """Guarda pesos + todo lo necesario para reproducir la inferencia.

    `stats` ({"mean", "std"} de train) va en el checkpoint porque sin ellos
    las imagenes se normalizan distinto que en el entrenamiento: el modelo
    sigue prediciendo algo, solo que peor, y nada avisa del error.
    """
    torch.save(
        {"model_state": model.state_dict(), "base_channels": int(base_channels),
         "img_size": int(img_size), "val_iou": float(val_iou),
         "mean": float(stats["mean"]), "std": float(stats["std"])},
        path,
    )


def load_checkpoint(path, map_location="cpu"):
    """Devuelve (modelo en modo eval, dict del checkpoint).

    weights_only=True: el checkpoint solo tiene tensores y numeros, asi que no
    hace falta ejecutar pickle arbitrario al cargarlo.
    Los checkpoints anteriores a `save_checkpoint` no traen "mean"/"std".
    """
    checkpoint = torch.load(path, map_location=map_location, weights_only=True)
    model = UNet(in_channels=1, out_channels=1, base_channels=checkpoint["base_channels"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(map_location).eval()
    return model, checkpoint


if __name__ == "__main__":
    # sanity check de shapes, corre sin datos reales
    for base_c in (16, 32):
        for size in (128, 256):
            model = UNet(in_channels=1, out_channels=1, base_channels=base_c)
            dummy = torch.randn(2, 1, size, size)
            out = model(dummy)
            n_params = count_parameters(model)
            assert out.shape == (2, 1, size, size), out.shape
            print(f"base_channels={base_c:3d} img_size={size:3d} -> out {tuple(out.shape)} "
                  f"| params={n_params:,}")
