"""
U-Net 2D (Ronneberger et al., 2015). Predice a partir de una sola imagen, sin
cortes vecinos ni secuencia temporal.

Etapa 1: 1 salida (miocardio). Etapa 2: 4 salidas (fondo, VD, miocardio, VI).
Devuelve logits; la perdida y la metrica aplican la sigmoid o la softmax.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) x2. No cambia H ni W."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block=nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Down(nn.Module):
    """MaxPool + DoubleConv. Reduce H y W a la mitad."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block=nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.block(x)


class Up(nn.Module):
    """ConvTranspose2d + concatenacion con el skip del encoder + DoubleConv."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upconv=nn.ConvTranspose2d(in_channels, in_channels//2, kernel_size=2, stride=2)
        self.conv=DoubleConv(in_channels//2+skip_channels, out_channels)

    def forward(self, x, skip):
        x=self.upconv(x)
        #con tamanos que no son multiplo de 16 puede faltar un pixel
        diff_h=skip.shape[-2]-x.shape[-2]
        diff_w=skip.shape[-1]-x.shape[-1]
        x=F.pad(x, [diff_w//2, diff_w-diff_w//2, diff_h//2, diff_h-diff_h//2])
        x=torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=32):
        super().__init__()
        c=base_channels
        self.in_conv=DoubleConv(in_channels, c)
        self.down1=Down(c, c*2)
        self.down2=Down(c*2, c*4)
        self.down3=Down(c*4, c*8)
        self.down4=Down(c*8, c*16)
        self.up1=Up(c*16, skip_channels=c*8, out_channels=c*8)
        self.up2=Up(c*8, skip_channels=c*4, out_channels=c*4)
        self.up3=Up(c*4, skip_channels=c*2, out_channels=c*2)
        self.up4=Up(c*2, skip_channels=c, out_channels=c)
        self.out_conv=nn.Conv2d(c, out_channels, kernel_size=1)

    def forward(self, x):
        x1=self.in_conv(x)   #(N, c,   H,    W)
        x2=self.down1(x1)    #(N, 2c,  H/2,  W/2)
        x3=self.down2(x2)    #(N, 4c,  H/4,  W/4)
        x4=self.down3(x3)    #(N, 8c,  H/8,  W/8)
        x5=self.down4(x4)    #(N, 16c, H/16, W/16)

        x=self.up1(x5, x4)
        x=self.up2(x, x3)
        x=self.up3(x, x2)
        x=self.up4(x, x1)    #(N, c,   H,    W)
        return self.out_conv(x)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(path, model, base_channels, img_size, val_iou, stats):
    """Guarda los pesos y lo necesario para repetir la inferencia.

    Sin la media y el std de train las imagenes se normalizarian distinto.
    """
    torch.save(
        {"model_state": model.state_dict(), "base_channels": int(base_channels),
         "img_size": int(img_size), "val_iou": float(val_iou),
         "mean": float(stats["mean"]), "std": float(stats["std"])},
        path,
    )


def load_checkpoint(path, map_location="cpu"):
    """Devuelve (modelo en modo eval, checkpoint). Sirve para las dos etapas."""
    checkpoint=torch.load(path, map_location=map_location, weights_only=True)
    model=UNet(in_channels=1, out_channels=checkpoint.get("n_clases", 1),
               base_channels=checkpoint["base_channels"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(map_location)
    model.eval()
    return model, checkpoint


if __name__=="__main__":
    for base_channels in (16, 32):
        for size in (128, 256):
            model=UNet(in_channels=1, out_channels=1, base_channels=base_channels)
            out=model(torch.randn(2, 1, size, size))
            assert out.shape==(2, 1, size, size), out.shape
            print(f"base_channels={base_channels} size={size}: params={count_parameters(model):,}")
