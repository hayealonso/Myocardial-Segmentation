"""Invariantes del pipeline de segmentacion de miocardio.

Cada modulo trae ademas su propia verificacion de shapes en `__main__`
(`uv run python model.py`, `uv run python losses.py`). Lo que hay aca es lo
otro: los errores que no lanzan excepcion ni cambian ninguna dimension, y que
por eso solo aparecen como "el IoU bajo un poco" o "el modelo no mejora".

Ese es el criterio para agregar un test a este archivo: que la falla que detecta
sea silenciosa. Un error de shape ya se manifiesta solo la primera vez que se
ejecuta el script; una mascara rotada distinto que su imagen, no.

    uv run pytest -q                          # todo
    uv run pytest -q -m "not slow"            # sin los que entrenan (~2 s)
    uv run pytest -q -m "not slow and not datos"   # sin datos privados ni checkpoint

Los tests marcados `datos` usan cardiac_data/ y outputs/, que no estan en el
repo (datos privados). Si no estan en disco se saltan, no fallan.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

import train
from cardiac_dataset import (
    MYOCARDIUM_LABEL,
    SPLITS,
    CardiacDataset,
    SplitData,
    _binarize_mask,
    get_dataloaders,
    load_splits,
)
from losses import BCEDiceLoss, dice_loss, iou_score
from model import UNet, load_checkpoint, save_checkpoint

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "cardiac_data"
CKPT_DIR = ROOT / "outputs"


def _escribir_splits(carpeta: Path, n: int = 4, size: int = 16, seed: int = 0) -> None:
    """Splits sinteticos con brillos muy distintos, para que se note si val o
    test se cuelan en los estadisticos de normalizacion."""
    rng = np.random.default_rng(seed)
    for split, brillo in (("train", 60), ("val", 200), ("test", 10)):
        imgs = np.clip(rng.normal(brillo, 20, (n, size, size)), 0, 255).astype(np.uint8)
        annot = rng.integers(0, 3, (n, size, size)).astype(np.uint8)
        np.save(carpeta / f"{split}_imgs.npy", imgs)
        np.save(carpeta / f"{split}_annot.npy", annot)


# --------------------------------------------------------------------------
# Datos
# --------------------------------------------------------------------------


def test_binarizacion_deja_solo_el_miocardio():
    """Clase 1 (miocardio) -> 1; fondo (0) y cavidad del LV (2) -> 0.

    Si la cavidad quedara como positivo, el modelo aprenderia a segmentar el
    ventriculo completo: entrena sin error, el IoU contra esas mascaras sube, y
    el resultado mide otra estructura. data_description.txt fija 1 = miocardio.
    """
    assert MYOCARDIUM_LABEL == 1
    annot = np.array([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)
    out = _binarize_mask(annot)
    assert out.dtype == np.uint8
    assert np.array_equal(out, [[0, 1, 0], [0, 1, 0]])


def test_normalizacion_usa_solo_estadisticos_de_train(tmp_path):
    """Media y std salen de train y se aplican igual a los tres splits.

    Calcularlos sobre todo el conjunto (o por split) no lanza nada: es fuga de
    informacion de test al preprocesamiento, y estandarizar cada split por
    separado borra justamente las diferencias de brillo que el modelo debe
    tolerar.
    """
    _escribir_splits(tmp_path)
    data = load_splits(str(tmp_path))

    train_raw = np.load(tmp_path / "train_imgs.npy").astype(np.float32) / 255.0
    assert data["_stats"]["mean"] == pytest.approx(train_raw.mean(), rel=1e-5)
    assert data["_stats"]["std"] == pytest.approx(train_raw.std(), rel=1e-5)

    assert abs(data["train"].images.mean()) < 1e-4
    assert data["train"].images.std() == pytest.approx(1.0, abs=1e-3)
    assert data["val"].images.mean() > 1.0, "val se estandarizo con sus propios stats"
    assert data["test"].images.mean() < -1.0, "test se estandarizo con sus propios stats"
    for split in SPLITS:
        assert data[split].images.dtype == np.float32

    # cambiar val no puede mover los stats
    np.save(tmp_path / "val_imgs.npy", np.full((4, 16, 16), 255, dtype=np.uint8))
    assert load_splits(str(tmp_path))["_stats"] == data["_stats"]


def test_stats_guardados_reproducen_la_normalizacion(tmp_path):
    """Evaluar test con los stats del checkpoint debe dar exactamente lo mismo
    que recalcularlos desde train: si no, la evaluacion sin train en disco
    normaliza distinto que el entrenamiento y el modelo rinde peor sin avisar."""
    _escribir_splits(tmp_path)
    completo = load_splits(str(tmp_path))
    solo_test = load_splits(str(tmp_path), stats=completo["_stats"], splits=("test",))

    assert set(solo_test) == {"test", "_stats"}
    assert np.array_equal(solo_test["test"].images, completo["test"].images)
    assert np.array_equal(solo_test["test"].masks, completo["test"].masks)

    with pytest.raises(ValueError):
        load_splits(str(tmp_path), splits=("test",))


def test_aumentacion_transforma_imagen_y_mascara_igual():
    """Flips y rotaciones deben aplicarse identicos a imagen y mascara.

    Si la mascara rotara distinto (o no rotara), el modelo entrenaria contra
    etiquetas desplazadas la mitad de las veces: la perdida baja igual, solo que
    menos, y el sintoma parece falta de capacidad.

    La mascara se define como umbral de la imagen, asi que despues de cualquier
    transformacion tiene que seguir siendolo.
    """
    rng = np.random.default_rng(0)
    img = rng.normal(size=(1, 16, 16)).astype(np.float32)
    mask = (img > 0.3).astype(np.uint8)
    ds = CardiacDataset(SplitData(images=img, masks=mask), augment=True)

    np.random.seed(0)
    vistas = set()
    for _ in range(64):
        x, m = ds[0]
        assert torch.equal((x > 0.3).float(), m), "imagen y mascara se transformaron distinto"
        vistas.add(x.numpy().tobytes())
    # flips + rot90 generan las 8 simetrias del cuadrado
    assert len(vistas) == 8, f"la aumentacion no esta activa o esta incompleta: {len(vistas)}"


def test_val_y_test_no_se_aumentan_ni_se_barajan():
    """Solo train lleva aumentacion y shuffle.

    Con val aumentado, el IoU de validacion se vuelve aleatorio entre epocas y
    la seleccion del mejor checkpoint pasa a depender de la suerte; con test
    aumentado, el numero reportado no es reproducible.
    """
    rng = np.random.default_rng(0)
    split = SplitData(
        images=rng.normal(size=(4, 8, 8)).astype(np.float32),
        masks=rng.integers(0, 2, (4, 8, 8)).astype(np.uint8),
    )
    tr, va, te = get_dataloaders({"train": split, "val": split, "test": split}, batch_size=2)

    assert tr.dataset.augment and isinstance(tr.sampler, RandomSampler)
    for loader in (va, te):
        assert not loader.dataset.augment
        assert isinstance(loader.sampler, SequentialSampler)
        a, b = loader.dataset[1], loader.dataset[1]
        assert torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])


@pytest.mark.parametrize("img_size", [128, 100])
def test_resize_mantiene_la_mascara_binaria_y_alineada(img_size):
    """La mascara se redimensiona con `nearest`, la imagen con `bilinear`.

    Con bilinear en la mascara aparecen valores intermedios (0.5) que la BCE
    acepta sin quejarse como si fueran etiquetas blandas. Y si imagen y mascara
    no quedaran alineadas tras el resize, el borde del miocardio quedaria
    corrido un pixel: el IoU baja un poco y nada lo delata.
    """
    img = np.zeros((1, 256, 256), dtype=np.float32)
    img[0, 64:128, 32:96] = 1.0
    mask = (img > 0.5).astype(np.uint8)
    ds = CardiacDataset(SplitData(images=img, masks=mask), img_size=img_size)
    x, m = ds[0]

    assert x.shape == m.shape == (1, img_size, img_size)
    assert set(torch.unique(m).tolist()) <= {0.0, 1.0}
    pred = (x > 0.5).float()
    iou = (pred * m).sum() / ((pred + m).clamp(0, 1)).sum()
    assert iou > 0.95, f"imagen y mascara desalineadas tras el resize (IoU={iou:.3f})"


# --------------------------------------------------------------------------
# Metrica y perdida
# --------------------------------------------------------------------------


def test_iou_casos_limite():
    """Perfecto = 1, opuesto = 0, sin miocardio y bien predicho = 1, sin
    miocardio pero con falsos positivos = 0."""
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:5, 2:5] = 1.0
    perfect = (target * 2 - 1) * 10
    assert iou_score(perfect, target).item() == pytest.approx(1.0)
    assert iou_score(-perfect, target).item() == pytest.approx(0.0, abs=1e-6)

    vacio = torch.zeros(1, 1, 8, 8)
    assert iou_score(torch.full((1, 1, 8, 8), -10.0), vacio).item() == 1.0
    assert iou_score(torch.full((1, 1, 8, 8), 10.0), vacio).item() == pytest.approx(0.0, abs=1e-6)


def test_iou_valor_parcial_conocido():
    """2 pixeles predichos, 3 reales, 1 en comun: IoU = 1 / (2 + 3 - 1) = 0.25."""
    target = torch.zeros(1, 1, 4, 4)
    target[0, 0, 0, :3] = 1.0
    logits = torch.full((1, 1, 4, 4), -10.0)
    logits[0, 0, 0, 2:4] = 10.0
    assert iou_score(logits, target).item() == pytest.approx(0.25, abs=1e-5)


def test_iou_promedia_por_imagen_y_no_por_pixel():
    """El IoU reportado es el promedio de los IoU de cada imagen.

    Agregar intersecciones y uniones de todo el batch antes de dividir no falla,
    pero premia las imagenes con miocardio grande y esconde los cortes apicales
    chicos, que son justamente los dificiles. Tampoco seria comparable con lo
    ya reportado.
    """
    target = torch.zeros(2, 1, 16, 16)
    target[0, 0, :12, :12] = 1.0  # miocardio grande, bien predicho
    target[1, 0, 0, 0] = 1.0      # miocardio de un pixel, no detectado
    logits = torch.where(target > 0, 10.0, -10.0)
    logits[1] = -10.0
    assert iou_score(logits, target).item() == pytest.approx(0.5, abs=1e-5)


def test_dice_promedia_por_imagen():
    """Mismo criterio que el IoU: la perdida no puede ignorar las imagenes con
    poco miocardio por quedar diluidas en el total de pixeles del batch."""
    target = torch.zeros(2, 1, 16, 16)
    target[0, 0, :12, :12] = 1.0
    target[1, 0, 0, 0] = 1.0
    logits = torch.where(target > 0, 20.0, -20.0)
    logits[1] = -20.0
    assert dice_loss(logits, target).item() == pytest.approx(0.5, abs=1e-3)


def test_perdida_estable_con_logits_extremos():
    """La perdida recibe logits y debe tolerar valores enormes sin NaN.

    Un sigmoid seguido de BCELoss satura a logits grandes y el gradiente se
    anula o se vuelve NaN: el entrenamiento se detiene sin error. Por eso la
    U-Net no termina en sigmoid y la perdida usa BCEWithLogitsLoss.
    """
    target = torch.zeros(2, 1, 8, 8)
    target[:, :, 2:5, 2:5] = 1.0
    logits = ((target * 2 - 1) * 1e4).requires_grad_()
    loss = BCEDiceLoss()(logits, target)
    loss.backward()
    assert torch.isfinite(loss) and loss.item() < 1e-3
    assert torch.isfinite(logits.grad).all()

    malo = BCEDiceLoss()(-(target * 2 - 1) * 1e4, target)
    assert torch.isfinite(malo) and malo.item() > 1.0


# --------------------------------------------------------------------------
# Modelo
# --------------------------------------------------------------------------


@pytest.mark.parametrize("size", [64, 75, 100, 128])
def test_unet_preserva_el_tamano(size):
    """75 y 100 no son multiplos de 16: Up rellena el desfase de cada nivel.

    Si la salida saliera de otro tamano, el resize posterior o el broadcasting
    podrian dejarlo pasar en algunas operaciones."""
    net = UNet(base_channels=4).eval()
    with torch.no_grad():
        out = net(torch.randn(2, 1, size, size))
    assert out.shape == (2, 1, size, size)


def test_unet_devuelve_logits():
    """La ultima capa es una Conv2d sin activacion.

    `iou_score` y `BCEDiceLoss` aplican sigmoid adentro. Si la red ya terminara
    en sigmoid, se aplicaria dos veces: todo pixel quedaria con probabilidad
    > 0.5 y la mascara predicha seria la imagen completa.
    """
    torch.manual_seed(0)
    net = UNet(base_channels=4).eval()
    assert isinstance(net.out_conv, nn.Conv2d)
    with torch.no_grad():
        net.out_conv.bias.fill_(-5.0)  # sin activacion final, la salida tiene que quedar negativa
        out = net(torch.randn(2, 1, 32, 32))
    assert (out < 0).all(), "la salida no puede ser negativa: hay una activacion despues de out_conv"


def test_checkpoint_ida_y_vuelta(tmp_path):
    """Lo guardado tiene que reproducir el modelo exacto, en modo eval y con los
    stats de normalizacion.

    En modo train, BatchNorm usa la estadistica del batch: la prediccion de una
    imagen depende de las otras imagenes del batch, y la figura de predicciones
    cambia segun cuales se elijan.
    """
    torch.manual_seed(0)
    model = UNet(base_channels=4)
    model.train()
    model(torch.randn(4, 1, 32, 32))  # mueve las running stats de BatchNorm
    model.eval()

    path = tmp_path / "m.pt"
    save_checkpoint(path, model, 4, 32, 0.5, {"mean": np.float32(0.13), "std": 0.2})
    cargado, ck = load_checkpoint(path)

    assert not cargado.training
    assert isinstance(ck["mean"], float) and ck["mean"] == pytest.approx(0.13)
    assert (ck["base_channels"], ck["img_size"], ck["std"]) == (4, 32, 0.2)

    x = torch.randn(3, 1, 32, 32)
    with torch.no_grad():
        assert torch.allclose(cargado(x), model(x), atol=1e-6)
        assert torch.allclose(cargado(x[:1]), cargado(x)[:1], atol=1e-5), (
            "la prediccion depende del batch"
        )


# --------------------------------------------------------------------------
# Entrenamiento
# --------------------------------------------------------------------------


def test_run_epoch_promedia_por_muestra():
    """Con un ultimo batch mas chico, el promedio se pondera por muestra.

    Promediar los promedios de cada batch no falla, pero le da al ultimo batch
    incompleto el mismo peso que a uno lleno y el IoU reportado cambia con el
    batch size. Aca: IoU por muestra [0, 0, 1] -> 1/3; promediando por batch
    daria (0 + 1) / 2, y ponderando por batch_size en vez del tamano real, 2/3.
    """
    mask = np.zeros((3, 8, 8), dtype=np.uint8)
    mask[:, 2:6, 2:6] = 1
    imgs = np.full((3, 8, 8), -5.0, dtype=np.float32)  # no predice nada -> IoU 0
    imgs[2] = mask[2] * 10.0 - 5.0  # la muestra del batch incompleto: perfecta

    identidad = nn.Conv2d(1, 1, kernel_size=1)
    with torch.no_grad():
        identidad.weight.fill_(1.0)
        identidad.bias.zero_()

    loader = DataLoader(CardiacDataset(SplitData(images=imgs, masks=mask)), batch_size=2)
    _, iou = train.run_epoch(identidad, loader, BCEDiceLoss(), "cpu")
    assert iou == pytest.approx(1 / 3, abs=1e-5)


def test_evaluar_no_modifica_el_modelo():
    """run_epoch sin optimizador no puede tocar ni pesos ni running stats.

    Si la evaluacion quedara en modo train, BatchNorm actualizaria sus running
    stats con val y test en cada epoca: no hay gradiente ni error, pero el
    modelo guardado ya habria "visto" los datos de evaluacion.
    """
    torch.manual_seed(0)
    model = UNet(base_channels=4)
    rng = np.random.default_rng(0)
    split = SplitData(
        images=rng.normal(size=(4, 32, 32)).astype(np.float32),
        masks=rng.integers(0, 2, (4, 32, 32)).astype(np.uint8),
    )
    antes = {k: v.clone() for k, v in model.state_dict().items()}
    train.run_epoch(model, DataLoader(CardiacDataset(split), batch_size=2), BCEDiceLoss(), "cpu")
    for k, v in model.state_dict().items():
        assert torch.equal(v, antes[k]), f"la evaluacion modifico {k}"


def test_scheduler_maximiza_el_iou():
    """ReduceLROnPlateau observa el IoU de val, que se maximiza.

    Con mode="min" el lr se reduciria cada vez que el modelo mejora: entrena
    sin error, solo que se estanca antes. Mientras el IoU sube el lr no se toca;
    cuando se estanca, baja.
    """
    opt, sch = train.build_optimizer(nn.Linear(1, 1), lr=1e-3)
    for v in np.linspace(0.1, 0.9, 10):
        sch.step(float(v))
    assert opt.param_groups[0]["lr"] == pytest.approx(1e-3), "redujo el lr mientras mejoraba"
    for _ in range(5):
        sch.step(0.9)
    assert opt.param_groups[0]["lr"] < 1e-3, "no redujo el lr al estancarse"


def _anillos(n: int, size: int = 64, seed: int = 0):
    """Anillos ruidosos alrededor de una cavidad brillante: la geometria del
    miocardio en un corte de eje corto, sin datos privados."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:size, :size]
    imgs = np.empty((n, size, size), dtype=np.float32)
    masks = np.empty((n, size, size), dtype=np.uint8)
    for i in range(n):
        cy, cx = rng.uniform(20, size - 20, 2)
        r_in, grosor = rng.uniform(5, 9), rng.uniform(3, 6)
        d = np.hypot(yy - cy, xx - cx)
        anillo = (d >= r_in) & (d < r_in + grosor)
        img = 0.2 + 0.8 * (d < r_in) + 0.5 * anillo + rng.normal(0, 0.15, (size, size))
        imgs[i], masks[i] = (img - 0.4) / 0.4, anillo
    return SplitData(images=imgs, masks=masks)


@pytest.mark.slow
def test_el_pipeline_aprende_a_segmentar_anillos():
    """Extremo a extremo: dataset + aumentacion + U-Net + perdida + optimizador.

    Si alguna pieza estuviera rota en silencio (mascara desalineada, logits con
    signo invertido, scheduler mal orientado), el modelo no llegaria a segmentar
    un anillo sintetico casi trivial.
    """
    torch.manual_seed(0)
    np.random.seed(0)
    tr = DataLoader(CardiacDataset(_anillos(64, seed=0), augment=True), batch_size=8, shuffle=True)
    va = DataLoader(CardiacDataset(_anillos(16, seed=1)), batch_size=8)

    model = UNet(base_channels=8)
    opt, _ = train.build_optimizer(model, lr=3e-3)
    loss_fn = BCEDiceLoss()
    for _ in range(12):
        train.run_epoch(model, tr, loss_fn, "cpu", opt)
    _, val_iou = train.run_epoch(model, va, loss_fn, "cpu")
    assert val_iou > 0.7, f"no aprendio a segmentar anillos sinteticos: IoU={val_iou:.3f}"


# --------------------------------------------------------------------------
# Datos reales y checkpoint (se saltan si no estan en disco)
# --------------------------------------------------------------------------


def _splits_en_disco():
    return [
        s for s in SPLITS
        if (DATA / f"{s}_imgs.npy").exists() and (DATA / f"{s}_annot.npy").exists()
    ]


@pytest.mark.datos
@pytest.mark.parametrize("split", SPLITS)
def test_datos_reales_tienen_el_formato_documentado(split):
    """Lo que dice data_description.txt: (N, 256, 256) uint8, mascara en {0, 1, 2}.

    Una version de los datos con otra codificacion (por ejemplo 0/255, o la
    cavidad como 1) se binarizaria sin error y entrenaria contra otra cosa.
    """
    if split not in _splits_en_disco():
        pytest.skip(f"{split} no esta en {DATA}")
    imgs = np.load(DATA / f"{split}_imgs.npy", mmap_mode="r")
    annot = np.load(DATA / f"{split}_annot.npy", mmap_mode="r")

    assert imgs.shape == annot.shape and imgs.shape[1:] == (256, 256)
    assert imgs.dtype == np.uint8 and annot.dtype == np.uint8
    conteo = np.bincount(np.asarray(annot).ravel(), minlength=256)
    assert conteo[3:].sum() == 0, "la mascara tiene clases fuera de {0, 1, 2}"
    por_imagen = (np.asarray(annot) == MYOCARDIUM_LABEL).reshape(len(annot), -1).sum(1)
    assert (por_imagen > 0).all(), "hay imagenes sin miocardio (el EDA no encontro ninguna)"


@pytest.mark.datos
def test_no_hay_imagenes_repetidas_entre_splits():
    """data_description.txt asegura que los splits no comparten sujetos.

    No hay ids de sujeto para verificarlo, pero una imagen identica en dos
    splits es fuga segura: el IoU de test sale optimista y no hay error.
    """
    presentes = _splits_en_disco()
    if len(presentes) < 2:
        pytest.skip("hacen falta al menos dos splits en disco")
    hashes = {}
    for split in presentes:
        imgs = np.load(DATA / f"{split}_imgs.npy", mmap_mode="r")
        hashes[split] = {hashlib.sha1(np.ascontiguousarray(im).tobytes()).hexdigest() for im in imgs}
    for i, a in enumerate(presentes):
        for b in presentes[i + 1:]:
            comunes = hashes[a] & hashes[b]
            assert not comunes, f"{len(comunes)} imagenes repetidas entre {a} y {b}"


@pytest.mark.datos
def test_checkpoint_coincide_con_los_resultados_guardados():
    """best_model.pt, history.json y test_results.json deben ser de la misma corrida.

    Un checkpoint de otra corrida (u otro img_size) se carga sin error y produce
    numeros que no coinciden con los que se reportan.
    """
    ck_path, hist_path, res_path = (
        CKPT_DIR / "best_model.pt", CKPT_DIR / "history.json", CKPT_DIR / "test_results.json"
    )
    if not (ck_path.exists() and hist_path.exists() and res_path.exists()):
        pytest.skip("faltan checkpoint o resultados en outputs/")
    _, ck = load_checkpoint(ck_path)
    history = json.loads(hist_path.read_text())
    results = json.loads(res_path.read_text())

    assert ck["val_iou"] == pytest.approx(max(h["val_iou"] for h in history))
    assert ck["val_iou"] == pytest.approx(results["best_val_iou"])
    assert (ck["img_size"], ck["base_channels"]) == (results["img_size"], results["base_channels"])
    assert "mean" in ck and "std" in ck, "checkpoint sin stats de normalizacion"


@pytest.mark.datos
@pytest.mark.slow
def test_checkpoint_reproduce_su_iou_de_validacion():
    """El preprocesamiento actual tiene que reproducir exacto el val_iou guardado.

    Es el test de regresion de todo el pipeline de inferencia: cualquier cambio
    en normalizacion, binarizacion, resize, umbral o metrica mueve este numero,
    aunque cada pieza por separado siga pareciendo correcta.
    """
    ck_path = CKPT_DIR / "best_model.pt"
    if "val" not in _splits_en_disco() or not ck_path.exists():
        pytest.skip("faltan cardiac_data/val_* o outputs/best_model.pt")
    model, ck = load_checkpoint(ck_path)
    if "mean" not in ck:
        pytest.skip("checkpoint antiguo sin stats de normalizacion")

    data = load_splits(str(DATA), stats={"mean": ck["mean"], "std": ck["std"]}, splits=("val",))
    loader = DataLoader(CardiacDataset(data["val"], img_size=ck["img_size"]), batch_size=16)
    _, val_iou = train.run_epoch(model, loader, BCEDiceLoss(), "cpu")
    assert val_iou == pytest.approx(ck["val_iou"], abs=1e-6)


# --------------------------------------------------------------------------
# Visualizacion
# --------------------------------------------------------------------------


def test_mapa_de_errores_colorea_cada_caso_y_deja_el_fondo_transparente():
    """Verde = acierto, rojo = falso positivo, azul = falso negativo.

    Con los colores intercambiados la figura se ve igual de razonable, pero se
    lee al reves: un modelo que sobresegmenta parece uno que se queda corto.
    Y si los verdaderos negativos no fueran transparentes, la capa oscurece
    toda la resonancia y no solo los errores.
    """
    from visualize_predictions import (
        COLOR_FN, COLOR_FP, COLOR_REAL, COLOR_TP, capa_de_mascara, mapa_de_errores,
    )

    mask = np.array([[1, 0], [1, 0]])
    pred = np.array([[1, 1], [0, 0]])
    capa = mapa_de_errores(mask, pred)

    assert capa.shape == (2, 2, 4)
    assert tuple(capa[0, 0]) == pytest.approx(COLOR_TP)  # real 1, predicho 1
    assert tuple(capa[0, 1]) == pytest.approx(COLOR_FP)  # real 0, predicho 1
    assert tuple(capa[1, 0]) == pytest.approx(COLOR_FN)  # real 1, predicho 0
    assert capa[1, 1, 3] == 0.0, "los verdaderos negativos deben ser transparentes"
    assert COLOR_TP[1] == 1.0 and COLOR_FP[0] == 1.0 and COLOR_FN[2] == 1.0

    # una mascara de un solo valor tiene que verse con su color, no casi blanca
    capa = capa_de_mascara(mask, COLOR_REAL)
    assert tuple(capa[0, 0]) == pytest.approx(COLOR_REAL)
    assert capa[0, 1, 3] == 0.0
