"""Invariantes de la etapa 2 (ACDC original): mismo criterio que test_miocardio.py.

Solo entran fallas silenciosas: las que no lanzan excepcion y aparecen despues
como "el Dice del VD es raro" o "la FE sale corrida". Los tests `datos` usan
data/ACDC/ y se saltan si no esta.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

import acdc_data as A
from losses import CEDiceLoss, dice_multiclase_loss, dice_y_iou

ROOT = Path(__file__).resolve().parents[1]
ACDC = ROOT / "data" / "ACDC" / "database"
hay_acdc = pytest.mark.skipif(not ACDC.exists(), reason="data/ACDC no esta en disco")


# --------------------------------------------------------------------------
# Preprocesamiento
# --------------------------------------------------------------------------


def test_normalizacion_no_depende_de_la_escala_del_equipo():
    """El mismo corazon guardado como float32 [0, 1] o como int16 [0, 4000]
    tiene que quedar igual: si no, la red ve a cada equipo en otro rango y
    rinde peor en los equipos poco frecuentes, sin ningun error."""
    rng = np.random.default_rng(0)
    vol = rng.gamma(2.0, 1.0, (32, 32, 6)).astype(np.float32)
    a = A.normalizar_volumen(vol)
    b = A.normalizar_volumen(vol * 1000.0 + 5.0)
    assert np.allclose(a, b, atol=1e-4)
    assert abs(a.mean()) < 1e-4 and a.std() == pytest.approx(1.0, abs=1e-3)


def test_un_valor_extremo_no_aplasta_el_contraste():
    """Un pixel saturado (artefacto) no puede definir la escala: con min-max
    o z-score sin recorte, todo el corazon quedaria comprimido cerca de un valor."""
    rng = np.random.default_rng(0)
    vol = rng.normal(100, 10, (32, 32, 4)).astype(np.float32)
    con_artefacto = vol.copy()
    con_artefacto[0, 0, 0] = 1e6
    assert np.corrcoef(A.normalizar_volumen(vol).ravel()[1:],
                       A.normalizar_volumen(con_artefacto).ravel()[1:])[0, 1] > 0.99
    assert A.normalizar_volumen(con_artefacto).std() == pytest.approx(1.0, abs=1e-2)


@pytest.mark.parametrize("factor", [1.37 / 1.5, 1.68 / 1.5])
def test_remuestreo_mantiene_alineadas_imagen_y_mascara(factor):
    """La mascara usa `nearest-exact`: `nearest` toma la esquina del pixel y
    corre la mascara ~0.47 px respecto de la imagen bilinear, siempre hacia el
    mismo lado. El IoU contra esas mascaras baja un poco, y nada lo delata.

    Un solo bloque no distingue sesgo de redondeo; el promedio de muchos si.
    """
    rng = np.random.default_rng(0)
    tam = (round(120 * factor), round(100 * factor))
    yy, xx = np.mgrid[:tam[0], :tam[1]]
    centro = lambda m: np.array([(m * yy).sum(), (m * xx).sum()]) / m.sum()
    desplazamientos = []
    for _ in range(100):
        lab = np.zeros((1, 120, 100), dtype=np.uint8)
        y0, x0, h, w = rng.integers(10, 60), rng.integers(10, 50), rng.integers(10, 40), rng.integers(10, 40)
        lab[0, y0:y0 + h, x0:x0 + w] = 2
        img = A.remuestrear(lab.astype(np.float32), tam, etiqueta=False)[0]
        rem = A.remuestrear(lab, tam, etiqueta=True)[0]
        assert set(np.unique(rem)) <= {0, 2}
        desplazamientos.append(centro(rem == 2) - centro(img))
    sesgo = np.abs(np.mean(desplazamientos, axis=0)).max()
    assert sesgo < 0.15, f"la mascara queda corrida {sesgo:.2f} px respecto de la imagen"


@pytest.mark.parametrize("shape", [(150, 173), (287, 333), (192, 192), (201, 160)])
def test_centrar_y_descentrar_son_inversas(shape):
    """El recorte se deshace exacto; con imagenes mas grandes que 192, lo que
    quedo fuera vuelve como relleno. Un desfase de un pixel entre ambas
    funciones mueve toda la prediccion sin cambiar ninguna forma."""
    rng = np.random.default_rng(0)
    a = rng.normal(size=(3, *shape)).astype(np.float32)
    c = A.centrar(a, A.TAMANO, relleno=-9.0)
    assert c.shape == (3, A.TAMANO, A.TAMANO)
    back = A.descentrar(c, shape, relleno=-9.0)
    dentro = back != -9.0
    assert np.array_equal(back[dentro], a[dentro])
    if max(shape) <= A.TAMANO:
        assert dentro.all()


def test_ida_y_vuelta_a_la_grilla_nativa_conserva_volumenes():
    """Preparar para la red y volver a la grilla nativa no puede cambiar las
    mascaras: los volumenes (y con ellos la FE) se miden despues de esta vuelta.

    Geometria tipica del ACDC: 1.37 mm en el plano, forma no cuadrada.
    """
    yy, xx = np.mgrid[:216, :256]
    d = np.hypot(yy - 100, xx - 140)
    corte = np.select([d < 14, d < 22, (np.abs(yy - 100) < 18) & (np.abs(xx - 105) < 10)],
                      [A.VI, A.MIOCARDIO, A.VD], A.FONDO).astype(np.uint8)
    gt = np.stack([corte] * 5, axis=-1)
    img = gt.astype(np.float32) * 50 + 10
    _, etiquetas, geo = A.preparar_volumen(img, gt, (1.37, 1.37, 10.0))

    onehot = np.eye(A.N_CLASES, dtype=np.float32)[etiquetas].transpose(0, 3, 1, 2)
    vuelta = A.a_nativo(onehot, geo).argmax(1)
    gt_z = np.moveaxis(gt, -1, 0)
    assert vuelta.shape == gt_z.shape
    for c in (A.VD, A.MIOCARDIO, A.VI):
        dice, _ = dice_y_iou(vuelta, gt_z, c)
        assert dice > 0.97, f"clase {c}: dice ida y vuelta {dice:.3f}"
        assert abs((vuelta == c).sum() / (gt_z == c).sum() - 1) < 0.03


def test_fuera_del_recorte_la_prediccion_es_fondo():
    """Una imagen mas grande que el recorte: lo que la red no vio tiene que
    volver como fondo, no como la clase 0 de un one-hot vacio por casualidad."""
    geo = A.Geometria(shape_hw=(300, 300), remuestreado_hw=(300, 300), espaciado=(1.5, 1.5, 10))
    probs = np.zeros((1, A.N_CLASES, A.TAMANO, A.TAMANO), dtype=np.float32)
    probs[:, A.MIOCARDIO] = 1.0
    nativo = A.a_nativo(probs, geo).argmax(1)[0]
    assert (nativo[:40] == A.FONDO).all() and (nativo[150, 150] == A.MIOCARDIO)


# --------------------------------------------------------------------------
# Pliegues y Dataset
# --------------------------------------------------------------------------


def _pacientes_falsos(por_grupo=20):
    return [A.Paciente(f"patient{g}{i:02d}", "training", g, 170.0, 70.0, 1, 10, Path("."))
            for g in A.GRUPOS for i in range(por_grupo)]


def test_pliegues_estratificados_y_reproducibles():
    """Cada pliegue tiene 4 pacientes de cada grupo. Sin estratificar, un
    pliegue puede quedar sin DCM y el clasificador se evalua con una clase
    ausente; con otra semilla en cada corrida, las predicciones fuera de
    pliegue mezclarian modelos que si vieron al paciente."""
    ps = _pacientes_falsos()
    a = A.pliegues_por_paciente(ps, k=5, seed=0)
    assert set(a) == {p.pid for p in ps}
    for f in range(5):
        for g in A.GRUPOS:
            assert sum(1 for p in ps if p.grupo == g and a[p.pid] == f) == 4
    assert a == A.pliegues_por_paciente(ps, k=5, seed=0)
    assert a != A.pliegues_por_paciente(ps, k=5, seed=1)


def test_aumentacion_multiclase_transforma_imagen_y_mascara_igual():
    """Flips y rotaciones identicos en imagen y mascara, y la variacion de
    contraste solo en la imagen. La imagen se construye por bandas de la
    mascara, asi que tras cualquier transformacion se puede recuperar."""
    rng = np.random.default_rng(0)
    lab = rng.integers(0, A.N_CLASES, (1, 12, 12)).astype(np.uint8)
    ds = A.CortesACDC(lab.astype(np.float32) * 2.0, lab, augment=True)
    np.random.seed(0)
    vistas = set()
    for _ in range(80):
        x, y = ds[0]
        assert y.dtype == torch.int64 and x.shape == (1, 12, 12)
        recuperada = np.digitize(x[0].numpy(), [1.0, 3.0, 5.0])
        assert np.array_equal(recuperada, y.numpy()), "imagen y mascara se transformaron distinto"
        vistas.add(y.numpy().tobytes())
    assert len(vistas) == 8


# --------------------------------------------------------------------------
# Perdida y metricas multiclase
# --------------------------------------------------------------------------


def test_dice_multiclase_con_clase_ausente():
    """En cortes apicales el VD no aparece. Prediccion perfecta con una clase
    ausente en todo el batch tiene que dar perdida ~0, no castigar la
    probabilidad residual (con eps=1e-6 en vez de suavizado 1 daba 0.17)."""
    y = torch.zeros(2, 16, 16, dtype=torch.long)
    y[:, 4:12, 4:12] = A.MIOCARDIO
    y[:, 6:10, 6:10] = A.VI  # sin VD en todo el batch
    logits = torch.nn.functional.one_hot(y, A.N_CLASES).permute(0, 3, 1, 2).float() * 20
    assert dice_multiclase_loss(logits, y).item() < 1e-3
    assert CEDiceLoss()(logits, y).item() < 1e-3

    permutado = logits[:, [0, 3, 2, 1]]  # VD <-> VI
    assert dice_multiclase_loss(permutado, y).item() > 0.3


def test_dice_y_iou_valores_conocidos():
    pred = np.array([2, 2, 0, 0, 1])
    real = np.array([2, 0, 2, 0, 1])
    dice, iou = dice_y_iou(pred, real, 2)
    assert dice == pytest.approx(0.5) and iou == pytest.approx(1 / 3)
    assert dice_y_iou(pred, real, 3) == (1.0, 1.0)  # ausente en ambas


# --------------------------------------------------------------------------
# Datos reales
# --------------------------------------------------------------------------


@pytest.mark.datos
@hay_acdc
def test_listado_oficial_completo():
    """100 + 50 pacientes, 20 + 10 por grupo, como documenta el ACDC."""
    ps = A.listar_pacientes(ACDC)
    assert len(ps) == 150
    for split, n in (("training", 20), ("testing", 10)):
        for g in A.GRUPOS:
            assert sum(1 for p in ps if p.split == split and p.grupo == g) == n
    assert all(1.0 < p.superficie_corporal < 3.0 for p in ps)


@pytest.mark.datos
@pytest.mark.slow
@hay_acdc
def test_convencion_de_etiquetas_del_acdc():
    """1 = VD y 3 = VI son cavidades: se achican de ED a ES. 2 = miocardio:
    su volumen casi no cambia (la anotacion varia hasta ~40 %, patient089).
    Un error de deteccion muy claro: la mediana entre pacientes es ~1. Confundir esta convencion con la de los .npy
    (1 = miocardio) entrenaria la red para segmentar el VD como "miocardio"."""
    ps = A.listar_pacientes(ACDC, "training")[::4]
    ratios = []
    for p in ps:
        v = {}
        for fase in A.FASES:
            _, gt, _ = A.cargar_fase(p, fase)
            v[fase] = {c: int((gt == c).sum()) for c in (A.VD, A.MIOCARDIO, A.VI)}
        assert v["ED"][A.VI] > v["ES"][A.VI], p.pid
        assert v["ED"][A.VD] > v["ES"][A.VD], p.pid
        ratios.append(v["ES"][A.MIOCARDIO] / v["ED"][A.MIOCARDIO])
    assert all(0.6 < r < 1.6 for r in ratios)
    assert 0.9 < np.median(ratios) < 1.2


@pytest.mark.datos
@pytest.mark.slow
@hay_acdc
def test_el_recorte_no_corta_ningun_corazon():
    """192 px a 1.5 mm contienen todas las anotaciones del ACDC (el peor caso
    necesita 179). Si el recorte cortara un corazon, el volumen medido saldria
    menor y la FE mal, sin ningun error."""
    for p in A.listar_pacientes(ACDC):
        for fase in A.FASES:
            _, gt, esp = A.cargar_fase(p, fase)
            rem_hw = A._tamano_remuestreado(gt.shape[:2], esp[:2])
            rem = A.remuestrear(np.moveaxis(gt, -1, 0), rem_hw, etiqueta=True)
            recortado = A.centrar(rem, A.TAMANO)
            for c in (A.VD, A.MIOCARDIO, A.VI):
                assert (recortado == c).sum() == (rem == c).sum(), f"{p.pid} {fase} clase {c}"
