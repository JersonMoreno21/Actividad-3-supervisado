"""Tests del dataset supervisado generado con Dijkstra como profesor."""

import os

import pytest

from mio_router.graph import dijkstra

from supervised.dataset import (
    CANDIDATAS,
    DATASET_CSV,
    COLUMNA_DESTINO,
    COLUMNA_ORIGEN,
    COLUMNA_TIEMPO,
    COLUMNA_TRANSBORDOS,
    cargar_dataset,
    contar_transbordos,
    grafo_estaciones,
    pares_od,
    preparar,
)

pytestmark = pytest.mark.skipif(
    not os.path.exists(DATASET_CSV),
    reason="dataset no generado (python -m supervised generar)",
)


def test_dataset_columnas_y_tamano():
    df = cargar_dataset()
    assert list(df.columns) == CANDIDATAS
    # 81 estaciones + 9 nodos zona, pares ordenados sin auto-pares
    assert len(df) == 90 * 89
    assert df[COLUMNA_ORIGEN].nunique() == 90
    assert df[COLUMNA_DESTINO].nunique() == 90


def test_etiquetas_validas():
    df = cargar_dataset()
    assert (df[COLUMNA_TIEMPO] > 0).all()
    assert (df[COLUMNA_TRANSBORDOS] >= 0).all()
    assert df[COLUMNA_TIEMPO].dtype.kind in "iu"
    assert df[COLUMNA_TRANSBORDOS].dtype.kind in "iu"
    # ningún par consigo mismo
    assert (df[COLUMNA_ORIGEN] != df[COLUMNA_DESTINO]).all()


def test_pares_od_completos():
    _, _, meta = preparar()
    pares = pares_od(meta)
    assert len(pares) == 90 * 89
    assert len(set(pares)) == len(pares)
    assert ("univalle", "chiminangos") in pares


def test_grafo_estaciones_sin_paradas():
    relations, g, meta = preparar()
    assert set(g.adj) <= set(relations["estaciones"])
    assert "univalle" in g.adj
    # sin aristas de última milla (paradas externas)
    codigos = {e.codigo_ruta for aristas in g.adj.values() for e in aristas}
    assert "caminata" not in codigos
    assert len(meta.estaciones) == 90


def test_contar_transbordos_sintetico():
    tramos = [
        ("a", "calle_5", 2),
        ("b", "calle_5", 2),
        ("c", "transbordo", 4),
        ("d", "carrera_1", 3),
    ]
    assert contar_transbordos(tramos) == 2


def test_contar_transbordos_igual_que_cli():
    from mio_router.cli import _contar_transbordos

    _, g, meta = preparar()
    for a, b in [("univalle", "chiminangos"), ("capri", "unidad_deportiva"),
                 ("zona_universidades", "nuevo_latir")]:
        res = dijkstra(g, a, b)
        assert res is not None, f"sin ruta entre {a} y {b}"
        _, tramos = res
        assert contar_transbordos(tramos) == _contar_transbordos(tramos)


def test_un_par_recalculado_coincide_con_el_csv():
    df = cargar_dataset()
    _, g, _ = preparar()
    fila = df[df[COLUMNA_ORIGEN] == "univalle"].iloc[0]
    tiempo, tramos = dijkstra(g, fila[COLUMNA_ORIGEN], fila[COLUMNA_DESTINO])
    assert tiempo == fila[COLUMNA_TIEMPO]
    assert contar_transbordos(tramos) == fila[COLUMNA_TRANSBORDOS]
