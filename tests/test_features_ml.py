"""Tests de las features (sin fuga de etiquetas)."""

import math

import pytest

from supervised.dataset import COLUMNA_TIEMPO, COLUMNA_TRANSBORDOS, preparar
from supervised.features import FEATURES, construir_meta, vector_features


@pytest.fixture(scope="module")
def ctx():
    relations, g, meta = preparar()
    return relations, g, meta


def test_numero_de_features(ctx):
    _, _, meta = ctx
    x = vector_features("univalle", "chiminangos", meta)
    assert len(FEATURES) == 18
    assert len(x) == len(FEATURES)
    assert all(isinstance(v, float) for v in x)
    assert all(math.isfinite(v) for v in x)


def test_features_sin_etiquetas(ctx):
    prohibidas = {COLUMNA_TIEMPO, COLUMNA_TRANSBORDOS, "origen", "destino"}
    assert not (set(FEATURES) & prohibidas)


def test_distancia_y_deltas(ctx):
    _, _, meta = ctx
    x = dict(zip(FEATURES, vector_features("univalle", "chiminangos", meta)))
    assert x["dist_km"] > 0
    # la distancia es simétrica
    y = dict(zip(FEATURES, vector_features("chiminangos", "univalle", meta)))
    assert x["dist_km"] == pytest.approx(y["dist_km"])
    assert x["dlat"] == pytest.approx(-y["dlat"])


def test_flags_misma_zona_y_corredor(ctx):
    _, _, meta = ctx
    mismo = dict(zip(FEATURES, vector_features("univalle", "universidades", meta)))
    assert mismo["misma_zona"] == 1.0
    assert mismo["mismo_corredor"] == 1.0

    # primer par con zona Y corredor distintos (no está hardcodeado)
    a, b = next(
        (x, y)
        for x in meta.estaciones
        for y in meta.estaciones
        if x != y
        and meta.zona_de(x) != meta.zona_de(y)
        and meta.corredor_de(x) != meta.corredor_de(y)
    )
    otro = dict(zip(FEATURES, vector_features(a, b, meta)))
    assert otro["misma_zona"] == 0.0
    assert otro["mismo_corredor"] == 0.0


def test_arista_directa(ctx):
    _, g, meta = ctx
    d = dict(zip(FEATURES, vector_features("univalle", "buitrera", meta)))
    lejos = dict(zip(FEATURES, vector_features("univalle", "chiminangos", meta)))
    assert d["arista_directa"] == float(g.hay_adyacente("univalle", "buitrera")) == 1.0
    assert lejos["arista_directa"] == 0.0


def test_meta_codifica_zonas_y_tipos(ctx):
    relations, g, meta = ctx
    assert len(meta.estaciones) == 90
    assert "zona_paso_del_comercio" in meta.estaciones
    assert meta.tipo["zona_paso_del_comercio"] == "zona"
    assert set(meta.corredor_idx) and min(meta.corredor_idx.values()) == 0
    assert meta.grado["univalle"] == len(g.vecinos("univalle"))


def test_construir_meta_es_determinista(ctx):
    relations, g, _ = ctx
    a = construir_meta(relations, g)
    b = construir_meta(relations, g)
    assert a.corredor_idx == b.corredor_idx
    assert a.zona_idx == b.zona_idx
    assert a.nombre_a_slug == b.nombre_a_slug


def test_nueve_zonas_normalizadas(ctx):
    """Regresión: la zona se guardaba en crudo y como slug a la vez, así que
    'Universidades' y 'universidades' eran dos zonas distintas (18 en vez de
    9) y `idx_zona_o`/`idx_zona_d` marcaban distinto a la misma zona."""
    _, _, meta = ctx
    assert len(meta.zona_idx) == 9, sorted(meta.zona_idx)

    # los miembros de una zona comparten índice
    a, b = "univalle", "universidades"
    assert meta.zona_de(a) == meta.zona_de(b)
    assert meta.zona_idx[meta.zona_de(a)] == meta.zona_idx[meta.zona_de(b)]
