"""Tests de predicción y resolución de nombres de estación/zona."""

import pytest

from mio_router.graph import dijkstra

from supervised.dataset import contar_transbordos, preparar
from supervised.predict import predecir, resolver_estacion
from supervised.train import entrenar


@pytest.fixture(scope="module")
def ctx():
    relations, g, meta = preparar()
    return relations, g, meta


@pytest.fixture(scope="module")
def bundle(tmp_path_factory):
    """Entrena un modelo pequeño en un directorio temporal y lo recarga."""
    from supervised.dataset import cargar_dataset
    from supervised.train import cargar_modelos

    dir_modelos = str(tmp_path_factory.mktemp("models"))
    entrenar(cargar_dataset(), guardar=True, dir_modelos=dir_modelos,
             muestra=800, cv_folds=3)
    return cargar_modelos(dir_modelos=dir_modelos)


def test_resolver_estacion_y_zona(ctx):
    _, _, meta = ctx
    assert resolver_estacion("Univalle", meta) == "univalle"
    assert resolver_estacion("UNIVALLE", meta) == "univalle"
    # "Paso del Comercio" es una zona (nodo virtual zona_*)
    assert resolver_estacion("paso_del_comercio", meta) == "zona_paso_del_comercio"
    assert resolver_estacion("Paso del Comercio", meta) == "zona_paso_del_comercio"
    assert resolver_estacion("zona_universidades", meta) == "zona_universidades"
    assert resolver_estacion("7 de Agosto", meta) == "e7_de_agosto"
    assert resolver_estacion("Cañaveralejo", meta) == "canaveralejo"


def test_resolver_estacion_desconocida(ctx):
    _, _, meta = ctx
    with pytest.raises(ValueError, match="desconocida"):
        resolver_estacion("EstacionInexistente", meta)


def test_predecir_contrasta_con_dijkstra(ctx, bundle):
    _, g, _ = ctx
    res = predecir("Univalle", "Chiminangos", bundle=bundle)

    assert res["origen"] == "univalle"
    assert res["destino"] == "chiminangos"
    assert isinstance(res["tiempo_pred"], int)
    assert isinstance(res["transbordos_pred"], int)
    assert res["tiempo_pred"] >= 0
    assert res["transbordos_pred"] >= 0

    # la "real" viene del profesor (Dijkstra)
    tiempo, tramos = dijkstra(g, "univalle", "chiminangos")
    assert res["tiempo_real"] == tiempo
    assert res["transbordos_real"] == contar_transbordos(tramos)
    assert res["error_tiempo"] == abs(res["tiempo_pred"] - tiempo)


def test_predecir_por_modelo(ctx, bundle):
    from supervised.features import FEATURES

    res = predecir("univalle", "chiminangos", bundle=bundle)
    assert set(res["por_modelo"]["tiempo"]) == set(bundle["modelos_tiempo"])
    assert set(res["por_modelo"]["transbordos"]) == set(bundle["modelos_transbordos"])
    assert len(res["features"]) == len(FEATURES)


def test_predecir_prediccion_razonable(bundle):
    res = predecir("Capri", "Nuevo Latir", bundle=bundle)
    assert res["error_tiempo"] <= 60
    assert res["error_transbordos"] <= 10


def test_predecir_misma_estacion(bundle):
    with pytest.raises(ValueError, match="distintas"):
        predecir("Univalle", "Univalle", bundle=bundle)


def test_predecir_estacion_inexistente(bundle):
    with pytest.raises(ValueError, match="desconocida"):
        predecir("NoExiste", "Univalle", bundle=bundle)


def test_predecir_usa_el_grafo_que_se_le_pasa(ctx, bundle):
    """Regresión: pasar solo `g` (o solo `meta`) hacía que ambos se
    reconstruyeran y el grafo del llamante se ignorara.

    Aquí el grafo vacío no tiene camino, así que el resultado tiene que ser
    `sin_ruta` en vez de la ruta que tendría el grafo real.
    """
    from mio_router.graph import Graph

    g = Graph()
    g.adj = {"univalle": [], "chiminangos": []}
    res = predecir("univalle", "chiminangos", bundle=bundle, g=g)

    assert res["sin_ruta"] is True
    assert res["tiempo_real"] is None
    assert res["transbordos_real"] is None
    assert res["error_tiempo"] is None
    assert res["tiempo_pred"] >= 0


def test_predecir_rechaza_features_desactualizadas(ctx, bundle):
    _, g, meta = ctx
    otro = dict(bundle)
    otro["features"] = list(bundle["features"])[:-1]
    with pytest.raises(ValueError, match="Reentrena"):
        predecir("univalle", "chiminangos", bundle=otro, g=g, meta=meta)


def test_predecir_no_devuelve_minutos_negativos(ctx, bundle):
    """Un modelo lineal puede predecir minutos negativos: se recorta a 0."""
    _, g, meta = ctx

    class _Modelo:
        def __init__(self, valor):
            self.valor = valor

        def predict(self, X):
            import numpy as np
            return np.array([self.valor], dtype=float)

    otro = dict(bundle)
    otro["modelos_tiempo"] = {"neg": _Modelo(-5.0)}
    otro["mejor_tiempo"] = "neg"
    otro["modelos_transbordos"] = {"neg": _Modelo(-1.0)}
    otro["mejor_transbordos"] = "neg"

    res = predecir("univalle", "chiminangos", bundle=otro, g=g, meta=meta)
    assert res["tiempo_pred"] == 0
    assert res["transbordos_pred"] == 0
