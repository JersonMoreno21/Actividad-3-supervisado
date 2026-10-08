"""Tests del loader y builder sobre los datasets oficiales CSV."""

import os

import pytest

from mio_router.loaders import (
    cargar_estaciones,
    cargar_paradas,
    distancia_metros,
    mercator_a_wgs84,
    minutos_entre,
    slugificar,
)
from mio_router.builder import construir_relations
from mio_router.graph import construir_grafo, dijkstra, encontrar_ruta

DATA_EST = os.path.join("data", "Estaciones_de_Parada_2025.csv")
DATA_PAR = os.path.join("data", "ptosparadas.csv")

pytestmark = pytest.mark.skipif(
    not (os.path.exists(DATA_EST) and os.path.exists(DATA_PAR)),
    reason="Datasets CSV no disponibles en data/",
)


class TestLoaders:
    def test_cargar_estaciones(self):
        ests = cargar_estaciones()
        assert len(ests) == 81
        # Coordenadas WGS84 válidas para Cali (~3.4 N, ~-76.5 W)
        for e in ests[:5]:
            assert 3.0 < e.lat < 4.0
            assert -77.0 < e.lon < -76.0

    def test_cargar_paradas(self):
        paradas = cargar_paradas()
        assert len(paradas) > 1000
        p = paradas[0]
        assert p.stop_id
        assert 3.0 < p.lat < 4.0

    def test_mercator_a_wgs84(self):
        lat, lon = mercator_a_wgs84(-8520027.79731554, 375458.722082816)
        # Coordenadas del dataset: zona de Cali (~3.2–3.5 N, ~76.5 W)
        assert 3.2 < lat < 3.6
        assert -77.0 < lon < -76.0

    def test_slugificar(self):
        assert slugificar("Cañaveralejo") == "canaveralejo"
        assert slugificar("7 de Agosto") == "e7_de_agosto"
        assert slugificar("Paso del Comercio") == "paso_del_comercio"

    def test_distancia_y_minutos(self):
        d = distancia_metros(3.45, -76.55, 3.46, -76.54)
        assert d > 0
        m = minutos_entre(3.45, -76.55, 3.46, -76.54)
        assert m >= 1


class TestBuilder:
    def test_relations_dataset(self):
        r = construir_relations()
        assert len(r["estaciones"]) >= 81  # estaciones + zonas virtuales
        assert len(r["rutas"]) == 11       # corredores
        assert len(r["conecta"]) > 0
        assert len(r["facts"]) > 1000
        assert len(r["rules"]) >= 1

    def test_grafo_conectado(self):
        r = construir_relations()
        g = construir_grafo(r)
        # estaciones reales (sin zonas ni paradas)
        ests = {s for s, v in r["estaciones"].items() if v[1] != "zona"}
        # alcance por BFS usando aristas de corredor/transbordo/zona
        from collections import deque

        vis = set()
        # empezar por una estación con aristas
        inicio = next(iter(ests))
        q = deque([inicio])
        vis.add(inicio)
        while q:
            u = q.popleft()
            for e in g.vecinos(u):
                if e.destino in ests and e.destino not in vis:
                    vis.add(e.destino)
                    q.append(e.destino)
        # al menos la mayoria de estaciones debe quedar alcanzable
        assert len(vis) >= len(ests) * 0.8

    def test_ruta_univalle_chiminangos(self):
        r = construir_relations()
        g = construir_grafo(r)
        res = dijkstra(g, "univalle", "chiminangos")
        assert res is not None
        tiempo, tramos = res
        assert tiempo > 0
        assert len(tramos) > 0

    def test_ruta_criterio_transbordos(self):
        r = construir_relations()
        g = construir_grafo(r)
        res = encontrar_ruta(g, "univalle", "nuevo_latir", criterio="transbordos")
        assert res is not None
        costo, tramos = res
        assert isinstance(costo, int)
        assert costo >= 0
