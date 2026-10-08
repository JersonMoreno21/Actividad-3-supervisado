"""Tests para el grafo y algoritmos de búsqueda."""

import pytest

from mio_router.parser import parse_kb, extract_relations
from mio_router.graph import (
    construir_grafo,
    dijkstra,
    encontrar_ruta,
    heuristica,
    Edge,
    Graph,
)


class TestGraph:
    """Tests para la estructura Graph."""

    def test_construir_grafo_tiene_estaciones(self):
        """Test que el grafo se construye con estaciones."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)
        assert len(g.adj) > 0

    def test_grafo_tiene_aristas(self):
        """Test que el grafo tiene aristas."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)
        # Debería haber aristas de conexión
        total_vecinos = sum(len(v) for v in g.adj.values())
        assert total_vecinos > 0

    def test_grafo_vecinos(self):
        """Test que se pueden obtener vecinos de un nodo."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)
        
        vecinos = g.vecinos("paso_del_comercio")
        assert len(vecinos) > 0

    def test_grafo_arista_valida(self):
        """Test que las aristas tienen la estructura correcta."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)
        
        for nodo, aristas in g.adj.items():
            for arista in aristas:
                assert hasattr(arista, "destino")
                assert hasattr(arista, "minutos")
                assert hasattr(arista, "codigo_ruta")


class TestDijkstra:
    """Tests para Dijkstra."""

    def test_dijkstra_paso_a_menga(self):
        """Test Dijkstra: paso_del_comercio a menga."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)
        
        resultado = dijkstra(g, "paso_del_comercio", "menga")
        assert resultado is not None
        tiempo, tramos = resultado
        assert tiempo == 10  # 6 + 4 minutos

    def test_dijkstra_paso_a_universidades(self):
        """Test Dijkstra: paso_del_comercio a universidades."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)
        
        resultado = dijkstra(g, "paso_del_comercio", "universidades")
        assert resultado is not None
        tiempo, tramos = resultado
        assert tiempo > 0

    def test_dijkstra_mismo_nodo(self):
        """Dijkstra de un nodo a sí mismo: coste 0 sin tramos."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)

        resultado = dijkstra(g, "paso_del_comercio", "paso_del_comercio")
        assert resultado is not None
        tiempo, tramos = resultado
        assert tiempo == 0
        assert tramos == []


class TestEncontrarRuta:
    """Tests para encontrar_ruta con diferentes criterios."""

    def test_encontrar_ruta_tiempo(self):
        """Test encontrar_ruta con criterio tiempo."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)
        
        resultado = encontrar_ruta(g, "paso_del_comercio", "universidades", criterio="tiempo")
        assert resultado is not None
        costo, tramos = resultado
        assert costo > 0

    def test_encontrar_ruta_transbordos(self):
        """Test encontrar_ruta con criterio transbordos."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)
        
        resultado = encontrar_ruta(g, "paso_del_comercio", "universidades", criterio="transbordos")
        assert resultado is not None
        costo, tramos = resultado
        # Costo debería ser número de transbordos
        assert isinstance(costo, (int, float))

    def test_encontrar_ruta_equilibrado(self):
        """Test encontrar_ruta con criterio equilibrado."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)
        
        resultado = encontrar_ruta(g, "paso_del_comercio", "universidades", criterio="equilibrado")
        assert resultado is not None
        costo, tramos = resultado
        assert costo > 0


class TestHeuristica:
    """Tests para la heurística de A*."""

    def test_heuristica_tiene_valor(self):
        """Test que la heurística retorna un valor numérico."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)
        
        h = heuristica("paso_del_comercio", "universidades", g)
        assert isinstance(h, float)
        assert h >= 0

    def test_heuristica_origen_destino_mismo(self):
        """Test heurística cuando origen = destino."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)
        
        h = heuristica("paso_del_comercio", "paso_del_comercio", g)
        assert h == 0.0


class TestIntegridadDelGrafo:
    """Regresiones: aristas duplicadas y determinismo del Dijkstra."""

    def test_sin_aristas_duplicadas(self):
        """Cada par de nodos aparece como mucho una vez por dirección.

        El builder guarda los dos sentidos de los transbordos y las zonas, y
        `construir_grafo` añadía además el inverso: el grado de todos los
        nodos salía inflado y con él las features `grado_o`/`grado_d`.
        """
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        g = construir_grafo(relations)

        for nodo, aristas in g.adj.items():
            destinos = [a.destino for a in aristas]
            assert len(destinos) == len(set(destinos)), f"duplicados en {nodo}"

    def test_dijkstra_es_determinista(self):
        """La misma consulta devuelve siempre la misma ruta.

        Antes la selección del nodo mínimo iteraba un `set`, así que el
        resultado dependía de PYTHONHASHSEED y `od.csv` cambiaba entre
        ejecuciones.
        """
        from mio_router.builder import construir_relations

        relations = construir_relations()
        g = construir_grafo(relations)
        resultados = [dijkstra(g, "univalle", "chiminangos") for _ in range(5)]
        assert resultados[0] is not None
        assert all(r == resultados[0] for r in resultados)