"""Tests para el parser de la base de conocimiento MIO."""

import pytest

from mio_router.parser import parse_kb, extract_relations


class TestParser:
    """Tests para el parser de hechos y reglas."""

    def test_parser_lee_hechos(self):
        """Test que el parser lee correctamente los hechos de la KB."""
        kb = parse_kb("kb/mio.pl")
        # Debería tener 20 estaciones
        assert len(kb.facts) > 0

    def test_parser_estaciones(self):
        """Test que se extraen correctamente las estaciones."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        assert "estaciones" in relations
        assert len(relations["estaciones"]) == 20

    def test_parser_rutas(self):
        """Test que se extraen correctamente las rutas."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        assert "rutas" in relations
        assert len(relations["rutas"]) > 0

    def test_parser_conecta(self):
        """Test que se extraen correctamente las conexiones."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        assert "conecta" in relations
        assert len(relations["conecta"]) > 0

    def test_parser_transbordos(self):
        """Test que se extraen correctamente los transbordos."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        assert "transbordo" in relations
        assert len(relations["transbordo"]) > 0

    def test_parser_coordenadas(self):
        """Test que se extraen correctamente las coordenadas."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        assert "coordenadas" in relations
        assert len(relations["coordenadas"]) == 20


class TestParserSpecificFacts:
    """Tests para hechos específicos."""

    def test_hecho_estacion(self):
        """Test que un hecho estación se parsea correctamente."""
        kb = parse_kb("kb/mio.pl")
        # Verificar que hay un hecho de estación con nombre paso_del_comercio
        found = any(
            f.predicate == "estacion" and "paso_del_comercio" in f.args
            for f in kb.facts
        )
        assert found

    def test_hecho_ruta(self):
        """Test que un hecho ruta se parsea correctamente."""
        kb = parse_kb("kb/mio.pl")
        found = any(
            f.predicate == "ruta" for f in kb.facts
        )
        assert found

    def test_hecho_sirve(self):
        """Test que un hecho sirve se parsea correctamente."""
        kb = parse_kb("kb/mio.pl")
        found = any(
            f.predicate == "sirve" for f in kb.facts
        )
        assert found

    def test_hecho_conecta(self):
        """Test que un hecho conecta se parsea correctamente."""
        kb = parse_kb("kb/mio.pl")
        found = any(
            f.predicate == "conecta" for f in kb.facts
        )
        assert found

    def test_hecho_transbordo(self):
        """Test que un hecho transbordo se parsea correctamente."""
        kb = parse_kb("kb/mio.pl")
        found = any(
            f.predicate == "transbordo" for f in kb.facts
        )
        assert found

    def test_hecho_coordenadas(self):
        """Test que un hecho coordenadas se parsea correctamente."""
        kb = parse_kb("kb/mio.pl")
        found = any(
            f.predicate == "coordenadas" for f in kb.facts
        )
        assert found