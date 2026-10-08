"""Tests para el parser de la base de conocimiento MIO."""

import pytest

from mio_router.parser import parse_kb, extract_relations


@pytest.fixture(scope="module")
def kb_reglas():
    """kb/reglas.pl: 8 reglas con cuerpo (es el archivo que las pruebas de
    inferencia usan de verdad)."""
    return parse_kb("kb/reglas.pl")


class TestParser:
    """Tests para el parser de hechos y reglas."""

    def test_parser_lee_hechos(self):
        """Test que el parser lee correctamente los hechos de la KB."""
        kb = parse_kb("kb/mio.pl")
        # la KB legacy tiene ~95 hechos (estaciones, rutas, conecta, ...)
        assert len(kb.facts) >= 90

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


class TestParserDeReglas:
    """kb/reglas.pl: el parser debe conservar cuerpo y variables.

    Regresión: el cuerpo se partía por ',' ignorando la profundidad de
    paréntesis y se pasaba a minúsculas, así que las variables (X, Y) de
    kb/reglas.pl morían y ninguna regla podía dispararse.
    """

    def test_las_ocho_reglas_estan(self, kb_reglas):
        assert len(kb_reglas.rules) == 8
        assert len(kb_reglas.facts) == 0

    @pytest.mark.parametrize(
        "predicado,cuerpo_esperado",
        [
            ("conectado", [("conecta", ("X", "Y", "_", "_"))]),
            ("viaje_directo", [("conecta", ("X", "Y", "R", "_"))]),
            (
                "alcanzable_dos_saltos",
                [("conecta", ("X", "Z", "_", "_")),
                 ("conecta", ("Z", "Y", "_", "_"))],
            ),
            (
                "misma_zona",
                [("estacion", ("_", "X", "_", "Z")),
                 ("estacion", ("_", "Y", "_", "Z"))],
            ),
        ],
    )
    def test_cuerpo_y_variables(self, kb_reglas, predicado, cuerpo_esperado):
        regla = next(r for r in kb_reglas.rules if r.head_predicate == predicado)
        assert list(regla.body_atoms) == list(cuerpo_esperado)

    def test_variables_mayusculas_intactas(self, kb_reglas):
        """Regresión: `.lower()` en los argumentos convertía X en la constante x."""
        for r in kb_reglas.rules:
            cabeza = [a for a in r.head_args if a[:1].isupper()]
            cuerpo = [a for _, args in r.body_atoms for a in args if a[:1].isupper()]
            assert cabeza, f"{r.head_predicate} sin variables en la cabeza"
            assert cuerpo, f"{r.head_predicate}: cuerpo parseado sin variables"
            for v in cabeza:
                assert v.lower() not in cuerpo

    def test_comentarios_ignorados(self, kb_reglas):
        for r in kb_reglas.rules:
            assert not r.head_predicate.startswith("%")
            for pred, _ in r.body_atoms:
                assert not pred.startswith("%")