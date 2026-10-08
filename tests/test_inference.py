"""Tests para el motor de inferencia."""

import pytest

from mio_router.parser import parse_kb, extract_relations
from mio_router.inference import (
    backward_chain,
    derivar_alcanzable,
    derivar_viaje_directo,
    derivar_requiere_transbordo,
    explicar_historial,
    unificar,
    Substitution,
)


@pytest.fixture(scope="module")
def relations():
    """Relations canónicas (hechos de los CSV + reglas de kb/reglas.pl)."""
    from mio_router.builder import construir_relations

    return construir_relations()


class TestUnificacion:
    """Tests para la unificación."""

    def test_unificacion_iguales(self):
        """Test unificación de términos idénticos."""
        result = unificar("A", "A")
        assert result is not None
        # Al unificar una variable consigo misma, no hay cambio
        assert len(result.vars_) == 0

    def test_unificacion_distintas_variables(self):
        """Test que dos variables diferentes se unifican (mapeo creado)."""
        # En Prolog/Datalog, mayúsculas son variables.
        # Al unificar "A" y "B", se crea un mapeo A->B.
        result = unificar("A", "B")
        assert result is not None
        # Se crea el mapeo A -> B
        assert result.vars_.get("A") == "B"

    def test_unificacion_variable_constante(self):
        """Test unificación de variable con constante."""
        result = unificar("X", "valor")
        assert result is not None
        # La variable X debería mapearse a "valor"
        assert result.vars_.get("X") == "valor"

    def test_unificacion_dos_variables(self):
        """Test unificación de dos variables."""
        result = unificar("X", "Y")
        assert result is not None
        # Después de unificar, X mapea a Y
        assert result.vars_.get("X") == "Y"


class TestBackwardChaining:
    """Tests para backward chaining."""

    def test_backward_chain_hecho_directo(self):
        """Test backward chaining con hecho directo."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        
        # Query para estación con 4 argumentos: (predicado, arg1, arg2, arg3, arg4)
        # El hecho es: estacion(ID, nombre, tipo, zona)
        # Usamos: estacion(1, paso_del_comercio, terminal, norte)
        query = ("estacion", "1", "paso_del_comercio", "terminal", "norte")
        soluciones = backward_chain(query, relations)
        # Debería encontrar la estación paso_del_comercio
        assert len(soluciones) > 0

    def test_backward_chain_regla(self, relations):
        """Query sin hechos directos: se deriva con las reglas externas.

        Antes los cuerpos de kb/reglas.pl se parseaban vacíos (partición por
        ',' ignorando paréntesis) y ninguna regla llegaba a dispararse.
        """
        soluciones = backward_chain(("conectado", "X", "Y"), relations)
        assert len(soluciones) > 100
        # variables realmente instanciadas (no literales 'X'/'Y')
        assert all(
            isinstance(s["X"], str) and s["X"][0].islower()
            and isinstance(s["Y"], str) and s["Y"][0].islower()
            for s in soluciones
        )
        assert any(s["X"] != s["Y"] for s in soluciones)

    def test_backward_chain_hecho_no_usa_reglas(self, relations):
        """Un hecho conocido no necesita reglas y sigue resolviéndose."""
        soluciones = backward_chain(("estacion", "1", "paso_del_comercio",
                                     "terminal", "norte"), relations)
        assert isinstance(soluciones, list)


class TestReglasDelSistema:
    """Las 8 reglas de kb/reglas.pl deben dispararse de verdad."""

    def test_las_ocho_reglas_se_parsean(self, relations):
        reglas = relations["rules"]
        assert len(reglas) == 8
        for r in reglas:
            assert r.body_atoms, f"regla {r.head_predicate} sin cuerpo"
            for pred, args in r.body_atoms:
                assert pred
                assert len(args) > 0

    @pytest.mark.parametrize(
        "query,minimo",
        [
            (("conectado", "X", "Y"), 100),
            (("viaje_directo", "X", "Y", "R"), 100),
            (("alcanzable_un_salto", "X", "Y"), 100),
            (("alcanzable_dos_saltos", "X", "Y"), 100),
            (("misma_zona", "X", "Y"), 100),
            (("pertenece_corredor", "E", "R"), 50),
            (("posible_transbordo", "E"), 10),
        ],
    )
    def test_reglas_disparan(self, relations, query, minimo):
        soluciones = backward_chain(query, relations)
        assert len(soluciones) >= minimo, query
        assert all(isinstance(v, str) for s in soluciones for v in s.values())

    def test_alcanzable_dos_saltos_encadena_z(self, relations):
        soluciones = backward_chain(("alcanzable_dos_saltos", "X", "Y"), relations)
        con_z = [s for s in soluciones if "Z" in s]
        assert con_z, "el cuerpo de dos saltos debe instanciar la variable Z"
        assert all(s["Z"] not in (s["X"], s["Y"]) for s in con_z[:50])

    def test_misma_zona_comparte_zona(self, relations):
        soluciones = backward_chain(("misma_zona", "univalle", "Y"), relations)
        assert soluciones
        zona_univalle = relations["estaciones"]["univalle"][2]
        for s in soluciones:
            assert relations["estaciones"][s["Y"]][2] == zona_univalle
        assert any(s["Y"] != "univalle" for s in soluciones)

    def test_variables_de_la_query_no_colisionan_con_las_de_la_regla(self, relations):
        """Una query con 'X'/'Y' no puede quedar mapeada a sí misma.

        Sin estandarización de variables, la X de la query y la X de la
        regla eran la misma y devolvía pares falsos (o vacíos).
        """
        soluciones = backward_chain(("viaje_directo", "X", "Y", "R"), relations)
        assert soluciones
        for s in soluciones:
            assert s["X"][0].islower() and s["Y"][0].islower() and s["R"][0].islower()
            assert s["X"] != s["Y"]


class TestDerivarAlcanzable:
    """Tests para derivar pares alcanzable."""

    def test_derivar_alcanzable_directo(self):
        """Test que se derivan pares alcanzables directamente."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        
        alc = derivar_alcanzable(relations)
        assert len(alc) > 0
        # Los primeros elementos deberían tener el formato esperado
        if alc:
            assert "X" in alc[0] or "Y" in alc[0]


class TestDerivarViajeDirecto:
    """Tests para derivar viajes directos."""

    def test_viaje_directo_existe(self):
        """Test que existe un viaje directo entre estaciones conectadas."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        
        # Terminal Paso del Comercio a Andrés Sinán debería tener viaje directo
        vd = derivar_viaje_directo(
            "paso_del_comercio", "andres_sinan", relations
        )
        assert len(vd) > 0

    def test_viaje_directo_solo_si_hay_conexion(self, relations):
        """viaje_directo/3 solo se satisface si existe una conecta/4.

        Antes el cuerpo se parseaba vacío y la regla devolvía (o no) al azar
        según los hechos disponibles.
        """
        hechos_conecta = [f.args for f in relations["facts"]
                          if f.predicate == "conecta"]
        conectados = {(a, b) for a, b, _, _ in hechos_conecta}

        a, b, ruta = hechos_conecta[0][:3]
        sol = derivar_viaje_directo(a, b, relations)
        assert sol, f"debería haber viaje directo {a} -> {b}"
        assert sol[0]["ruta"] == ruta

        nodos = [n for n, v in relations["estaciones"].items()
                 if v[1] != "zona"][:30]
        x, y = next(
            (p, q) for p in nodos for q in nodos
            if p != q and (p, q) not in conectados and (q, p) not in conectados
        )
        assert derivar_viaje_directo(x, y, relations) == []


class TestDerivarRequiereTransbordo:
    """requiere_transbordo debe depender de la ruta, no de la KB entera."""

    def test_transbordos_de_un_viaje_largo(self, relations):
        rt = derivar_requiere_transbordo("univalle", "chiminangos", relations)
        assert rt, "un viaje entre corredores distintos tiene transbordos"
        for t in rt:
            assert t["origen"] == "univalle" and t["destino"] == "chiminangos"
            assert t["ruta1"] != t["ruta2"]
            assert t["estacion_transbordo"]

    def test_viaje_directo_no_tiene_transbordos(self, relations):
        """Estaciones vecinas: el camino de tiempo mínimo no cambia de ruta.

        Antes se devolvían los 95 transbordos de la KB entera.
        """
        assert derivar_requiere_transbordo("cien_palos", "primitivo", relations) == []

    def test_estaciones_desconocidas_o_iguales(self, relations):
        assert derivar_requiere_transbordo("NoExiste", "TampocoExiste", relations) == []
        assert derivar_requiere_transbordo("univalle", "univalle", relations) == []


class TestExplicarHistorial:
    """Tests para la explicación del historial."""

    def test_explicar_historial_no_vacio(self):
        """Test que el historial de explicación no esté vacío."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        
        # Ejecutar una inferencia primero para poblar el historial
        query = ("estacion", "1", "paso_del_comercio", "terminal")
        backward_chain(query, relations)
        
        explicacion = explicar_historial(relations)
        # Debería contener algo
        assert len(explicacion) > 0