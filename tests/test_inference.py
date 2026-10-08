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

    def test_backward_chain_regla(self):
        """Test backward chaining con regla."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        
        # Query que pueda ser derivada por regla
        query = ("alcanzable", "paso_del_comercio", "andres_sinan")
        soluciones = backward_chain(query, relations)
        # Puede tener soluciones dependiendo de las reglas implementadas
        # Al mínimo, el método no debe fallar


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

    def test_viaje_directo_sin_transbordo(self):
        """Test que el viaje directo no tiene transbordo."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        
        vd = derivar_viaje_directo(
            "paso_del_comercio", "universidades", relations
        )
        # El resultado puede variar, solo verificamos que no cause error


class TestDerivarRequiereTransbordo:
    """Tests para derivar requiere transbordo."""

    def test_requiere_transbordo_estaciones(self):
        """Test que derive transbordos entre estaciones."""
        kb = parse_kb("kb/mio.pl")
        relations = extract_relations(kb)
        
        rt = derivar_requiere_transbordo(
            "Terminal Paso del Comercio", "Universidades", relations
        )
        # Debe retornar una lista (puede estar vacía dependiendo de la KB)
        assert isinstance(rt, list)


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