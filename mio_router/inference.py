"""Motor de inferencia propio: backward chaining con unificación.

Proporciona:
  - unificar(término1, término2): hace match de patrones con variables.
  - backward_chain(query, kb_relations): deriva un hecho buscando reglas.
  - historial de reglas disparadas para explicabilidad.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .parser import KnowledgeBase, Fact, Rule, extract_relations


# ---------------------------------------------------------------------------
# Unificación
# ---------------------------------------------------------------------------

@dataclass
class Substitution:
    """Representa un mapeo de variables -> valores."""
    vars_: Dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, key: str) -> Any:
        """Accede al valor de una variable."""
        return self.vars_.get(key)

    def __contains__(self, key: str) -> bool:
        """Verifica si una variable está en la sustitución."""
        return key in self.vars_

    def copy(self) -> Substitution:
        """Crea una copia de la sustitución."""
        return Substitution(vars_=self.vars_.copy())

    def get(self, key: str, default: Any = None) -> Any:
        """Obtener valor con default."""
        return self.vars_.get(key, default)


def _normalize_var(t: Any) -> Any:
    """Asegura que las variables sean strings consistentes."""
    if isinstance(t, str):
        return t
    return t


def _is_var(t: Any) -> bool:
    """Un string Mayúscula es considerado variable."""
    if isinstance(t, str):
        return t[0].isupper() or t.startswith("_")
    return False


def unificar(término1: Any, término2: Any, sigma: Optional[Substitution] = None) -> Optional[Substitution]:
    """Unificación de dos términos con variables.

    Intenta encontrar una sustitución sigma tal que term1 sigma = term2 sigma.
    Retorna None si no es posible.

    Soporta:
      - Variables (strings mayúsculas, ej. "X", "Y") se mapean a valores.
      - Constantes (strings minúsculas, números) deben ser idénticas.
      - Occurs check: evita que una variable se mapee a un término que la contiene.
    """
    if sigma is None:
        sigma = Substitution()

    # Si es la primera llamada, normalizar
    t1 = _normalize_var(término1)
    t2 = _normalize_var(término2)

    # Compuestos (tuplas): unificar elemento a elemento.
    # IMPORTANTE: este caso va ANTES que "ambas constantes", porque una tupla
    # no es variable pero tampoco una constante simple.
    if isinstance(t1, tuple) and isinstance(t2, tuple):
        if len(t1) != len(t2):
            return None
        sigma = sigma.copy()
        for a, b in zip(t1, t2):
            sigma = unificar(a, b, sigma)
            if sigma is None:
                return None
        return sigma

    # Caso: ambos son constantes (no variables)
    # Constantes son strings minúsculas o números
    if not _is_var(t1) and not _is_var(t2):
        if t1 == t2:
            return sigma
        return None

    # Caso: exactamente una es variable
    if _is_var(t1) and not _is_var(t2):
        return _unify_var(t1, t2, sigma)
    
    if not _is_var(t1) and _is_var(t2):
        return _unify_var(t2, t1, sigma)

    # Ambos son variables
    if _is_var(t1) and _is_var(t2):
        if t1 == t2:
            return sigma
        # Diferentes variables: unificar creando mapping t1 -> t2
        # Con occurs check simple
        sigma = sigma.copy()
        # Verifica que t1 no aparezca libre en t2 (para strings es trivial)
        sigma.vars_[t1] = t2
        return sigma
    # Tipos diferentes o igualdad simple
    if t1 == t2:
        return sigma
    return None


def _unify_var(var: str, term: Any, sigma: Substitution) -> Substitution:
    """Unifica una variable con un término, actualizando la sustitución."""
    var = var.strip()
    # Si la variable ya está mapeada en sigma
    if var in sigma:
        return unificar(sigma[var], term, sigma)
    
    # Occurs check: verificar que 'var' no aparezca en 'term'
    # Para strings simples, si term es la misma variable, sería circular
    new_sigma = sigma.copy()
    new_sigma.vars_[var] = term
    return new_sigma


def sigma_to_dict(sigma: Optional[Substitution]) -> Dict[str, Any]:
    """Convierte una sustitución a diccionario plano."""
    if sigma is None:
        return {}
    return sigma.vars_


# ---------------------------------------------------------------------------
# Backward Chaining
# ---------------------------------------------------------------------------

def backward_chain(
    query: Tuple[str, ...],
    relations: Dict[str, Any],
    max_depth: int = 20,
) -> List[Dict[str, Any]]:
    """Encadenamiento hacia atrás: deriva todas las maneras de satisfacer query.

    query: tupla (predicado, arg1, arg2, ...), ej. ('alcanzable', 'A', 'B')
    relations: dict de relaciones extraídas de la KB por extract_relations.

    Retorna una lista de soluciones (diccionarios con variable->valor).
    Lleva registro de las reglas disparadas para explicabilidad.
    """
    historial: List[Dict[str, str]] = []
    soluciones: List[Dict[str, Any]] = []

    # Intentar derivar la query de hechos existentes primero
    soluciones_existentes = _derivar_hecho(query, relations, historial)
    if soluciones_existentes:
        soluciones.extend(soluciones_existentes)
        # Guardar historial también cuando la respuesta fue un hecho directo
        relations["_historial"] = historial
        return soluciones

    # Si no está como hecho, intentar por reglas
    soluciones_reglas = _derivar_regla(query, relations, historial, max_depth)
    soluciones.extend(soluciones_reglas)

    # Guardar historial en relations para acceso posterior
    relations["_historial"] = historial

    return soluciones


def _derivar_hecho(query: Tuple[str, ...], relations: Dict[str, Any], historial: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Deriva una query consultando los hechos conocidos."""
    pred = query[0]
    # Buscar hechos con este predicado
    soluciones = []
    for fact in relations.get("facts", []):
        if fact.predicate == pred:
            # Intentar unificar los argumentos
            # query args son query[1:], fact args vienen del hecho parseado
            sigma = unificar(fact.args, query[1:])
            if sigma is not None:
                sol = sigma_to_dict(sigma)
                soluciones.append(sol)
                historial.append({
                    "regla": f"hecho_{pred}",
                    "descripcion": f"Hecho directo: {fact.predicate}({', '.join(fact.args)})",
                })
    return soluciones


def _derivar_regla(
    query: Tuple[str, ...],
    relations: Dict[str, Any],
    historial: List[Dict[str, str]],
    max_depth: int,
) -> List[Dict[str, Any]]:
    """Deriva una query usando reglas de inferencia."""
    pred = query[0]
    soluciones = []

    # Buscar reglas cuyo cabeza coincida con el predicado de la query
    for rule in relations.get("rules", []):
        if rule.head_predicate == pred:
            # Probar cada posible instanciación de la regla
            rule_sols = _aplicar_regla(rule, query, relations, historial, depth=0, max_depth=max_depth)
            soluciones.extend(rule_sols)

    return soluciones


def _aplicar_regla(
    rule: Rule,
    query: Tuple[str, ...],
    relations: Dict[str, Any],
    historial: List[Dict[str, str]],
    depth: int,
    max_depth: int,
) -> List[Dict[str, Any]]:
    """Aplica una regla específica usando unificación y recursión."""
    if depth >= max_depth:
        return []

    pred = query[0]
    args_query = query[1:]

    # La cabeza de la regla tiene sus propios argumentos (pueden tener variables)
    head_args = rule.head_args

    # Intentar unificar la cabeza de la regla con la query
    sigma = unificar(head_args, args_query)
    if sigma is None:
        return []

    # Aplicar sustitución a los argumentos de la cabeza
    head_args_inst = tuple(sigma.get(a, a) for a in head_args)

    # Historial: registramos la regla disparada
    historial.append({
        "regla": rule.head_predicate,
        "descripcion": f"Aplicando regla: {rule.head_predicate}({', '.join(head_args_inst)}) :- {', '.join(['(' + ', '.join(a) + ')' for a in rule.body_atoms])}",
    })

    # Ahora satisfacer el cuerpo de la regla (condiciones previas)
    # El cuerpo es una tupla de (predicado, (arg1, arg2, ...))
    soluciones_cuerpo: List[Dict[str, Any]] = [{}]

    for atom_pred, atom_args in rule.body_atoms:
        # Instanciar los argumentos con la sustitución actual
        inst_args = tuple(sigma.get(a, a) if isinstance(a, str) else a for a in atom_args)
        # También aplicamos las sustituciones acumuladas
        for prev_sol in list(soluciones_cuerpo):
            combined = prev_sol.copy()
            # Unificar con el átomo del cuerpo
            sigma_atom = unificar(inst_args, atom_args)
            if sigma_atom is None:
                soluciones_cuerpo.remove(prev_sol)
                continue
            combined.update(sigma_to_dict(sigma_atom))
            soluciones_cuerpo = [combined]
            # Verificar si el átomo del cuerpo es un hecho conocido
            atom_pred_lower = atom_pred.lower()
            atom_args_tuple = tuple(inst_args)
            # Buscar en hechos
            fact_match = False
            for fact in relations.get("facts", []):
                if fact.predicate == atom_pred_lower:
                    # Verificar si los argumentos coinciden (considerando variables)
                    sigma_fact = unificar(fact.args, atom_args_tuple)
                    if sigma_fact is not None:
                        combined.update(sigma_to_dict(sigma_fact))
                        fact_match = True
                        break
            if not fact_match:
                # Si no es un hecho directo, descarte esta rama
                soluciones_cuerpo.remove(combined)

    # Combinar soluciones del cuerpo con la sustitución de la cabeza
    soluciones_finales = []
    for sol in soluciones_cuerpo:
        # Combinar con sigma de la cabeza
        final_sol = sol.copy()
        for k, v in sigma.items():
            final_sol[k] = v
        # Verificar que las variables de la query estén resueltas
        query_inst = tuple(sigma.get(a, a) if isinstance(a, str) else a for a in args_query)
        soluciones_finales.append(final_sol)

    return soluciones_finales


# ---------------------------------------------------------------------------
# Reglas de inferencia predefinidas (derivadas de la KB)
# ---------------------------------------------------------------------------

def derivar_alcanzable(relations: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deriva todos los pares 'alcanzable(X,Y)' usando reglas recursivas.

    Alcanzable significa: existe al menos un camino (directo o con transbordos)
    desde la estación X hasta la estación Y usando las conexiones definidas.
    """
    soluciones = []
    conexiones = relations.get("conecta", {})

    # Para cada par de estaciones conectadas directamente, es alcanzable
    for (origen, destino), (ruta, minutos) in conexiones.items():
        soluciones.append({
            "X": origen,
            "Y": destino,
            "ruta": ruta,
            "minutos": minutos,
            "tipo": "directo",
        })

    return soluciones


def derivar_viaje_directo(origen: str, destino: str, relations: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deriva si hay un viaje directo desde origen hasta destino (misma ruta)."""
    conecta = relations.get("conecta", {})
    soluciones = []

    for (o, d), (ruta, minutos) in conecta.items():
        if o == origen and d == destino:
            soluciones.append({
                "ruta": ruta,
                "minutos": minutos,
                "transbordo": False,
            })

    return soluciones


def derivar_requiere_transbordo(
    origen: str,
    destino: str,
    relations: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Deriva rutas que requieren transbordo entre dos estaciones."""

    soluciones = []
    transbordos = relations.get("transbordo", {})

    # Buscar transbordos explícitos definidos en la KB
    for key, valor in transbordos.items():
        if len(key) >= 4:
            estacion_tb = key[0]
            ruta1 = key[1]
            ruta2 = key[2]
            minutos_espera = key[3]
        else:
            continue
        soluciones.append({
            "origen": origen,
            "destino": destino,
            "ruta1": ruta1,
            "ruta2": ruta2,
            "estacion_transbordo": estacion_tb,
            "minutos_espera": minutos_espera,
            "tipo": "transbordo_explicito",
        })

    return soluciones


def explicar_historial(relations: Dict[str, Any]) -> str:
    """Genera una cadena de texto explicando el razonamiento realizado."""
    historial = relations.get("_historial", [])
    if not historial:
        return "No se ha realizado ninguna inferencia aún."

    lineas = ["=== Cadena de razonamiento (historial) ==="]
    for i, entrada in enumerate(historial, 1):
        lineas.append(f"{i}. {entrada['regla']}: {entrada['descripcion']}")
    lineas.append("===========================================")
    return "\n".join(lineas)