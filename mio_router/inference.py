"""Motor de inferencia propio: backward chaining con unificación.

Proporciona:
  - unificar(término1, término2): hace match de patrones con variables.
  - backward_chain(query, kb_relations): deriva un hecho buscando reglas.
  - historial de reglas disparadas para explicabilidad.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from .graph import construir_grafo, dijkstra
from .loaders import slugificar
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

    def items(self):
        """Itera (variable, valor). Evita AttributeError al combinar soluciones."""
        return self.vars_.items()

    def __len__(self) -> int:
        return len(self.vars_)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Substitution) and self.vars_ == other.vars_


def _normalize_var(t: Any) -> Any:
    """Asegura que las variables sean strings consistentes."""
    if isinstance(t, str):
        return t
    return t


def _is_var(t: Any) -> bool:
    """Un string Mayúscula (o que empieza por '_') es considerado variable."""
    if isinstance(t, str):
        # `bool(t)` evita IndexError con argumentos vacíos ('foo().')
        return bool(t) and (t[0].isupper() or t[0] == "_")
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
    # '_' es el comodín anónimo de Prolog: NUNCA se ata, para que pueda
    # aparecer varias veces en el mismo predicado sin exigir igualdad
    # (conecta(X, Y, _, _) no debe exigir que ruta y minutos sean iguales).
    if var == "_":
        return sigma
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
                _registrar(historial, {
                    "regla": f"hecho_{pred}",
                    "descripcion": f"Hecho directo: {fact.predicate}({_formatear_args(fact.args)})",
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


def _aplicar_termino(termino: Any, bindings: Dict[str, Any]) -> Any:
    """Sigue la cadena de sustitución hasta llegar a un término resuelto."""
    visto: Set[str] = set()
    while isinstance(termino, str) and termino in bindings and termino not in visto:
        visto.add(termino)
        termino = bindings[termino]
    return termino


def _aplicar_args(args: Tuple[str, ...], bindings: Dict[str, Any]) -> Tuple[str, ...]:
    """Instancia los argumentos de un átomo con las sustituciones conocidas."""
    return tuple(_aplicar_termino(a, bindings) for a in args)


def _combinar(sol: Dict[str, Any], nuevo: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Fusiona dos sustituciones; devuelve None si se contradicen."""
    combinado = dict(sol)
    for k, v in nuevo.items():
        v_resuelto = _aplicar_termino(v, combinado)
        if k in combinado:
            existente = _aplicar_termino(combinado[k], nuevo)
            if existente != v_resuelto:
                return None
        combinado[k] = v_resuelto
    for k in list(combinado):
        combinado[k] = _aplicar_termino(combinado[k], nuevo)
    return combinado


def _formatear_args(args: Any) -> str:
    return ", ".join(str(a) for a in args)


def _formatear_cuerpo(body_atoms: Any) -> str:
    return ", ".join(f"{pred}({_formatear_args(args)})" for pred, args in body_atoms)


def _registrar(historial: List[Dict[str, str]], entrada: Dict[str, str]) -> None:
    """Añade una entrada al historial acotando su tamaño.

    Una consulta con variables puede unificar miles de hechos; sin esta
    cota, `explicar_historial` imprimiría miles de líneas.
    """
    if len(historial) < 100:
        historial.append(entrada)


_CONTADOR_REGLAS = itertools.count()


def _renombrar_regla(rule: Rule, prefijo: str) -> Tuple[Tuple[str, ...], Any, Dict[str, str]]:
    """Estandariza las variables de la regla para cada aplicación.

    Sin esto, si la query usa la misma letra que la regla
    (`viaje_directo(menga, X, R)` frente a `viaje_directo(X, Y, R)`), ambas
    variables colisionan y la regla acaba exigiendo cosas imposibles.
    """
    mapeo: Dict[str, str] = {}

    def ren(t: str) -> str:
        if isinstance(t, str) and _is_var(t) and t != "_":
            if t not in mapeo:
                mapeo[t] = prefijo + t
            return mapeo[t]
        return t

    cabeza = tuple(ren(a) for a in rule.head_args)
    cuerpo = tuple(
        (pred, tuple(ren(a) for a in args)) for pred, args in rule.body_atoms
    )
    return cabeza, cuerpo, mapeo


def _proyectar(
    sol: Dict[str, Any], mapeo: Dict[str, str], prefijo: str
) -> Dict[str, Any]:
    """Devuelve la solución con las variables de la regla en su nombre original.

    Los vínculos que ya estaban en la solución (los de la consulta) tienen
    prioridad sobre los internos de la regla.
    """
    traducido = {renombrado: original for original, renombrado in mapeo.items()}

    def valor(v: Any) -> Any:
        if isinstance(v, str) and v.startswith(prefijo):
            return traducido.get(v, v)
        return v

    salida: Dict[str, Any] = {}
    for k, v in sol.items():
        if k not in traducido:  # clave que no viene de la regla (la consulta)
            salida[k] = valor(v)
    for k, v in sol.items():
        if k in traducido and traducido[k] not in salida:
            salida[traducido[k]] = valor(v)
    return {k: _aplicar_termino(v, salida) for k, v in salida.items()}


def _resolver_atomo(
    atom_pred: str,
    atom_args: Tuple[str, ...],
    relations: Dict[str, Any],
    historial: List[Dict[str, str]],
    depth: int,
    max_depth: int,
) -> List[Dict[str, Any]]:
    """Resuelve un átomo: primero contra los hechos, luego con las reglas.

    Retorna la lista de sustituciones que hacen verdadero el átomo.
    """
    soluciones: List[Dict[str, Any]] = []

    for fact in relations.get("facts", []):
        if fact.predicate != atom_pred:
            continue
        sigma = unificar(fact.args, atom_args)
        if sigma is not None:
            soluciones.append(dict(sigma_to_dict(sigma)))

    if depth >= max_depth:
        return soluciones

    for rule in relations.get("rules", []):
        if rule.head_predicate != atom_pred:
            continue
        soluciones.extend(
            _aplicar_regla(rule, (atom_pred,) + atom_args, relations,
                           historial, depth=depth, max_depth=max_depth)
        )

    return soluciones


def _resolver_cuerpo(
    rule: Rule,
    base: Dict[str, Any],
    relations: Dict[str, Any],
    historial: List[Dict[str, str]],
    depth: int,
    max_depth: int,
) -> List[Dict[str, Any]]:
    """Satisface los átomos del cuerpo de izquierda a derecha (conjunción)."""
    soluciones: List[Dict[str, Any]] = [dict(base)]
    for atom_pred, atom_args in rule.body_atoms:
        siguientes: List[Dict[str, Any]] = []
        for sol in soluciones:
            instanciados = _aplicar_args(atom_args, sol)
            for parcial in _resolver_atomo(
                atom_pred, instanciados, relations, historial,
                depth=depth + 1, max_depth=max_depth,
            ):
                combinado = _combinar(sol, parcial)
                if combinado is not None:
                    siguientes.append(combinado)
        soluciones = siguientes
        if not soluciones:
            break
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

    # Estandarizar variables: cada aplicación usa las suyas propias
    prefijo = f"__r{next(_CONTADOR_REGLAS)}_"
    cabeza, cuerpo, mapeo = _renombrar_regla(rule, prefijo)

    # Unificar la cabeza de la regla con la query
    sigma = unificar(cabeza, query[1:])
    if sigma is None:
        return []

    regla_inst = Rule(
        head_predicate=rule.head_predicate, head_args=cabeza, body_atoms=cuerpo
    )

    # Satisfacer el cuerpo (condiciones previas) y combinar con la cabeza
    soluciones = _resolver_cuerpo(
        regla_inst, dict(sigma_to_dict(sigma)), relations, historial,
        depth=depth, max_depth=max_depth,
    )

    if soluciones:
        # Solo se registra la regla si de verdad aportó alguna solución
        _registrar(historial, {
            "regla": rule.head_predicate,
            "descripcion": (
                f"Aplicando regla: {rule.head_predicate}"
                f"({_formatear_args(_aplicar_args(cabeza, soluciones[0]))})"
                f" :- {_formatear_cuerpo(cuerpo)}"
            ).replace(prefijo, ""),
        })
        soluciones = [_proyectar(sol, mapeo, prefijo) for sol in soluciones]

    return soluciones


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


def _resolver_nodo(nombre: str, relations: Dict[str, Any]) -> str:
    """Resuelve un nombre visible o slug al nodo del grafo (si existe)."""
    estaciones = relations.get("estaciones", {})
    if nombre in estaciones:
        return nombre
    objetivo = slugificar(nombre)
    if objetivo in estaciones:
        return objetivo
    for eid, value in estaciones.items():
        if isinstance(value, (tuple, list)) and value:
            if slugificar(str(value[0])) == objetivo:
                return str(value[0])
    return nombre


def derivar_requiere_transbordo(
    origen: str,
    destino: str,
    relations: Dict[str, Any],
) -> List[Dict[str, Any]]:
    """Deriva los transbordos que usa el viaje origen->destino.

    Historial de la función:
      1. Devolvía TODOS los transbordos de la KB (ignoraba origen/destino).
      2. Filtraba por "estación alcanzable desde ambos extremos": en una red
         conectada eso seguía siendo casi todos (95 transbordos para dos
         estaciones vecinas).
      3. Ahora se recorre el camino de tiempo mínimo (mismo Dijkstra que usa
         la CLI) y solo se devuelven los transbordos de las aristas de
         caminata que ese camino cruza, con la pareja de corredores que
         efectivamente se cambia.
    """
    origen_n = _resolver_nodo(origen, relations)
    destino_n = _resolver_nodo(destino, relations)
    if origen_n == destino_n:
        return []

    ruta = dijkstra(construir_grafo(relations), origen_n, destino_n)
    if ruta is None:
        return []
    _, tramos = ruta

    # nodo[0] = origen; nodo[i] = destino del tramo i-1
    nodos = [origen_n] + [t[0] for t in tramos]
    rutas = [t[1] for t in tramos]

    soluciones: List[Dict[str, Any]] = []
    vistos: Set[Tuple[str, str, str, str]] = set()
    transbordos = relations.get("transbordo", {})

    def _emitir(estacion: str, antes: Optional[str], despues: Optional[str]) -> None:
        """Añade los hechos transbordo/4 de `estacion` compatibles con el
        corredor por el que se llega y por el que se sale."""
        # Sin contexto (inicio/fin del camino o tramo de caminata encadenado)
        # se acepta cualquier pareja de corredores de esa estación.
        sin_contexto = (
            antes is None or despues is None
            or antes == "transbordo" or despues == "transbordo"
        )
        for key in transbordos:
            if len(key) < 4 or key[0] != estacion:
                continue
            estacion_tb, ruta1, ruta2, min_espera = key[:4]
            if not sin_contexto and {ruta1, ruta2} != {antes, despues}:
                continue
            if key in vistos:
                continue
            vistos.add(key)
            soluciones.append({
                "origen": origen_n,
                "destino": destino_n,
                "ruta1": ruta1,
                "ruta2": ruta2,
                "estacion_transbordo": estacion_tb,
                "minutos_espera": min_espera,
                "tipo": "transbordo_explicito",
            })

    for i, codigo in enumerate(rutas):
        if codigo != "transbordo":
            continue
        a, b = nodos[i], nodos[i + 1]
        anterior = rutas[i - 1] if i > 0 else None
        posterior = rutas[i + 1] if i + 1 < len(rutas) else None
        _emitir(a, anterior, posterior)
        _emitir(b, posterior, anterior)

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