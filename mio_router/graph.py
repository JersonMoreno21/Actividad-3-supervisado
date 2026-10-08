"""Grafo derivado de la base de conocimiento MIO y algoritmos de búsqueda."""

from __future__ import annotations

import heapq
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import math

from .parser import extract_relations


# ---------------------------------------------------------------------------
# Grafo del sistema MIO
# ---------------------------------------------------------------------------

@dataclass
class Edge:
    """Arista en el grafo: destino, minutos, codigo_ruta."""
    destino: str
    minutos: int
    codigo_ruta: str


@dataclass
class Graph:
    """Grafo para el sistema MIO.

    Estructura: {origen: [(destino, minutos, ruta), ...]}
    """
    adj: Dict[str, List[Edge]] = field(default_factory=dict)
    coordenadas: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    def agregar_arista(self, origen: str, destino: str, minutos: int, codigo_ruta: str) -> None:
        """Añade una arista al grafo (direccional)."""
        if origen not in self.adj:
            self.adj[origen] = []
        self.adj[origen].append(Edge(destino=destino, minutos=minutos, codigo_ruta=codigo_ruta))

    def hay_adyacente(self, origen: str, destino: str) -> bool:
        """Verifica si hay una conexión directa."""
        return any(e.destino == destino for e in self.adj.get(origen, []))

    def vecinos(self, nodo: str) -> List[Edge]:
        """Retorna las aristas salientes de un nodo."""
        return self.adj.get(nodo, [])


def construir_grafo(relations: Dict[str, Any]) -> Graph:
    """Construye un grafo a partir de las relaciones extraídas de la KB."""
    g = Graph()
    g.coordenadas = relations.get("coordenadas", {})

    # Añadir conexiones directas desde hechos 'conecta' (bidireccional:
    # los buses del MIO operan en ambos sentidos)
    conecta = relations.get("conecta", {})
    for (origen, destino), (ruta, minutos) in conecta.items():
        g.agregar_arista(origen, destino, minutos, ruta)
        g.agregar_arista(destino, origen, minutos, ruta)

    # Transbordos: en el builder ya se crean aristas reales de caminata entre
    # corredores. Aquí solo se conservan como contexto si no existiera esa arista
    # (los self-loops no ayudan a Dijkstra, se omiten para no ensuciar el grafo).
    transbordos = relations.get("transbordo", {})
    for key, valor in transbordos.items():
        if len(key) < 4:
            continue
        estacion_tb = key[0]
        # Si el nodo no tiene ninguna arista de movimiento, al menos dejarlo en adj
        if estacion_tb not in g.adj:
            g.adj[estacion_tb] = []

    return g


# ---------------------------------------------------------------------------
# Dijkstra: ruta de menor tiempo
# ---------------------------------------------------------------------------

def dijkstra(g: Graph, inicio: str, fin: str) -> Optional[Tuple[int, List[Tuple[str, str, int]]]]:
    """Dijkstra: encuentra la ruta de menor tiempo desde inicio hasta fin.

    Retorna (tiempo_total, tramos) donde tramos es una lista de
    (estacion_destino, codigo_ruta, minutos_tramo).
    None si no hay camino.
    """
    # Collect all nodes: keys from adj + all destinations
    all_nodes: set = set(g.adj.keys())
    for adj_list in g.adj.values():
        for e in adj_list:
            all_nodes.add(e.destino)

    if inicio not in all_nodes or fin not in all_nodes:
        return None

    # Distancias acumuladas y nodos anteriores (para reconstruir camino)
    dist: Dict[str, int] = {inicio: 0}
    prev: Dict[str, Optional[str]] = {inicio: None}
    visitados: Set[str] = set()

    # Prioridad manual (fuerza bruta)
    max_iterations = len(all_nodes) + 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Elegir el nodo no visitado con distancia mínima
        nodo_actual = None
        dist_min = float("inf")
        for nodo in all_nodes:
            if nodo not in visitados and dist.get(nodo, float("inf")) < dist_min:
                dist_min = dist.get(nodo, float("inf"))
                nodo_actual = nodo

        if nodo_actual is None or dist_min == float("inf"):
            # No hay más nodos alcanzables
            break

        if nodo_actual == fin:
            # Ya tenemos la ruta mínima al destino
            break

        visitados.add(nodo_actual)

        # Relajar aristas
        for borde in g.vecinos(nodo_actual):
            vecino = borde.destino
            nuevo_dist = dist[nodo_actual] + borde.minutos
            if nuevo_dist < dist.get(vecino, float("inf")):
                dist[vecino] = nuevo_dist
                prev[vecino] = nodo_actual

        # Agregar nodos vecinos al conjunto
        for borde in g.vecinos(nodo_actual):
            all_nodes.add(borde.destino)

    # Si el destino no fue alcanzado
    if fin not in dist:
        return None

    # Reconstruir camino
    tramos: List[Tuple[str, str, int]] = []
    actual = fin
    while prev[actual] is not None:
        anterior = prev[actual]
        # Buscar la arista entre anterior y actual
        minutos_tramo = 0
        codigo_ruta = ""
        for borde in g.vecinos(anterior):
            if borde.destino == actual:
                minutos_tramo = borde.minutos
                codigo_ruta = borde.codigo_ruta
                break
        tramos.append((actual, codigo_ruta, minutos_tramo))
        actual = anterior

    tramos.reverse()
    tiempo_total = dist[fin]

    return tiempo_total, tramos


# ---------------------------------------------------------------------------
# Función principal de búsqueda con criterio seleccionable
# ---------------------------------------------------------------------------

def _es_transbordo(codigo_ruta: str) -> bool:
    """Detecta si una arista representa un transbordo."""
    return codigo_ruta.startswith("transbordo_")


def _costo_minutos_y_transbordos(tramos: List[Tuple[str, str, int]]) -> Tuple[int, int]:
    """Extrae el total de minutos y número de transbordos de la ruta.

    Retorna (total_minutos, num_transbordos).
    """
    total_minutos = sum(minutos for _, _, minutos in tramos)
    transbordos = sum(1 for _, codigo_ruta, _ in tramos if _es_transbordo(codigo_ruta))
    return total_minutos, transbordos


def encontrar_ruta(
    g: Graph,
    inicio: str,
    fin: str,
    criterio: str = "tiempo",
) -> Optional[Tuple[int, List[Tuple[str, str, int]]]]:
    """Encuentra la mejor ruta según el criterio seleccionado.

    criterio:
      - "tiempo": minimiza minutos totales (Dijkstra clásico).
      - "transbordos": minimiza cambios de ruta; desempate por tiempo.
      - "equilibrado": minimiza tiempo + 5 * transbordos.

    Retorna (costo_segun_criterio, tramos) o None si no hay camino.
    Cada tramo es (estacion_destino, codigo_ruta, minutos).
    """
    if criterio == "tiempo":
        return dijkstra(g, inicio, fin)

    if inicio not in g.adj:
        return None

    # Búsqueda por estados (nodo, ruta_entrante) para contar transbordos.
    State = Tuple[str, Optional[str]]

    def clave(tm: Tuple[int, int]) -> Tuple[int, int]:
        t, m = tm
        if criterio == "transbordos":
            return (t, m)
        return (m + 5 * t, m)

    start: State = (inicio, None)
    dist: Dict[State, Tuple[int, int]] = {start: (0, 0)}
    prev: Dict[State, Optional[Tuple[State, Edge]]] = {start: None}

    heap: List[Tuple[Tuple[int, int], State]] = [(clave((0, 0)), start)]
    visitados: Set[State] = set()
    final_state: Optional[State] = None

    total_aristas = sum(len(v) for v in g.adj.values())
    max_iterations = total_aristas * 4 + 50
    iteration = 0

    while heap and iteration < max_iterations:
        iteration += 1
        c, state = heapq.heappop(heap)
        if state in visitados:
            continue
        if clave(dist[state]) != c:
            continue
        visitados.add(state)

        nodo, ruta_prev = state
        if nodo == fin:
            final_state = state
            break

        for borde in g.vecinos(nodo):
            # Self-loops (si quedaran) no desplazan; las aristas de transbordo
            # reales (caminata entre corredores) SÍ deben usarse.
            if borde.destino == nodo:
                continue
            t_prev, m_prev = dist[state]
            dt = 0 if (ruta_prev is None or ruta_prev == borde.codigo_ruta) else 1
            if borde.codigo_ruta.startswith("transbordo") and ruta_prev != borde.codigo_ruta:
                dt = 1
            nuevo: State = (borde.destino, borde.codigo_ruta)
            nuevo_tm = (t_prev + dt, m_prev + borde.minutos)
            if nuevo not in dist or clave(nuevo_tm) < clave(dist[nuevo]):
                dist[nuevo] = nuevo_tm
                prev[nuevo] = (state, borde)
                heapq.heappush(heap, (clave(nuevo_tm), nuevo))

    if final_state is None:
        candidatos = [s for s in dist if s[0] == fin]
        if not candidatos:
            return None
        final_state = min(candidatos, key=lambda s: clave(dist[s]))

    t, m = dist[final_state]
    tramos: List[Tuple[str, str, int]] = []
    cur = final_state
    while prev.get(cur) is not None:
        entrada = prev[cur]
        if entrada is None:
            break
        _pstate, borde = entrada
        tramos.append((borde.destino, borde.codigo_ruta, borde.minutos))
        cur = _pstate
    tramos.reverse()

    if criterio == "transbordos":
        return t, tramos
    return m + 5 * t, tramos


# ---------------------------------------------------------------------------
# A*: búsqueda A* con heurística de distancia euclídea normalizada
# ---------------------------------------------------------------------------

def heuristica(estacion: str, destino: str, g: Graph) -> float:
    """Heurística de A*: distancia euclídea normalizada a minutos.

    Usa la distancia euclídea entre coordenadas, normalizada a minutos.
    La heurística es admisible: nunca sobreestima el costo real.
    """
    coords_ini = g.coordenadas.get(estacion)
    coords_fin = g.coordenadas.get(destino)

    if coords_ini is None or coords_fin is None:
        return float("inf")

    lat1, lon1 = coords_ini
    lat2, lon2 = coords_fin

    # Distancia euclídea pura (en grados)
    d = math.sqrt((lat1 - lat2) ** 2 + (lon1 - lon2) ** 2)

    # Normalizar a "minutos" usando un factor aproximado.
    # En el MIO, 1 grado ≈ 111 km ≈ 50-70 minutos en tráfico.
    # Usamos 50 min/grado como factor conservador.
    minutos_por_grado = 50.0
    d_minutos = d * minutos_por_grado

    return max(0.0, d_minutos)


def a_estrella(
    g: Graph,
    inicio: str,
    fin: str,
    criterio: str = "tiempo",
) -> Optional[Tuple[int, List[Tuple[str, str, int]]]]:
    """A*: búsqueda A* optimizada.

    criterio: "tiempo" (minutos totales), "transbordos" (mínimo cambios),
              "equilibrado" (tiempo + penalización por transbordo).

    Retorna (costo_total, tramos) o None si no hay camino.
    """
    if inicio not in g.adj:
        return None

    # Collect all nodes for heuristic purposes
    all_nodes: set = set(g.adj.keys())
    for adj_list in g.adj.values():
        for e in adj_list:
            all_nodes.add(e.destino)

    # Inicializar costo g (usando tupla para comparacion: (minutos, transbordos))
    # y estructuras de seguimiento
    # Costo g: (minutos_acumulados, transbordos_acumulados)
    costo_g: Dict[str, Tuple[int, int]] = {inicio: (0, 0)}
    ruta_anterior: Dict[str, Optional[str]] = {inicio: None}
    transbordo_count: Dict[str, int] = {inicio: 0}

    # Priority queue: (f = g + h, g, nodo)
    # Usaremos lista simple, ordenando por f (primero minutos, luego transbordos)
    frontier: List[Tuple[Tuple[int, int], Tuple[int, int], str]] = [
        ((heuristica(inicio, fin, g), 0), (0, 0), inicio)
    ]
    visitados: Set[str] = set()

    prev_edge: Dict[str, Optional[Edge]] = {inicio: None}

    max_iterations = len(all_nodes) * 2 + 10
    iteration = 0

    while frontier and iteration < max_iterations:
        # Seleccionar el de menor f (sort por la tupla f)
        frontier.sort(key=lambda x: x[0])
        f, g_actual, nodo_actual = frontier.pop(0)
        iteration += 1

        if nodo_actual in visitados:
            continue

        if nodo_actual == fin:
            # Reconstruir camino y retornar según criterio
            return _reconstruir_a_estrella(
                g, prev_edge, costo_g, transbordo_count, fin, criterio
            )

        visitados.add(nodo_actual)

        for borde in g.vecinos(nodo_actual):
            vecino = borde.destino
            nuevo_g_minutos = g_actual[0] + borde.minutos
            # Contar transbordo si cambia de ruta
            nuevo_g_transbordes = g_actual[1]
            if ruta_anterior.get(nodo_actual) is not None and borde.codigo_ruta != ruta_anterior.get(nodo_actual):
                nuevo_g_transbordes += 1

            nuevo_g = (nuevo_g_minutos, nuevo_g_transbordes)

            if nuevo_g < costo_g.get(vecino, (float("inf"), float("inf"))):
                costo_g[vecino] = nuevo_g
                ruta_anterior[vecino] = borde.codigo_ruta if borde.codigo_ruta else borde.codigo_ruta
                transbordo_count[vecino] = nuevo_g_transbordes

                # Costo heurístico
                h = heuristica(vecino, fin, g)

                # Costo total f = g + h (usando minutos para f, transbordos como desempate)
                f_val = (nuevo_g_minutos + int(h), nuevo_g_transbordes)

                frontier.append((f_val, nuevo_g, vecino))
                prev_edge[vecino] = borde

    return None


def _reconstruir_a_estrella(
    g: Graph,
    prev_edge: Dict[str, Optional[Edge]],
    costo_g: Dict[str, Tuple[int, int]],
    transbordo_count: Dict[str, int],
    fin: str,
    criterio: str,
) -> Optional[Tuple[int, List[Tuple[str, str, int]]]]:
    """Reconstruye el camino desde el diccionario de aristas previas."""
    if fin not in costo_g:
        return None

    minutos_total, _ = costo_g[fin]
    tramos: List[Tuple[str, str, int]] = []

    actual = fin
    while prev_edge.get(actual) is not None:
        borde = prev_edge[actual]
        if borde is None:
            break
        minutos_tramo = borde.minutos
        codigo_ruta = borde.codigo_ruta
        tramos.append((actual, codigo_ruta, minutos_tramo))
        actual = borde.destino

    if not tramos:
        return None

    tramos.reverse()

    # Calcular número de transbordos en el camino
    transbordos = 0
    ruta_anterior: Optional[str] = None
    for estacion, codigo_ruta, minutos in tramos:
        if ruta_anterior is not None and codigo_ruta != ruta_anterior:
            transbordos += 1
        ruta_anterior = codigo_ruta

    # Ajustar costo según criterio
    if criterio == "tiempo":
        costo_total = minutos_total
    elif criterio == "transbordos":
        costo_total = transbordos
    else:  # equilibrado
        costo_total = minutos_total + 5 * transbordos

    return costo_total, tramos