"""Generación del dataset supervisado a partir del motor simbólico (profesor).

Flujo:
  1. Construir `relations` desde los CSV oficiales de MetroCali (`data/`).
  2. Construir el grafo MIO y restringirlo a las estaciones reales + nodos
     `zona_*` (las paradas externas son hojas de "caminata" y nunca aparecen
     en un camino óptimo entre dos estaciones).
  3. Para cada par ordenado (81 estaciones + 9 nodos `zona_*` = 90 × 89 = 8010),
     calcular con Dijkstra las etiquetas: `tiempo` (minutos) y `transbordos`
     (cambios de ruta o tramos de caminata entre plataformas).
  4. Guardar features + etiquetas en `datasets/od.csv`.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Iterable, List, Optional, Tuple

from mio_router.builder import construir_relations
from mio_router.graph import Graph, construir_grafo, dijkstra

from .features import FEATURES, MetaFeatures, construir_meta, vector_features

DATASETS_DIR = os.path.join("datasets")
DATASET_CSV = os.path.join(DATASETS_DIR, "od.csv")

COLUMNA_ORIGEN = "origen"
COLUMNA_DESTINO = "destino"
COLUMNA_TIEMPO = "tiempo"
COLUMNA_TRANSBORDOS = "transbordos"

CANDIDATAS = [COLUMNA_ORIGEN, COLUMNA_DESTINO] + FEATURES + [
    COLUMNA_TIEMPO,
    COLUMNA_TRANSBORDOS,
]


def contar_transbordos(tramos: Iterable[Tuple[str, str, int]]) -> int:
    """Cuenta transbordos de una ruta (misma regla que `mio_router.cli`).

    - un tramo cuyo código es `transbordo*` (caminata entre plataformas)
      cuenta como un transbordo;
    - si el código de ruta cambia respecto al anterior, también cuenta.
    """
    total = 0
    ruta_anterior: Optional[str] = None
    for _, codigo_ruta, _ in tramos:
        if not codigo_ruta:
            continue
        if codigo_ruta.startswith("transbordo"):
            total += 1
        elif ruta_anterior is not None and codigo_ruta != ruta_anterior:
            total += 1
        if not codigo_ruta.startswith("transbordo"):
            ruta_anterior = codigo_ruta
    return total


def grafo_estaciones(relations: Optional[Dict[str, Any]] = None) -> Graph:
    """Grafo restringido a estaciones reales + nodos `zona_*` (sin paradas)."""
    if relations is None:
        relations = construir_relations()
    g = construir_grafo(relations)

    estaciones = relations.get("estaciones", {})
    permitidos = {s for s, v in estaciones.items() if v[1] != "zona"}
    permitidos |= {n for n in g.adj if n.startswith("zona_")}

    g.adj = {
        n: [e for e in aristas if e.destino in permitidos]
        for n, aristas in g.adj.items()
        if n in permitidos
    }
    g.coordenadas = {k: v for k, v in g.coordenadas.items() if k in permitidos}
    return g


def preparar(relations: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Graph, MetaFeatures]:
    """Construye relations, grafo de estaciones y meta de features."""
    if relations is None:
        relations = construir_relations()
    g = grafo_estaciones(relations)
    meta = construir_meta(relations, g)
    return relations, g, meta


def pares_od(meta: MetaFeatures) -> List[Tuple[str, str]]:
    """Todos los pares ordenados origen≠destino (determinista por slug)."""
    estaciones = meta.estaciones
    return [(o, d) for o in estaciones for d in estaciones if o != d]


def generar_dataset(
    ruta: str = DATASET_CSV,
    relations: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> "Any":
    """Genera el dataset supervisado y lo escribe en `ruta` (CSV).

    Retorna el DataFrame de pandas.
    """
    import pandas as pd

    _, g, meta = preparar(relations)

    filas: List[Dict[str, Any]] = []
    for origen, destino in pares_od(meta):
        res = dijkstra(g, origen, destino)
        if res is None:
            continue
        tiempo, tramos = res
        fila = {
            COLUMNA_ORIGEN: origen,
            COLUMNA_DESTINO: destino,
        }
        fila.update(dict(zip(FEATURES, vector_features(origen, destino, meta))))
        fila[COLUMNA_TIEMPO] = int(tiempo)
        fila[COLUMNA_TRANSBORDOS] = contar_transbordos(tramos)
        filas.append(fila)
        if verbose and len(filas) % 1000 == 0:
            print(f"  {len(filas)} pares...")

    df = pd.DataFrame(filas, columns=CANDIDATAS)
    os.makedirs(os.path.dirname(ruta) or ".", exist_ok=True)
    df.to_csv(ruta, index=False)
    return df


def cargar_dataset(ruta: str = DATASET_CSV) -> "Any":
    """Carga `datasets/od.csv` validando columnas."""
    import pandas as pd

    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No existe el dataset: {ruta}. Genera con: python -m supervised generar"
        )
    df = pd.read_csv(ruta)
    faltantes = [c for c in CANDIDATAS if c not in df.columns]
    if faltantes:
        raise ValueError(f"Faltan columnas en {ruta}: {faltantes}")
    return df
