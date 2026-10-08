"""Features numéricas para el par origen→destino.

Todas las features son observables *antes* de calcular la ruta (no se usa
ninguna etiqueta), evitando fuga de datos (data leakage):

  - geografía: coordenadas WGS84, distancia haversine, deltas
  - topología: corredor/zona de cada extremo, mismo corredor, misma zona,
    si existe arista directa y el grado de cada nodo en el grafo
  - categoría: tipo lógico de estación (estación / terminal / parada)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Tuple

FEATURES: List[str] = [
    "lat_o",
    "lon_o",
    "lat_d",
    "lon_d",
    "dist_km",
    "dlat",
    "dlon",
    "idx_corredor_o",
    "idx_corredor_d",
    "idx_zona_o",
    "idx_zona_d",
    "mismo_corredor",
    "misma_zona",
    "arista_directa",
    "grado_o",
    "grado_d",
    "tipo_o",
    "tipo_d",
]

TIPOS: Dict[str, int] = {"estacion": 0, "terminal": 1, "parada": 2, "zona": 3}


def normalizar_nombre(texto: str) -> str:
    """Normaliza un nombre visible a clave de comparación (sin tildes, snake_case)."""
    from mio_router.loaders import quitar_tildes

    t = quitar_tildes(texto.lower())
    for sep in (" ", "-", "/", ".", ",", "(", ")", "°"):
        t = t.replace(sep, "_")
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_")


@dataclass
class MetaFeatures:
    """Tablas de codificación compartidas por entrenamiento y predicción."""

    coords: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    corredor: Dict[str, str] = field(default_factory=dict)
    zona: Dict[str, str] = field(default_factory=dict)
    tipo: Dict[str, str] = field(default_factory=dict)
    grado: Dict[str, int] = field(default_factory=dict)
    corredor_idx: Dict[str, int] = field(default_factory=dict)
    zona_idx: Dict[str, int] = field(default_factory=dict)
    estaciones: List[str] = field(default_factory=list)
    nombre_a_slug: Dict[str, str] = field(default_factory=dict)
    graph: Any = None

    def corredor_de(self, slug: str) -> str:
        return self.corredor.get(slug, "")

    def zona_de(self, slug: str) -> str:
        return self.zona.get(slug, "")


def _invertir_sirve(sirve: Dict[str, List[str]]) -> Dict[str, str]:
    """Invierte {corredor: [estaciones]} a {estacion: corredor}."""
    salida: Dict[str, str] = {}
    for corredor, estaciones in sirve.items():
        for e in estaciones:
            salida.setdefault(e, corredor)
    return salida


def construir_meta(relations: Dict[str, Any], graph: Any) -> MetaFeatures:
    """Construye la codificación de features a partir de relations + grafo.

    El universo de pares OD son las 81 estaciones reales + los 9 nodos-zona
    virtuales (`zona_*`), de modo que se pueda consultar tanto por estación
    como por nombre de zona ("Paso del Comercio").
    """
    estaciones: Dict[str, Tuple[str, str, str]] = relations.get("estaciones", {})
    sirve: Dict[str, List[str]] = relations.get("sirve", {})
    coords: Dict[str, Tuple[float, float]] = relations.get("coordenadas", {})

    corredor = _invertir_sirve(sirve)
    reales = sorted(estaciones)

    zona: Dict[str, str] = {}
    tipo: Dict[str, str] = {}
    for s, v in estaciones.items():
        tipo[s] = v[1]
        zona[s] = v[2]

    corredor_idx = {c: i for i, c in enumerate(sorted(set(corredor.values())))}
    zona_idx = {z: i for i, z in enumerate(sorted(set(zona.get(s, "") for s in reales)))}

    grado = {s: len(graph.vecinos(s)) for s in reales}

    # Índice de resolución de nombres: slug, nombre legible y nombre de zona
    nombre_a_slug: Dict[str, str] = {s: s for s in reales}
    for e in relations.get("estaciones_list", []):
        nombre_a_slug.setdefault(normalizar_nombre(e.nombre), e.slug)
        if e.zona_integration:
            nombre_a_slug.setdefault(
                normalizar_nombre(e.zona_integration),
                "zona_" + normalizar_nombre(e.zona_integration),
            )

    return MetaFeatures(
        coords=coords,
        corredor=corredor,
        zona=zona,
        tipo=tipo,
        grado=grado,
        corredor_idx=corredor_idx,
        zona_idx=zona_idx,
        estaciones=reales,
        nombre_a_slug=nombre_a_slug,
        graph=graph,
    )


def distancia_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia haversine en kilómetros."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def vector_features(origen: str, destino: str, meta: MetaFeatures) -> List[float]:
    """Vector de features (orden FEATURES) para el par origen→destino."""
    lat_o, lon_o = meta.coords[origen]
    lat_d, lon_d = meta.coords[destino]

    c_o, c_d = meta.corredor_de(origen), meta.corredor_de(destino)
    z_o, z_d = meta.zona_de(origen), meta.zona_de(destino)
    t_o, t_d = meta.tipo.get(origen, "estacion"), meta.tipo.get(destino, "estacion")

    return [
        lat_o,
        lon_o,
        lat_d,
        lon_d,
        distancia_km(lat_o, lon_o, lat_d, lon_d),
        lat_d - lat_o,
        lon_d - lon_o,
        float(meta.corredor_idx.get(c_o, -1)),
        float(meta.corredor_idx.get(c_d, -1)),
        float(meta.zona_idx.get(z_o, -1)),
        float(meta.zona_idx.get(z_d, -1)),
        float(c_o == c_d and c_o != ""),
        float(z_o == z_d and z_o != ""),
        float(meta.graph.hay_adyacente(origen, destino)),
        float(meta.grado.get(origen, 0)),
        float(meta.grado.get(destino, 0)),
        float(TIPOS.get(t_o, 0)),
        float(TIPOS.get(t_d, 0)),
    ]


def dataframe_features(df: Any, meta: MetaFeatures) -> Any:
    """Matriz X (n_samples × len(FEATURES)) desde un DataFrame del dataset."""
    import numpy as np

    return np.asarray(df[FEATURES].to_numpy(dtype=float), dtype=float)
