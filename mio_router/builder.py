"""Construcción de relations (KB lógica) y grafo a partir de los datasets CSV.

Flujo:
  1. Cargar estaciones y paradas desde data/*.csv
  2. Ordenar estaciones por corredor (proyección sobre el eje del corredor)
  3. Crear aristas consecutivas dentro de cada corredor (tiempo estimado)
  4. Crear aristas de transbordo entre estaciones cercanas de distintos corredores
  5. Vincular paradas externas a la estación MIO más cercana (última milla)
  6. Emitir 'facts' Prolog + reglas de kb/reglas.pl para el motor de inferencia
"""

from __future__ import annotations

import math
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

from .parser import Fact, KnowledgeBase, Rule, extract_relations, parse_kb
from .loaders import (
    Estacion,
    Parada,
    cargar_estaciones,
    cargar_paradas,
    distancia_metros,
    minutos_entre,
    slugificar,
)

REGLAS_PATH = os.path.join("kb", "reglas.pl")

# Distancia máxima (m) para considerar transbordo caminando entre corredores.
# Cubre pares reales como Calle 5 <-> Carrera 15 (~570 m) o Calle 13 <-> Calle 15 (~200 m).
RADIO_TRANSBORDO_M = 800.0
# Distancia máxima (m) para vincular parada externa a estación
RADIO_PARADA_M = 600.0
# Tiempo caminando parada -> estación (min) fijo si está cerca
CAMINATA_PARADA_MIN = 5
# Velocidad caminata ~5 km/h => 80 m/min
_METROS_POR_MIN_CAMINATA = 80.0
# Tiempo base de espera/transbordo en estación (min)
ESPERA_TRANSBORDO_MIN = 4


def _ordenar_corredor(estaciones: List[Estacion]) -> List[Estacion]:
    """Ordena estaciones de un corredor siguiendo las coordenadas (no el FID).

    Construye una cadena geográfica: parte de un extremo del corredor y va
    encadenando la estación no visitada más cercana (vecino más próximo).
    Se prueban ambos extremos del par más lejano y se queda la cadena más
    corta. Así el orden de las aristas sigue la ruta física real, aunque el
    orden de la lista (FID) del CSV esté desordenado.
    """
    if len(estaciones) <= 2:
        return list(estaciones)

    def dist(a: Estacion, b: Estacion) -> float:
        return distancia_metros(a.lat, a.lon, b.lat, b.lon)

    # Par de extremos: las dos estaciones más lejanas entre sí
    extremo_a, extremo_b = max(
        ((a, b) for a in estaciones for b in estaciones),
        key=lambda p: dist(p[0], p[1]),
    )

    def cadena_desde(inicio: Estacion) -> List[Estacion]:
        cadena = [inicio]
        pendientes = [e for e in estaciones if e.slug != inicio.slug]
        while pendientes:
            ultimo = cadena[-1]
            # Vecino más cercano; empate determinista por slug
            sig = min(pendientes, key=lambda e: (dist(ultimo, e), e.slug))
            cadena.append(sig)
            pendientes.remove(sig)
        return cadena

    def largo(cadena: List[Estacion]) -> float:
        return sum(dist(cadena[i], cadena[i + 1]) for i in range(len(cadena) - 1))

    c1 = cadena_desde(extremo_a)
    c2 = cadena_desde(extremo_b)
    return c1 if largo(c1) <= largo(c2) else c2


def _slug_corredor(corredor: str) -> str:
    """Slug de código de ruta para un corredor (ej. 'Calle 5' -> 'calle_5')."""
    return slugificar(corredor)


def construir_relations(
    estaciones: Optional[List[Estacion]] = None,
    paradas: Optional[List[Parada]] = None,
    reglas_path: str = REGLAS_PATH,
) -> Dict[str, Any]:
    """Construye el dict 'relations' (mismo formato que extract_relations).

    Si no se pasan estaciones/paradas, los carga desde los CSV por defecto.
    """
    if estaciones is None:
        estaciones = cargar_estaciones()
    if paradas is None:
        paradas = cargar_paradas()

    # Indexar por corredor
    por_corredor: Dict[str, List[Estacion]] = defaultdict(list)
    for e in estaciones:
        por_corredor[e.corredor].append(e)

    conecta: Dict[Tuple[str, str], Tuple[str, int]] = {}
    transbordo: Dict[Tuple[str, str, str, str], bool] = {}
    sirve: Dict[str, List[str]] = defaultdict(list)
    rutas: Dict[str, Tuple[str, str]] = {}
    estaciones_dict: Dict[str, Tuple[str, str, str]] = {}
    coordenadas: Dict[str, Tuple[float, float]] = {}

    facts: List[Fact] = []

    # --- Estaciones y coordenadas ---
    for i, e in enumerate(sorted(estaciones, key=lambda x: x.fid), start=1):
        # tipo lógico: terminal si es zona_integration y vagones altos, si no estacion/parada
        tipo_logico = "estacion"
        if e.tipo in ("MUL", "MIO") and e.vagones >= 5:
            tipo_logico = "terminal"
        elif e.tipo == "PLA":
            tipo_logico = "parada"

        estaciones_dict[e.slug] = (e.slug, tipo_logico, e.zona_integration or e.corredor)
        coordenadas[e.slug] = (e.lat, e.lon)

        facts.append(Fact("estacion", (str(i), e.slug, tipo_logico,
                                       slugificar(e.zona_integration or e.corredor))))
        facts.append(Fact("coordenadas", (e.slug, f"{e.lat:.6f}", f"{e.lon:.6f}")))
        facts.append(Fact("corredor", (e.slug, _slug_corredor(e.corredor))))
        facts.append(Fact("nombre_estacion", (e.slug, slugificar(e.nombre))))

    # --- Rutas = corredores ---
    for idx, corredor in enumerate(sorted(por_corredor.keys()), start=1):
        codigo = _slug_corredor(corredor)
        rutas[codigo] = (codigo, "corredor")
        facts.append(Fact("ruta", (codigo, codigo, "corredor")))

        lista = _ordenar_corredor(por_corredor[corredor])
        for e in lista:
            sirve[codigo].append(e.slug)
            facts.append(Fact("sirve", (codigo, e.slug)))

        # Aristas consecutivas (bidireccional en graph.construir_grafo)
        for a, b in zip(lista, lista[1:]):
            minutos = minutos_entre(a.lat, a.lon, b.lat, b.lon)
            conecta[(a.slug, b.slug)] = (codigo, minutos)
            facts.append(Fact("conecta", (a.slug, b.slug, codigo, str(minutos))))

    # --- Transbordos entre corredores distintos (estaciones cercanas) ---
    # Se crea una ARISTA REAL de caminata (no un self-loop) para que el grafo
    # quede conectado entre corredores, más el hecho transbordo/4 para inferencia.
    for i, a in enumerate(estaciones):
        for b in estaciones[i + 1:]:
            if a.corredor == b.corredor:
                continue
            d = distancia_metros(a.lat, a.lon, b.lat, b.lon)
            if d <= RADIO_TRANSBORDO_M:
                walk_min = max(2, int(round(d / _METROS_POR_MIN_CAMINATA)))
                codigo_tb = "transbordo"
                # Arista de caminata entre plataformas de corredores distintos
                if (a.slug, b.slug) not in conecta:
                    conecta[(a.slug, b.slug)] = (codigo_tb, walk_min)
                    facts.append(Fact("conecta", (a.slug, b.slug, codigo_tb, str(walk_min))))
                if (b.slug, a.slug) not in conecta:
                    conecta[(b.slug, a.slug)] = (codigo_tb, walk_min)
                    facts.append(Fact("conecta", (b.slug, a.slug, codigo_tb, str(walk_min))))

                key = (a.slug, _slug_corredor(a.corredor), _slug_corredor(b.corredor),
                       str(walk_min))
                if key not in transbordo:
                    transbordo[key] = True
                    facts.append(Fact("transbordo",
                                      (a.slug, _slug_corredor(a.corredor),
                                       _slug_corredor(b.corredor), str(walk_min))))
                key2 = (b.slug, _slug_corredor(b.corredor), _slug_corredor(a.corredor),
                        str(walk_min))
                if key2 not in transbordo:
                    transbordo[key2] = True
                    facts.append(Fact("transbordo",
                                      (b.slug, _slug_corredor(b.corredor),
                                       _slug_corredor(a.corredor), str(walk_min))))

    # --- Nodos-zona virtuales (ESTACION_M): permiten consultar por zona/terminal
    # ej. "Paso del Comercio" y además conectan estaciones que comparten zona.
    por_zona: Dict[str, List[Estacion]] = defaultdict(list)
    for e in estaciones:
        if e.zona_integration:
            por_zona[e.zona_integration].append(e)

    for zona, miembros in por_zona.items():
        hub = "zona_" + slugificar(zona)
        estaciones_dict[hub] = (hub, "zona", slugificar(zona))
        # Coordenada promedio de la zona (para heurística A*)
        lat_z = sum(e.lat for e in miembros) / len(miembros)
        lon_z = sum(e.lon for e in miembros) / len(miembros)
        coordenadas[hub] = (lat_z, lon_z)
        facts.append(Fact("estacion",
                          (f"z_{slugificar(zona)}", hub, "zona", slugificar(zona))))
        facts.append(Fact("coordenadas", (hub, f"{lat_z:.6f}", f"{lon_z:.6f}")))
        for e in miembros:
            # conexión hub <-> estación con costo SEGÚN LAS COORDENADAS:
            # tiempo real de caminata desde la estación al centro de la zona.
            # Así el hub no "teletransporta" entre estaciones lejanas y las
            # rutas siguen los corredores reales desde el origen.
            d_hub = distancia_metros(e.lat, e.lon, lat_z, lon_z)
            min_hub = max(1, int(round(d_hub / _METROS_POR_MIN_CAMINATA)))
            conecta[(hub, e.slug)] = ("zona", min_hub)
            conecta[(e.slug, hub)] = ("zona", min_hub)
            facts.append(Fact("conecta", (hub, e.slug, "zona", str(min_hub))))
            facts.append(Fact("conecta", (e.slug, hub, "zona", str(min_hub))))
            facts.append(Fact("en_zona", (e.slug, hub)))

    # --- Paradas externas: vínculo a estación más cercana ---
    # Se añaden como hechos 'parada' y 'conecta_parada' (no entran al grafo
    # principal de rutas MIO, pero sí a la KB para consultas de última milla).
    est_by_slug = {e.slug: e for e in estaciones}
    for p in paradas:
        facts.append(Fact("parada", (p.slug, p.stop_id or p.slug,
                                     slugificar(p.tipo or "externa"))))
        # estación más cercana
        mejor: Optional[Estacion] = None
        mejor_d = float("inf")
        for e in estaciones:
            d = distancia_metros(p.lat, p.lon, e.lat, e.lon)
            if d < mejor_d:
                mejor_d = d
                mejor = e
        if mejor is not None and mejor_d <= RADIO_PARADA_M:
            facts.append(Fact("cerca_de", (p.slug, mejor.slug, str(int(mejor_d)))))
            # grafo de última milla: parada -> estación (caminata)
            conecta[(p.slug, mejor.slug)] = ("caminata", CAMINATA_PARADA_MIN)
            facts.append(Fact("conecta", (p.slug, mejor.slug, "caminata",
                                          str(CAMINATA_PARADA_MIN))))
            coordenadas[p.slug] = (p.lat, p.lon)

    # --- Cargar reglas desde kb/reglas.pl ---
    rules: Tuple[Rule, ...] = ()
    if os.path.exists(reglas_path):
        kb_reglas = parse_kb(reglas_path)
        rules = kb_reglas.rules

    relations: Dict[str, Any] = {
        "estaciones": estaciones_dict,
        "rutas": rutas,
        "sirve": dict(sirve),
        "conecta": conecta,
        "transbordo": transbordo,
        "coordenadas": coordenadas,
        "facts": tuple(facts),
        "rules": rules,
        "paradas": {p.slug: p for p in paradas},
        "estaciones_list": estaciones,
        "paradas_list": paradas,
    }
    return relations


def relations_desde_kb(kb_path: str) -> Dict[str, Any]:
    """Compatibilidad: relations desde un archivo .pl (KB de hechos+reglas)."""
    return extract_relations(parse_kb(kb_path))


def exportar_kb_pl(relations: Dict[str, Any], salida: str = "kb/datos_generados.pl") -> str:
    """Exporta los hechos de relations a un archivo .pl (para inspección/depuración)."""
    os.makedirs(os.path.dirname(salida) or ".", exist_ok=True)
    lines = [
        "% Archivo GENERADO automáticamente desde data/*.csv",
        "% No editar a mano; regenerar con: python -m mio_router.builder.exportar",
        "",
    ]
    for fact in relations.get("facts", []):
        args = ", ".join(fact.args)
        lines.append(f"{fact.predicate}({args}).")
    with open(salida, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return salida


def _main_export() -> None:
    rel = construir_relations()
    path = exportar_kb_pl(rel)
    print(f"KB exportada: {path}")
    print(f"  estaciones: {len(rel['estaciones'])}")
    print(f"  rutas: {len(rel['rutas'])}")
    print(f"  conexiones: {len(rel['conecta'])}")
    print(f"  hechos: {len(rel['facts'])}")
    print(f"  reglas: {len(rel['rules'])}")


if __name__ == "__main__":
    _main_export()
