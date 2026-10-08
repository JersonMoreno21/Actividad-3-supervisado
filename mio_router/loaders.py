"""Carga de datasets oficiales del MIO (CSV de MetroCali).

Datasets:
  - data/Estaciones_de_Parada_2025.csv  -> estaciones del sistema MIO
  - data/ptosparadas.csv                -> paradas externas (afuera de estaciones)

Las estaciones usan coordenadas Web Mercator (X, Y); se convierten a WGS84
(lat, lon) para cálculos de distancia y heurística A*.
Las paradas ya traen LATITUD / LONGITUD en WGS84.
"""

from __future__ import annotations

import csv
import math
import os
import unicodedata
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

# Las rutas se anclan a la raíz del repositorio (padre del paquete) para que
# el código funcione aunque se ejecute desde otro directorio.
_RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_RAIZ, "data")
ESTACIONES_CSV = os.path.join(DATA_DIR, "Estaciones_de_Parada_2025.csv")
PARADAS_CSV = os.path.join(DATA_DIR, "ptosparadas.csv")

# Web Mercator (EPSG:3857) radio de la Tierra en metros
_MERCATOR_RADIUS = 20037508.34


@dataclass
class Estacion:
    """Estación MIO (del dataset de estaciones)."""
    id_estacion: str
    nombre: str
    slug: str
    corredor: str
    tipo: str           # DOB, SEN, MUL, TRI, MIO, PLA...
    integracion: str    # S/N de integración física
    direccion: str
    vagones: int
    zona_integration: str  # ESTACION_M (zona de integración / agrupación)
    lat: float
    lon: float
    fid: int


@dataclass
class Parada:
    """Parada externa (afuera de estaciones)."""
    stop_id: str
    direccion: str
    complemento: str
    tipo: str           # PARADA EXTERNA / ESTACION / TERMINAL
    barrio: str
    sector: str
    zona: str
    lat: float
    lon: float
    slug: str


def quitar_tildes(texto: str) -> str:
    """Quita tildes y normaliza a ASCII básico."""
    # ñ -> n para identifiers Prolog/URL amigables
    texto = texto.replace("ñ", "n").replace("Ñ", "N")
    descompuesto = unicodedata.normalize("NFD", texto)
    sin_tildes = "".join(c for c in descompuesto if unicodedata.category(c) != "Mn")
    return sin_tildes


def slugificar(nombre: str) -> str:
    """Convierte un nombre a identifier seguro (minúsculas, sin tildes, snake_case)."""
    t = quitar_tildes(nombre).lower()
    # separadores -> underscore
    for ch in (" ", "-", "/", ".", ",", "(", ")", "°"):
        t = t.replace(ch, "_")
    # colapsar underscores
    while "__" in t:
        t = t.replace("__", "_")
    t = t.strip("_")
    # Prolog/regex: si empieza con dígito, prefijo 'e'
    if t and t[0].isdigit():
        t = "e" + t
    return t or "sin_nombre"


def mercator_a_wgs84(x: float, y: float) -> Tuple[float, float]:
    """Convierte coordenadas Web Mercator (EPSG:3857) a (lat, lon) WGS84."""
    lon = x * 180.0 / _MERCATOR_RADIUS
    lat = math.degrees(2.0 * math.atan(math.exp(y * math.pi / _MERCATOR_RADIUS)) - math.pi / 2.0)
    return lat, lon


def _abrir_csv(ruta: str) -> List[Dict[str, str]]:
    """Abre un CSV UTF-8 y retorna filas como diccionarios."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No se encontró el dataset: {ruta}")
    with open(ruta, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def cargar_estaciones(ruta: str = ESTACIONES_CSV) -> List[Estacion]:
    """Carga las estaciones del MIO desde el CSV oficial."""
    filas = _abrir_csv(ruta)
    estaciones: List[Estacion] = []
    for fila in filas:
        nombre = (fila.get("NOMBRE_EST") or fila.get("NOMBRE") or "").strip()
        if not nombre:
            continue
        try:
            x = float(fila["X"])
            y = float(fila["Y"])
        except (KeyError, ValueError):
            continue
        lat, lon = mercator_a_wgs84(x, y)

        try:
            vagones = int(fila.get("VAGONES") or 0)
        except ValueError:
            vagones = 0
        try:
            fid = int(fila.get("FID") or 0)
        except ValueError:
            fid = 0

        estaciones.append(
            Estacion(
                id_estacion=(fila.get("ID_ESTACIO") or "").strip(),
                nombre=nombre,
                slug=slugificar(nombre),
                corredor=(fila.get("CORREDOR_T") or "").strip() or "Sin corredor",
                tipo=(fila.get("TIPO_ESTAC") or "").strip() or "EST",
                integracion=(fila.get("INTEGRACIO") or "").strip(),
                direccion=(fila.get("DIRECCION") or "").strip(),
                vagones=vagones,
                zona_integration=(fila.get("ESTACION_M") or "").strip(),
                lat=lat,
                lon=lon,
                fid=fid,
            )
        )
    return estaciones


def cargar_paradas(ruta: str = PARADAS_CSV) -> List[Parada]:
    """Carga las paradas externas desde el CSV de MetroCali."""
    filas = _abrir_csv(ruta)
    paradas: List[Parada] = []
    for fila in filas:
        try:
            lat = float(fila["LATITUD"])
            lon = float(fila["LONGITUD"])
        except (KeyError, ValueError):
            continue
        stop_id = (fila.get("STOPID") or "").strip()
        direccion = (fila.get("DIRECCION") or "").strip()
        slug = slugificar(direccion) if direccion else f"parada_{stop_id}"
        # Evitar slugs duplicados entre paradas
        paradas.append(
            Parada(
                stop_id=stop_id or fila.get("FID", ""),
                direccion=direccion,
                complemento=(fila.get("COMPLEMENT") or "").strip(),
                tipo=(fila.get("T_PARADA") or "").strip(),
                barrio=(fila.get("BARRIO") or "").strip(),
                sector=(fila.get("SECTOR") or "").strip(),
                zona=(fila.get("ZONA") or "").strip(),
                lat=lat,
                lon=lon,
                slug=slug,
            )
        )
    # Deduplicate slugs
    seen: Dict[str, int] = {}
    for p in paradas:
        if p.slug in seen:
            seen[p.slug] += 1
            p.slug = f"{p.slug}_{seen[p.slug]}"
        else:
            seen[p.slug] = 1
    return paradas


def distancia_metros(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distancia en metros entre dos puntos WGS84 (haversine)."""
    r = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def minutos_entre(lat1: float, lon1: float, lat2: float, lon2: float,
                  velocidad_kmh: float = 20.0) -> int:
    """Estima minutos de viaje entre dos puntos (velocidad media MIO ~20 km/h)."""
    km = distancia_metros(lat1, lon1, lat2, lon2) / 1000.0
    minutos = km / velocidad_kmh * 60.0
    return max(1, int(round(minutos)))
