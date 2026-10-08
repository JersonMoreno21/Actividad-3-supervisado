"""CLI para el router MIO: sistema de rutas inteligente en transporte masivo."""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Tuple

# Forzar UTF-8 en la consola para mostrar correctamente caracteres como 'ñ'
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from .parser import parse_kb, extract_relations
from .graph import construir_grafo, encontrar_ruta as buscar_en_grafo
from .builder import construir_relations, REGLAS_PATH

KB_PATH = "kb/mio.pl"  # KB de hechos legacy (fallback / tests)

# Caché global de relations (dataset por defecto)
_relations_cache: Optional[Dict[str, Any]] = None
_fuente_preferida: str = "dataset"


def _cargar_relations(fuente: Optional[str] = None) -> Dict[str, Any]:
    """Carga relations desde el dataset CSV (por defecto) o desde kb/mio.pl."""
    global _relations_cache, _fuente_preferida
    if fuente is not None:
        _fuente_preferida = fuente
        _relations_cache = None

    if _relations_cache is None:
        if _fuente_preferida == "kb":
            _relations_cache = extract_relations(parse_kb(KB_PATH))
        else:
            try:
                _relations_cache = construir_relations()
            except FileNotFoundError:
                # Fallback si faltan los CSVs
                _relations_cache = extract_relations(parse_kb(KB_PATH))
    return _relations_cache


# ---------------------------------------------------------------------------
# Utilidades de nombre (tolerancia a tildes/mayúsculas)
# ---------------------------------------------------------------------------

_normalizar_map: Optional[Dict[str, str]] = None
_nombres_kb: Optional[List[str]] = None


def _cargar_normalizador() -> None:
    """Carga los nombres de la KB/dataset para autocompletado y normalización."""
    global _normalizar_map, _nombres_kb
    if _normalizar_map is not None:
        return

    relations = _cargar_relations()
    _nombres_kb = []
    for eid, (nombre, tipo, zona) in relations.get("estaciones", {}).items():
        # Preferir nombre visible si está en estaciones_list
        _nombres_kb.append(nombre)

    # Enriquecer con nombres legibles desde estaciones_list (dataset)
    for e in relations.get("estaciones_list", []):
        if e.nombre not in _nombres_kb:
            _nombres_kb.append(e.nombre)
        if e.slug not in _nombres_kb:
            _nombres_kb.append(e.slug)

    # Paradas externas también son consultables
    for p in relations.get("paradas_list", []):
        if p.direccion:
            _nombres_kb.append(p.direccion)

    _normalizar_map = {}
    for nombre in _nombres_kb:
        clave = _quitar_tildes(nombre.lower())
        _normalizar_map[clave] = nombre


def _quitar_tildes(texto: str) -> str:
    """Quita tildes de un texto para comparación tolerante."""
    replacements = {
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u",
        "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U",
    }
    for til, sin_tilde in replacements.items():
        texto = texto.replace(til, sin_tilde)
    return texto


def _normalizar_nombre(nombre: str) -> str:
    """Normaliza un nombre para comparación (quita tildes, espacios->_)."""
    texto = _quitar_tildes(nombre.lower())
    # Unificar separadores: espacios y guiones -> guion bajo (formato de la KB)
    texto = texto.replace(" ", "_").replace("-", "_")
    # Colapsar guiones bajos repetidos
    while "__" in texto:
        texto = texto.replace("__", "_")
    return texto


def _buscar_estacion(nombre_busqueda: str) -> Optional[str]:
    """Busca una estación/parada/zona en la KB por nombre.

    Orden de prioridad:
      1. Estación exacta (nombre o slug)
      2. Zona virtual (ESTACION_M, ej. 'Paso del Comercio')
      3. Parada externa exacta
      4. Coincidencia parcial (estaciones primero, luego paradas)
    """
    _cargar_normalizador()
    relations = _cargar_relations()
    nombre_norm = _normalizar_nombre(nombre_busqueda)

    estaciones = relations.get("estaciones", {})
    estaciones_list = relations.get("estaciones_list", [])
    paradas_list = relations.get("paradas_list", [])

    # 1) Estación exacta: slug o nombre legible
    if nombre_norm in estaciones:
        return nombre_norm
    for e in estaciones_list:
        if _normalizar_nombre(e.nombre) == nombre_norm or _normalizar_nombre(e.slug) == nombre_norm:
            return e.slug
    # KB legacy: keys son IDs, values=(nombre, tipo, zona) — el nodo del grafo es el nombre
    for eid, value in estaciones.items():
        if isinstance(value, (tuple, list)) and value:
            nombre_nodo = value[0]
            if _normalizar_nombre(str(nombre_nodo)) == nombre_norm or str(eid) == nombre_norm:
                return str(nombre_nodo)

    # 2) Zona virtual
    for eid, (slug, tipo, _z) in estaciones.items():
        if tipo == "zona" and (_normalizar_nombre(slug) == nombre_norm
                               or nombre_norm in _normalizar_nombre(slug)):
            return slug
    # también matchear por nombre de zona legible
    for zona in set(e.zona_integration for e in estaciones_list if e.zona_integration):
        if _normalizar_nombre(zona) == nombre_norm:
            return "zona_" + _normalizar_nombre(zona).replace(" ", "_")

    # 3) Parada externa exacta (dirección o slug)
    for p in paradas_list:
        if p.direccion and _normalizar_nombre(p.direccion) == nombre_norm:
            return p.slug
        if _normalizar_nombre(p.slug) == nombre_norm:
            return p.slug

    # 4) Parcial: estaciones primero
    for e in estaciones_list:
        n = _normalizar_nombre(e.nombre)
        if nombre_norm and (nombre_norm in n or n in nombre_norm):
            return e.slug
    for e in estaciones_list:
        n = _normalizar_nombre(e.slug)
        if nombre_norm and (nombre_norm in n or n in nombre_norm):
            return e.slug
    # KB legacy parcial por nombre
    for eid, value in estaciones.items():
        if isinstance(value, (tuple, list)) and value:
            n = _normalizar_nombre(str(value[0]))
            if nombre_norm and n and (nombre_norm in n or n in nombre_norm):
                return str(value[0])

    # 5) Parcial: paradas
    for p in paradas_list:
        if p.direccion:
            n = _normalizar_nombre(p.direccion)
            if nombre_norm and (nombre_norm in n or n in nombre_norm):
                return p.slug

    return None


def _nombre_a_slug(nombre: str) -> Optional[str]:
    """Convierte un nombre visible a slug de nodo si existe en relations."""
    relations = _cargar_relations()
    estaciones = relations.get("estaciones", {})
    if nombre in estaciones:
        return nombre
    # Intentar slug directo
    from .loaders import slugificar
    s = slugificar(nombre)
    if s in estaciones:
        return s
    # Buscar por nombre legible en estaciones_list
    for e in relations.get("estaciones_list", []):
        if e.nombre.lower() == nombre.lower() or e.slug == nombre:
            return e.slug
    # Paradas: mapear dirección -> slug de parada
    for p in relations.get("paradas_list", []):
        if p.direccion and p.direccion.lower() == nombre.lower():
            return p.slug
        if p.slug == nombre:
            return p.slug
    return None


# ---------------------------------------------------------------------------
# Motor de búsqueda
# ---------------------------------------------------------------------------

def _contar_transbordos(tramos: List[Tuple[str, str, int]]) -> int:
    """Cuenta transbordos: cambios de ruta o aristas de tipo transbordo."""
    transbordos = 0
    prev_ruta: Optional[str] = None
    for _, codigo_ruta, _ in tramos:
        if not codigo_ruta:
            continue
        if codigo_ruta.startswith("transbordo"):
            transbordos += 1
        elif prev_ruta is not None and codigo_ruta != prev_ruta:
            transbordos += 1
        if not codigo_ruta.startswith("transbordo"):
            prev_ruta = codigo_ruta
    return transbordos


def _nombre_legible(slug: str) -> str:
    """Retorna el nombre legible de un nodo (estación/zona/parada) si existe."""
    relations = _cargar_relations()
    for e in relations.get("estaciones_list", []):
        if e.slug == slug:
            return e.nombre
    if slug.startswith("zona_"):
        zona = slug[len("zona_"):]
        # Buscar nombre original de la zona
        for e in relations.get("estaciones_list", []):
            if e.zona_integration:
                z_norm = _normalizar_nombre(e.zona_integration).replace(" ", "_")
                if z_norm == zona:
                    return f"Zona {e.zona_integration}"
        return f"Zona {zona.replace('_', ' ').title()}"
    for p in relations.get("paradas_list", []):
        if p.slug == slug:
            return p.direccion or p.slug
    return slug


def _tramos_formateados(
    origen: str,
    tramos: List[Tuple[str, str, int]],
) -> List[str]:
    """Formatea los tramos de la ruta como texto (muestra origen -> destino).

    Siempre muestra de dónde sale y a dónde llega cada tramo, empezando en
    el punto de origen, para que la ruta se siga geográficamente.
    """
    lineas: List[str] = []
    actual = origen
    for estacion, codigo_ruta, minutos in tramos:
        origen_txt = _nombre_legible(actual)
        destino_txt = _nombre_legible(estacion)
        if codigo_ruta and codigo_ruta.startswith("transbordo"):
            lineas.append(
                f"  {origen_txt} -> {destino_txt} (caminata transbordo, {minutos} min)"
            )
        elif codigo_ruta == "zona":
            lineas.append(f"  {origen_txt} -> {destino_txt} (integracion zona, {minutos} min)")
        elif codigo_ruta:
            lineas.append(f"  {origen_txt} -> {destino_txt} ({codigo_ruta}, {minutos} min)")
        else:
            lineas.append(f"  {origen_txt} -> {destino_txt} ({minutos} min)")
        actual = estacion
    return lineas


def encontrar_ruta(
    origen_nombre: str,
    destino_nombre: str,
    criterio: str = "tiempo",
    explicitar: bool = False,
) -> Optional[Dict[str, Any]]:
    """Encuentra la mejor ruta desde origen hasta destino.

    Retorna un diccionario con la ruta o None si no hay camino.
    """
    _cargar_normalizador()

    origen = _buscar_estacion(origen_nombre)
    destino = _buscar_estacion(destino_nombre)

    if origen is None:
        print(f"[ERROR] Estacion de origen '{origen_nombre}' no encontrada en la base de conocimiento.")
        return None
    if destino is None:
        print(f"[ERROR] Estacion de destino '{destino_nombre}' no encontrada en la base de conocimiento.")
        return None

    if origen == destino:
        return {
            "origen": origen,
            "destino": destino,
            "tiempo_total": 0,
            "transbordos": 0,
            "tramos": [(origen, "---", 0)],
            "misma_estacion": True,
        }

    relations = _cargar_relations()
    g = construir_grafo(relations)

    resultado = buscar_en_grafo(g, origen, destino, criterio=criterio)

    if resultado is None:
        print(f"[WARN] No hay ruta disponible desde '{origen}' hasta '{destino}'.")
        return None

    _costo, tramos = resultado
    tiempo_total = sum(minutos for _, _, minutos in tramos)
    transbordos = _contar_transbordos(tramos)

    result: Dict[str, Any] = {
        "origen": origen,
        "destino": destino,
        "tiempo_total": tiempo_total,
        "transbordos": transbordos,
        "tramos": tramos,
        "misma_estacion": False,
        "criterio": criterio,
        "costo_criterio": _costo,
    }

    if explicitar:
        result["explicacion"] = _generar_explicacion(
            relations, origen, destino, tramos, criterio
        )

    return result


def _generar_explicacion(
    relations: Dict[str, Any],
    origen: str,
    destino: str,
    tramos: List[Tuple[str, str, int]],
    criterio: str,
) -> str:
    """Genera una explicación detallada de por qué se eligió esta ruta."""
    lineas: List[str] = []

    lineas.append("=== Explicacion del razonamiento ===")
    lineas.append(f"Origen: {_nombre_legible(origen)}")
    lineas.append(f"Destino: {_nombre_legible(destino)}")
    lineas.append(f"Criterio de optimizacion: {criterio}")
    lineas.append("")

    # Historial de inferencias (si backward_chain se ejecuto antes)
    historial = relations.get("_historial", [])
    if historial:
        lineas.append("Reglas e inferencias aplicadas:")
        for i, h in enumerate(historial, 1):
            lineas.append(f"  {i}. {h['regla']}: {h['descripcion']}")
    else:
        if criterio == "tiempo":
            lineas.append("Algoritmo: Dijkstra (minimiza minutos).")
        elif criterio == "transbordos":
            lineas.append("Algoritmo: busqueda por estados (minimiza transbordos).")
        else:
            lineas.append("Algoritmo: busqueda por estados (minimiza tiempo + 5*transbordos).")
        lineas.append("Hechos usados: conecta/4 (conexiones directas entre estaciones).")
    lineas.append("")

    # Tramos de la ruta
    lineas.append("Ruta calculada:")
    lineas.extend(_tramos_formateados(origen, tramos))
    lineas.append("")

    # Estadisticas
    tiempo_total = sum(minutos for _, _, minutos in tramos)
    n_transbordos = _contar_transbordos(tramos)
    lineas.append(f"Tiempo total: {tiempo_total} minutos")
    lineas.append(f"Numero de transbordos: {n_transbordos}")

    return "\n".join(lineas)


# ---------------------------------------------------------------------------
# CLI principal
# ---------------------------------------------------------------------------

def main() -> None:
    """Punto de entrada del CLI."""
    import argparse

    parser = argparse.ArgumentParser(
        description="MIO Router: Encuentra la mejor ruta en el sistema de transporte Masivo de Santiago de Cali.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--origen",
        type=str,
        help="Estacion de origen (ej. 'paso_del_comercio')",
    )

    parser.add_argument(
        "--destino",
        type=str,
        help="Estacion de destino (ej. 'universidades')",
    )

    parser.add_argument(
        "--criterio",
        type=str,
        choices=["tiempo", "transbordos", "equilibrado"],
        default="tiempo",
        help="Criterio de optimizacion: tiempo (minutos), transbordos (minimo cambios), equilibrado (tiempo + penalizacion)",
    )

    parser.add_argument(
        "--explicar",
        action="store_true",
        help="Mostrar explicacion del razonamiento para justificar la ruta elegida",
    )

    parser.add_argument(
        "--listar-estaciones",
        action="store_true",
        help="Lista todas las estaciones de la base de conocimiento",
    )

    parser.add_argument(
        "--fuente",
        type=str,
        choices=["dataset", "kb"],
        default="dataset",
        help="Fuente de datos: dataset (CSV oficial, por defecto) o kb (mio.pl legacy)",
    )

    args = parser.parse_args()
    _cargar_relations(args.fuente)

    # Modo lista
    if args.listar_estaciones:
        _cargar_normalizador()
        relations = _cargar_relations()
        print("\nEstaciones en la base de conocimiento MIO:\n")
        # Preferir lista del dataset (nombres legibles)
        estaciones_list = relations.get("estaciones_list")
        if estaciones_list:
            for e in sorted(estaciones_list, key=lambda x: x.nombre.lower()):
                print(f"  [{e.tipo}] {e.nombre}  (corredor: {e.corredor}, zona: {e.zona_integration})")
        else:
            for eid, (nombre, tipo, zona) in sorted(
                relations.get("estaciones", {}).items(),
                key=lambda x: x[1][0],
            ):
                print(f"  [{tipo}] {nombre} (zona: {zona})")
        # Conteo de paradas externas
        paradas_list = relations.get("paradas_list", [])
        if paradas_list:
            print(f"\nParadas externas cargadas: {len(paradas_list)}")
        print()
        return

    # Modo interactivo (si no se pasan argumentos obligatorios)
    if not args.origen or not args.destino:
        print("=== MIO Router modo interactivo ===")
        print("Estaciones disponibles (mostrando algunas):")
        _cargar_normalizador()
        relations = _cargar_relations()
        estaciones_list = relations.get("estaciones_list") or []
        if estaciones_list:
            muestra = sorted(estaciones_list, key=lambda x: x.nombre.lower())[:15]
            for e in muestra:
                print(f"  [{e.tipo}] {e.nombre}")
        else:
            estaciones = sorted(
                relations.get("estaciones", {}).items(), key=lambda x: x[1][0]
            )
            for eid, (nombre, tipo, zona) in estaciones[:15]:
                print(f"  [{tipo}] {nombre}")
        print("... (use --origen y --destino para buscar una ruta)")
        print("Escriba 'salir' para terminar.\n")

        while True:
            ori = input("Origen: ").strip()
            if ori.lower() in ("salir", "quit", "exit"):
                break
            des = input("Destino: ").strip()
            if des.lower() in ("salir", "quit", "exit"):
                break

            ori_norm = _buscar_estacion(ori)
            des_norm = _buscar_estacion(des)

            if not ori_norm or not des_norm:
                print("[ERROR] Estacion no encontrada. Intente con nombres como:")
                for nombre, _, _ in estaciones[:10]:
                    print(f"  - {nombre}")
                continue

            print()
            resultado = encontrar_ruta(
                ori_norm, des_norm,
                criterio=args.criterio,
                explicitar=args.explicar,
            )
            if resultado is None:
                continue

            print(f"Ruta: {_nombre_legible(resultado['origen'])} -> {_nombre_legible(resultado['destino'])}")
            print(f"Tiempo total: {resultado['tiempo_total']} min")
            print(f"Transbordos: {resultado['transbordos']}")
            print("Tramos:")
            for linea in _tramos_formateados(resultado["origen"], resultado["tramos"]):
                print(linea)

            if args.explicar and not resultado.get("misma_estacion", False):
                print("\n" + resultado.get("explicacion", ""))

            break

    # Modo normal: CLI con argumentos
    if args.origen and args.destino:
        resultado = encontrar_ruta(
            args.origen,
            args.destino,
            criterio=args.criterio,
            explicitar=args.explicar,
        )

        if resultado is None:
            sys.exit(1)

        if resultado.get("misma_estacion"):
            print(f"\n[INFO] {resultado['origen']} y {resultado['destino']} son la misma estacion.")
            print(f"Tiempo: {resultado['tiempo_total']} min")
            return

        print(f"\n[OK] Ruta: {_nombre_legible(resultado['origen'])} -> {_nombre_legible(resultado['destino'])}")
        print(f"Tiempo total: {resultado['tiempo_total']} minutos")
        print(f"Transbordos: {resultado['transbordos']}")
        print("Tramos:")
        for linea in _tramos_formateados(resultado["origen"], resultado["tramos"]):
            print(linea)

        if args.explicar and not resultado.get("misma_estacion", False):
            print("\n" + resultado.get("explicacion", ""))


if __name__ == "__main__":
    main()
