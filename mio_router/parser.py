"""Parser simple para hechos y reglas en estilo Prolog/Datalog.

Lee un archivo kb/mio.pl y extrae:
  - Hechos: predicados sin cuerpo (ej. estacion(1, paso_del_comercio, terminal, norte)).
  - Reglas: predicados con cuerpo (ej. alcancible(X,Y) :- conecta(X,_,_,_)).
  - Extrae nombre de predicado, argumentos, tipo (hecho/regla) y cuerpo de reglas.
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Any


@dataclass
class Fact:
    """Representa un hecho: predicado(nombre, arg1, arg2, ...)"""
    predicate: str
    args: Tuple[str, ...]


@dataclass
class Rule:
    """Representa una regla: cabeza :-cuerpo."""
    head_predicate: str
    head_args: Tuple[str, ...]
    body_atoms: Tuple[Tuple[str, ...], ...]  # tupla de átomos en el cuerpo


@dataclass
class KnowledgeBase:
    """Base de conocimiento con hechos y reglas."""
    facts: Tuple[Fact, ...]
    rules: Tuple[Rule, ...]


def parse_kb(file_path: str) -> KnowledgeBase:
    """Parsea un archivo de base de conocimiento en estilo Prolog/Datalog.

    Ignora comentarios (líneas que comienzan con %) y líneas en blanco.
    Separa hechos (sin :-) de reglas (con :-).
    """
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    facts = []
    rules = []

    for line in lines:
        line = line.strip()
        if not line or line.startswith("%"):
            continue

        # Normalizar: quitar puntos finales
        line = line.rstrip(".")

        # Intentar parsear como regla (contiene ":-")
        if ":-" in line:
            rule = _parse_rule(line)
            if rule is not None:
                rules.append(rule)
        else:
            # Es un hecho
            fact = _parse_fact(line)
            if fact is not None:
                facts.append(fact)

    return KnowledgeBase(facts=tuple(facts), rules=tuple(rules))


def _parse_fact(line: str) -> Optional[Fact]:
    """Parsea un hecho estilo predicado(arg1, arg2, ...)."""
    # Patrón: nombre(arg1, arg2, ...) sin ":-"
    match = re.match(r'^([a-z_][a-z0-9_]*)\((.*)\)', line, re.IGNORECASE)
    if match:
        predicate = match.group(1).lower()
        args = tuple(a.strip().lower() for a in match.group(2).split(","))
        return Fact(predicate=predicate, args=args)
    return None


def _parse_rule(line: str) -> Optional[Rule]:
    """Parsea una regla estilo cabeza :-cuerpo.

    El cuerpo puede separar átomos con ',' (Prolog clásico, conjunción)
    o con ';' (alternativa). Se soportan ambos.
    """
    # Dividir en cabeza y cuerpo
    parts = line.split(":-", 1)
    if len(parts) != 2:
        return None

    head_str = parts[0].strip()
    body_str = parts[1].strip()

    # Parsear cabeza
    head = _parse_fact(head_str)
    if head is None:
        return None

    # Parsear cuerpo: separado por ',' (and) o ';' (or) — soportar ambos
    body_atoms = []
    if body_str:
        # Reemplazar ';' por ',' para tratarlos como lista de átomos
        # (en este motor la semántica es: todos los átomos deben satisfacerse,
        # es decir conjunción — más cercano a Prolog estándar ',')
        normalized = body_str.replace(";", ",")
        for atom_str in normalized.split(","):
            atom_str = atom_str.strip()
            if atom_str:
                match = re.match(r'^([a-z_][a-z0-9_]*)\((.*)\)', atom_str, re.IGNORECASE)
                if match:
                    pred = match.group(1).lower()
                    args = tuple(a.strip().lower() for a in match.group(2).split(","))
                    body_atoms.append((pred, args))

    return Rule(
        head_predicate=head.predicate,
        head_args=head.args,
        body_atoms=tuple(body_atoms),
    )


def extract_relations(kb: KnowledgeBase) -> Dict[str, List]:
    """Extrae relaciones útiles de la KB para uso del grafo y motor de inferencia.

    Retorna un diccionario con:
      - 'estaciones': dict{id: (nombre, tipo, zona)}
      - 'rutas': dict{codigo: (nombre, tipo)}
      - 'sirve': dict{codigo_ruta: [estaciones]}
      - 'conecta': dict{(origen, destino): (ruta, minutos)}
      - 'transbordo': dict{(estacion, ruta1, ruta2): minutos_espera}
      - 'coordenadas': dict{estacion: (lat, lon)}
    """
    rel = {
        "estaciones": {},
        "rutas": {},
        "sirve": {},
        "conecta": {},
        "transbordo": {},
        "coordenadas": {},
        "facts": kb.facts,  # Hechos raw de la KB para backward chaining
    }

    # Extraer hechos conocidos
    for fact in kb.facts:
        pred = fact.predicate
        args = fact.args

        if pred == "estacion":
            # estacion(ID, nombre, tipo, zona)
            eid, nombre, tipo, zona = args
            rel["estaciones"][eid] = (nombre, tipo, zona)

        elif pred == "ruta":
            # ruta(codigo, nombre, tipo)
            codigo, nombre, tipo = args
            rel["rutas"][codigo] = (nombre, tipo)

        elif pred == "sirve":
            # sirve(codigo_ruta, estacion)
            codigo_ruta, estacion = args
            if codigo_ruta not in rel["sirve"]:
                rel["sirve"][codigo_ruta] = []
            rel["sirve"][codigo_ruta].append(estacion)

        elif pred == "conecta":
            # conecta(origen, destino, ruta, minutos)
            origen, destino, ruta, minutos = args
            key = (origen, destino)
            rel["conecta"][key] = (ruta, int(minutos))

        elif pred == "transbordo":
            # transbordo(estacion, ruta1, ruta2, minutos_espera)
            estacion, ruta1, ruta2, minutos = args
            rel["transbordo"][(estacion, ruta1, ruta2, minutos)] = True

        elif pred == "coordenadas":
            # coordenadas(estacion, lat, lon)
            estacion, lat, lon = args
            rel["coordenadas"][estacion] = (float(lat), float(lon))

    # Extraer reglas para inferencia
    rel["rules"] = kb.rules

    return rel