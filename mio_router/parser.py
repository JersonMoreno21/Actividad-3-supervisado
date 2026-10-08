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
    with open(file_path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    facts = []
    rules = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Comentario '%' hasta fin de línea (dentro de una regla también)
        line = line.split("%", 1)[0].strip()
        if not line:
            continue

        # Un archivo puede traer varias sentencias en una línea:
        # `hecho(1). otro(2).` Se parte por '.' fuera de paréntesis (los
        # puntos decimales como 3.14 están dentro de un paréntesis y no
        # se tocan). Antes se usaba rstrip("."), que comía TODOS los puntos.
        for sentencia in _partir(line, "."):
            sentencia = sentencia.strip()
            if not sentencia:
                continue

            # Intentar parsear como regla (contiene ":-")
            if ":-" in sentencia:
                rule = _parse_rule(sentencia)
                if rule is not None:
                    rules.append(rule)
            else:
                # Es un hecho
                fact = _parse_fact(sentencia)
                if fact is not None:
                    facts.append(fact)

    return KnowledgeBase(facts=tuple(facts), rules=tuple(rules))


def _partir(texto: str, separadores: str) -> List[str]:
    """Divide `texto` en las posiciones de `separadores` a profundidad 0.

    Respeta los paréntesis: `conecta(X, Y, _, _)` no se trocea en la coma
    que lleva dentro. Sin esto, el cuerpo de todas las reglas de
    `kb/reglas.pl` se parseaba vacío y las reglas nunca se disparaban.
    """
    partes: List[str] = []
    actual: List[str] = []
    profundidad = 0
    for ch in texto:
        if ch == "(":
            profundidad += 1
        elif ch == ")":
            profundidad -= 1
        if profundidad <= 0 and ch in separadores:
            partes.append("".join(actual))
            actual = []
            continue
        actual.append(ch)
    partes.append("".join(actual))
    return partes


def _es_variable(token: str) -> bool:
    """Una variable Prolog empieza por mayúscula o por '_' (X, Y, _Cons)."""
    return bool(token) and (token[0].isupper() or token[0] == "_")


def _parse_args(texto: str) -> Tuple[str, ...]:
    """Parsea la lista de argumentos de un predicado.

    Conserva las variables tal cual (X, Y, _) y normaliza las constantes a
    minúsculas (compatibilidad con la KB legacy). Antes todo se pasaba por
    `.lower()`, lo que convertía X e Y en constantes y hacía que ningún
    predicado con variables pudiera unificarse.
    """
    args: List[str] = []
    for parte in _partir(texto, ","):
        token = parte.strip()
        if _es_variable(token):
            args.append(token)
        else:
            # Argumento vacío ('foo().') se conserva como cadena vacía
            args.append(token.lower())
    return tuple(args)


_ATOM_RE = re.compile(r"^([a-z_][a-z0-9_]*)\((.*)\)$", re.IGNORECASE)


def _parse_atom(texto: str) -> Optional[Tuple[str, Tuple[str, ...]]]:
    """Parsea `pred(arg1, arg2, ...)` → (predicado, argumentos)."""
    match = _ATOM_RE.match(texto.strip())
    if not match:
        return None
    return match.group(1).lower(), _parse_args(match.group(2))


def _parse_fact(line: str) -> Optional[Fact]:
    """Parsea un hecho estilo predicado(arg1, arg2, ...)."""
    atom = _parse_atom(line)
    if atom is None:
        return None
    predicate, args = atom
    return Fact(predicate=predicate, args=args)


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

    # Parsear cuerpo: separado por ',' (and) o ';' (or) — soportar ambos,
    # siempre a profundidad 0 para no romper predicados con varias comas.
    body_atoms: List[Tuple[str, Tuple[str, ...]]] = []
    if body_str:
        # (en este motor la semántica es: todos los átomos deben satisfacerse,
        # es decir conjunción — más cercano a Prolog estándar ',')
        for atom_str in _partir(body_str, ",;"):
            atom_str = atom_str.strip()
            if not atom_str:
                continue
            atom = _parse_atom(atom_str)
            if atom is not None:
                body_atoms.append(atom)

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