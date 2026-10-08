"""CLI de la versión supervisada: python -m supervised <comando>.

Comandos:
  generar    Construye datasets/od.csv con las etiquetas de Dijkstra (profesor)
  entrenar   Entrena los modelos y guarda models/model.joblib + metrics.json
  predecir   Predice tiempo/transbordos de un viaje y lo compara con Dijkstra
  evaluar    Reimprime las métricas del último entrenamiento
  listar     Lista las estaciones usadas como pares OD
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, List, Optional

from .dataset import DATASET_CSV, cargar_dataset, generar_dataset
from .predict import predecir
from .train import (
    METRICS_JSON,
    MODELS_DIR,
    entrenar,
    resumen,
)


def _cmd_generar(args: argparse.Namespace) -> int:
    df = generar_dataset(ruta=args.salida, verbose=True)
    print(f"Dataset: {args.salida}  ({len(df)} pares OD, {len(df.columns)} columnas)")
    return 0


def _cmd_entrenar(args: argparse.Namespace) -> int:
    df = cargar_dataset(args.dataset)
    metrics = entrenar(
        df,
        seed=args.seed,
        guardar=not args.sin_guardar,
        dir_modelos=args.dir_modelos,
        muestra=args.muestra,
        verbose=args.verbose,
    )
    print(resumen(metrics))
    if not args.sin_guardar:
        print(f"\nModelo: {os.path.join(args.dir_modelos, 'model.joblib')}")
        print(f"Métricas: {os.path.join(args.dir_modelos, 'metrics.json')}")
    return 0


def _cmd_predecir(args: argparse.Namespace) -> int:
    try:
        res = predecir(args.origen, args.destino, dir_modelos=args.dir_modelos)
    except (ValueError, FileNotFoundError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    print(f"[OK] Viaje: {res['origen']} -> {res['destino']}")
    print(f"Predicción (modelo {res['modelos']['tiempo']}): "
          f"{res['tiempo_pred']} min, {res['transbordos_pred']} transbordos")
    print(f"Real (Dijkstra)  : {res['tiempo_real']} min, "
          f"{res['transbordos_real']} transbordos")
    print(f"Error            : {res['error_tiempo']} min, "
          f"{res['error_transbordos']} transbordos")
    if args.todos:
        print("\nPor modelo:")
        for n, v in sorted(res["por_modelo"]["tiempo"].items()):
            print(f"  tiempo       {n:<22} {v:7.1f} min")
        for n, v in sorted(res["por_modelo"]["transbordos"].items()):
            print(f"  transbordos  {n:<22} {v:7d}")
    return 0


def _cmd_evaluar(args: argparse.Namespace) -> int:
    if not os.path.exists(METRICS_JSON):
        print("No hay métricas; entrena primero: python -m supervised entrenar",
              file=sys.stderr)
        return 1
    with open(METRICS_JSON, encoding="utf-8") as f:
        print(resumen(json.load(f)))
    return 0


def _cmd_listar(args: argparse.Namespace) -> int:
    from mio_router.builder import construir_relations

    relations = construir_relations()
    estaciones = {s: v for s, v in relations["estaciones"].items() if v[1] != "zona"}
    corredor = {}
    for c, lista in relations["sirve"].items():
        for e in lista:
            corredor.setdefault(e, c)
    for slug in sorted(estaciones):
        zona = estaciones[slug][2]
        c = corredor.get(slug, "-")
        if args.filtro and args.filtro.lower() not in slug:
            continue
        print(f"{slug:<34} corredor={c:<20} zona={zona}")
    print(f"\n{len(estaciones)} estaciones")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    # Consolas Windows (cp1252): no troncar con caracteres UTF-8 como '→'
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")

    parser = argparse.ArgumentParser(
        prog="python -m supervised",
        description="Versión supervisada (ML) del ruteo del MIO de Cali.",
    )
    sub = parser.add_subparsers(dest="comando", required=True)

    p = sub.add_parser("generar", help="Genera datasets/od.csv con Dijkstra como profesor")
    p.add_argument("--salida", default=DATASET_CSV)
    p.set_defaults(func=_cmd_generar)

    p = sub.add_parser("entrenar", help="Entrena los modelos supervisados")
    p.add_argument("--dataset", default=DATASET_CSV)
    p.add_argument("--dir-modelos", default=MODELS_DIR)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--muestra", type=int, default=None,
                   help="Entrena con N filas al azar (para pruebas rápidas)")
    p.add_argument("--sin-guardar", action="store_true")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(func=_cmd_entrenar)

    p = sub.add_parser("predecir", help="Predice un viaje origen-destino")
    p.add_argument("--origen", required=True)
    p.add_argument("--destino", required=True)
    p.add_argument("--dir-modelos", default=MODELS_DIR)
    p.add_argument("--todos", action="store_true", help="Muestra todos los modelos")
    p.set_defaults(func=_cmd_predecir)

    p = sub.add_parser("evaluar", help="Muestra las métricas del último entrenamiento")
    p.set_defaults(func=_cmd_evaluar)

    p = sub.add_parser("listar", help="Lista las estaciones (pares OD)")
    p.add_argument("--filtro", default=None)
    p.set_defaults(func=_cmd_listar)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
