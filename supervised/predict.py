"""Predicción de un viaje origen→destino con los modelos entrenados."""

from __future__ import annotations

import difflib
from typing import Any, Dict, Optional

from mio_router.graph import dijkstra

from .dataset import COLUMNA_TIEMPO, COLUMNA_TRANSBORDOS, contar_transbordos, preparar
from .features import FEATURES, MetaFeatures, vector_features
from .train import MODELS_DIR, cargar_modelos


def resolver_estacion(nombre: str, meta: MetaFeatures) -> str:
    """Resuelve un nombre de estación o zona a su nodo del grafo.

    Acepta tildes, mayúsculas, espacios y guiones ('Paso del Comercio',
    'paso_del_comercio', 'UNIVALLE'); las zonas se resuelven al nodo `zona_*`.
    """
    from .features import normalizar_nombre

    clave = normalizar_nombre(nombre)
    for candidato in (clave, meta.nombre_a_slug.get(clave, "")):
        if candidato in meta.estaciones:
            return candidato

    claves = list(meta.estaciones) + list(meta.nombre_a_slug)
    sugerencias = difflib.get_close_matches(clave, claves, n=3, cutoff=0.6)
    sugerencias = list(dict.fromkeys(
        meta.nombre_a_slug.get(s, s) for s in sugerencias
    ))
    extra = f" ¿Quisiste decir: {', '.join(sugerencias)}?" if sugerencias else ""
    raise ValueError(f"Estación desconocida: {nombre!r}.{extra}")


def _estimar_real(origen: str, destino: str, g: Any) -> Dict[str, int]:
    """Etiquetas reales del profesor (Dijkstra) para comparar la predicción."""
    res = dijkstra(g, origen, destino)
    if res is None:
        return {COLUMNA_TIEMPO: -1, COLUMNA_TRANSBORDOS: -1}
    tiempo, tramos = res
    return {
        COLUMNA_TIEMPO: int(tiempo),
        COLUMNA_TRANSBORDOS: contar_transbordos(tramos),
    }


def predecir(
    origen: str,
    destino: str,
    bundle: Optional[Dict[str, Any]] = None,
    dir_modelos: str = MODELS_DIR,
    relations: Optional[Dict[str, Any]] = None,
    g: Any = None,
    meta: Optional[MetaFeatures] = None,
) -> Dict[str, Any]:
    """Predice tiempo y transbordos de un viaje y los contrasta con Dijkstra."""
    # Solo se reconstruye lo que falte: antes, pasar solo `g` o solo `meta`
    # hacía que ambos se reconstruyeran y el grafo del llamante se ignorara.
    if meta is None or g is None:
        rel_local, g_local, meta_local = preparar(relations)
        if relations is None:
            relations = rel_local
        if g is None:
            g = g_local
        if meta is None:
            meta = meta_local

    o = resolver_estacion(origen, meta)
    d = resolver_estacion(destino, meta)
    if o == d:
        raise ValueError("El origen y el destino deben ser estaciones distintas.")

    if bundle is None:
        bundle = cargar_modelos(dir_modelos)

    # El bundle se entrenó con unas features: si cambiaron, la predicción
    # saldría silenciosamente mal.
    if list(bundle.get("features", FEATURES)) != list(FEATURES):
        raise ValueError(
            "El modelo guardado usa un conjunto de features distinto al "
            "actual. Reentrena: python -m supervised entrenar"
        )

    x = [vector_features(o, d, meta)]

    preds_tiempo = {n: float(m.predict(x)[0])
                    for n, m in bundle["modelos_tiempo"].items()}
    preds_trans = {n: int(round(float(m.predict(x)[0])))
                   for n, m in bundle["modelos_transbordos"].items()}

    mejor_t = bundle["mejor_tiempo"]
    mejor_r = bundle["mejor_transbordos"]
    # Redondeo de modelos lineales/GB que pueden predecir minutos negativos
    tiempo_pred = max(0, int(round(preds_tiempo[mejor_t])))
    trans_pred = max(0, preds_trans[mejor_r])

    real = _estimar_real(o, d, g)
    sin_ruta = real[COLUMNA_TIEMPO] < 0
    tiempo_real = None if sin_ruta else real[COLUMNA_TIEMPO]
    trans_real = None if sin_ruta else real[COLUMNA_TRANSBORDOS]

    return {
        "origen": o,
        "destino": d,
        "tiempo_pred": tiempo_pred,
        "transbordos_pred": trans_pred,
        "tiempo_real": tiempo_real,
        "transbordos_real": trans_real,
        "sin_ruta": sin_ruta,
        "error_tiempo": None if sin_ruta else abs(tiempo_pred - real[COLUMNA_TIEMPO]),
        "error_transbordos": (
            None if sin_ruta else abs(trans_pred - real[COLUMNA_TRANSBORDOS])
        ),
        "modelos": {"tiempo": mejor_t, "transbordos": mejor_r},
        "por_modelo": {"tiempo": preds_tiempo, "transbordos": preds_trans},
        "features": dict(zip(FEATURES, x[0])),
    }
