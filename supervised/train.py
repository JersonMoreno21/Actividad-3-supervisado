"""Entrenamiento y evaluación de los modelos supervisados.

Dos tareas sobre los mismos pares origen→destino:

  - **regresión** del `tiempo` total (minutos): Ridge, Random Forest,
    Gradient Boosting + dos baselines (media y velocidad media de la red);
  - **clasificación** de `transbordos`: Regresión Logística, KNN,
    Random Forest + baseline de clase más frecuente.

Split **held-out por origen** (los pares espejo (a,b)/(b,a) no se cruzan
entre train y test) con `random_state` fijo; el mejor modelo se elige con
validación cruzada sobre train y las métricas finales se calculan una única
vez sobre el test. Guardado del bundle en `models/model.joblib` +
`models/metrics.json`.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np

from .dataset import (
    COLUMNA_DESTINO,
    COLUMNA_ORIGEN,
    COLUMNA_TIEMPO,
    COLUMNA_TRANSBORDOS,
    cargar_dataset,
)
from .features import FEATURES

# Anclado a la raíz del repo (no depende del CWD)
MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"
)
MODELO_JOBLIB = os.path.join(MODELS_DIR, "model.joblib")
METRICS_JSON = os.path.join(MODELS_DIR, "metrics.json")

SEED = 42
TEST_SIZE = 0.2
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class BaselineVelocidad:
    """Predice tiempo = distancia_km × velocidad_media (ajustada en train)."""

    def __init__(self, min_por_km: float = 1.0) -> None:
        self.min_por_km = float(min_por_km)

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        """Compatible con sklearn.base.clone (necesario para la CV)."""
        return {"min_por_km": self.min_por_km}

    def set_params(self, **params: Any) -> "BaselineVelocidad":
        for nombre, valor in params.items():
            if nombre != "min_por_km":
                raise ValueError(f"Parámetro desconocido: {nombre}")
            self.min_por_km = float(valor)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineVelocidad":
        idx = FEATURES.index("dist_km")
        dist = np.maximum(X[:, idx], 1e-6)
        self.min_por_km = float(np.median(y / dist))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = FEATURES.index("dist_km")
        return np.asarray(X, dtype=float)[:, idx] * self.min_por_km


# ---------------------------------------------------------------------------
# Fábricas de modelos (sin ajustar: se clonan por nombre)
# ---------------------------------------------------------------------------

def _modelos_tiempo(seed: int) -> Dict[str, Any]:
    from sklearn.dummy import DummyRegressor
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "baseline_media": DummyRegressor(strategy="mean"),
        "baseline_velocidad": BaselineVelocidad(),
        "ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "random_forest": RandomForestRegressor(
            n_estimators=120, random_state=seed, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingRegressor(random_state=seed),
    }


def _modelos_transbordos(seed: int) -> Dict[str, Any]:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    return {
        "baseline_frecuente": DummyClassifier(strategy="most_frequent"),
        "logreg": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=seed)
        ),
        "knn": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=5)),
        "random_forest": RandomForestClassifier(
            n_estimators=120, random_state=seed, n_jobs=-1
        ),
    }


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def metricas_regresion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "r2": round(float(r2_score(y_true, y_pred)), 4),
    }


def metricas_clasificacion(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_true, y_pred, average="macro", zero_division=0)), 4),
        "mae": round(float(mean_absolute_error(y_true, y_pred)), 3),
    }


# ---------------------------------------------------------------------------
# Split: held-out por origen (sin fuga por pares espejo)
# ---------------------------------------------------------------------------
#
# El dataset contiene (a,b) y (b,a) con la MISMA etiqueta (tiempo idéntico en
# los 8010 pares), así que un split aleatorio por filas deja el 78 % del test
# con su espejo ya visto en train y las métricas salen infladas.
#
# Regla elegida (la más estricta):
#   - los orígenes se parten en grupo de train (80 %) y grupo de test (20 %);
#   - train  = pares cuyo ORIGEN y DESTINO están en el grupo de train;
#   - test   = pares cuyo ORIGEN está en el grupo de test (nunca se ha
#              visto ese origen como origen en train: generaliza a nodos nuevos);
#   - se descartan los pares origen(train) -> destino(test), porque su espejo
#     destino(train) -> origen(test) tendría exactamente la misma etiqueta.

def origenes_test(df: Any, seed: int = SEED) -> List[str]:
    """Elige los orígenes del conjunto de test de forma determinista."""
    import random

    origenes = sorted(df[COLUMNA_ORIGEN].unique())
    if not origenes:
        return []
    rng = random.Random(seed)
    barajados = list(origenes)
    rng.shuffle(barajados)
    n_test = max(1, int(round(len(origenes) * TEST_SIZE)))
    return sorted(barajados[:n_test])


def indices_split(
    df: Any, seed: int = SEED, muestra: Optional[int] = None
) -> Dict[str, Any]:
    """Índices (train/test/descartados) + metadatos del split por origen."""
    if muestra is not None and muestra < len(df):
        df = df.sample(n=muestra, random_state=seed).reset_index(drop=True)

    test_origins = set(origenes_test(df, seed=seed))
    origen_en_test = df[COLUMNA_ORIGEN].isin(test_origins).to_numpy()
    destino_en_test = df[COLUMNA_DESTINO].isin(test_origins).to_numpy()

    mascara_train = ~origen_en_test & ~destino_en_test
    mascara_test = origen_en_test
    mascara_descartada = ~origen_en_test & destino_en_test

    return {
        "train": np.flatnonzero(mascara_train),
        "test": np.flatnonzero(mascara_test),
        "descartados": np.flatnonzero(mascara_descartada),
        "origenes_test": sorted(test_origins),
        "n_origenes": int(df[COLUMNA_ORIGEN].nunique()),
        "df": df,
    }


def _preparar_matrices(df: Any):
    """Matrices X, y_tiempo, y_transbordos a partir del DataFrame."""
    X = df[FEATURES].to_numpy(dtype=float)
    y_tiempo = df[COLUMNA_TIEMPO].to_numpy(dtype=float)
    y_trans = df[COLUMNA_TRANSBORDOS].to_numpy()
    return X, y_tiempo, y_trans


def separar(df: Any, seed: int = SEED, muestra: Optional[int] = None):
    """Split 80/20 **por origen** → 6 salidas (X/y de train y test).

    Ver el comentario de la sección: sin esta regla los pares espejo
    (a,b)/(b,a) filtrarían la etiqueta del test en train.
    """
    split = indices_split(df, seed=seed, muestra=muestra)
    X, y_tiempo, y_trans = _preparar_matrices(split["df"])

    tr, te = split["train"], split["test"]
    return (
        X[tr], X[te],
        y_tiempo[tr], y_tiempo[te],
        y_trans[tr], y_trans[te],
    )


# ---------------------------------------------------------------------------
# Selección del mejor modelo por validación cruzada (NUNCA sobre el test)
# ---------------------------------------------------------------------------
#
# Antes el "mejor" se elegía mirando las métricas del propio conjunto de test:
# eso deja de ser un conjunto de evaluación final. La CV se hace solo sobre
# train, con GroupKFold por origen (una misma estación no aparece en dos
# pliegues a la vez) y el test se toca una única vez al final.

def _splits_por_origen(X: Any, y: Any, grupos: Any, folds: int, seed: int):
    """Genera los pliegues de la CV agrupando por origen."""
    from sklearn.model_selection import GroupKFold

    n_grupos = len(set(grupos))
    n_folds = max(2, min(folds, n_grupos))
    try:
        cv = GroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    except TypeError:  # sklearn < 1.6 no admite shuffle
        cv = GroupKFold(n_splits=n_folds)
    return list(cv.split(X, y, grupos)), n_folds


def _clonar(modelo: Any) -> Any:
    """Copia del modelo sin estado ajustado (una por pliegue)."""
    import copy

    return copy.deepcopy(modelo)


def _cv_regresion(
    modelos: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    grupos: Any,
    folds: int,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    from sklearn.metrics import mean_absolute_error, r2_score

    splits, n_folds = _splits_por_origen(X, y, grupos, folds, seed)
    salidas: Dict[str, Any] = {}
    for nombre, modelo in modelos.items():
        maes, r2s = [], []
        for tr_idx, va_idx in splits:
            m = _clonar(modelo)
            m.fit(X[tr_idx], y[tr_idx])
            pred = m.predict(X[va_idx])
            maes.append(mean_absolute_error(y[va_idx], pred))
            r2s.append(r2_score(y[va_idx], pred))
        salidas[nombre] = {
            "mae": round(float(np.mean(maes)), 3),
            "mae_std": round(float(np.std(maes)), 3),
            "r2": round(float(np.mean(r2s)), 4),
        }
        if verbose:
            print(f"  [cv tiempo] {nombre}: {salidas[nombre]}")
    salidas["_folds"] = n_folds
    return salidas


def _cv_clasificacion(
    modelos: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    grupos: Any,
    folds: int,
    seed: int,
    verbose: bool = False,
) -> Dict[str, Any]:
    from sklearn.metrics import accuracy_score, f1_score

    splits, n_folds = _splits_por_origen(X, y, grupos, folds, seed)
    salidas: Dict[str, Any] = {}
    for nombre, modelo in modelos.items():
        accs, f1s = [], []
        for tr_idx, va_idx in splits:
            m = _clonar(modelo)
            m.fit(X[tr_idx], y[tr_idx])
            pred = m.predict(X[va_idx])
            accs.append(accuracy_score(y[va_idx], pred))
            f1s.append(f1_score(y[va_idx], pred, average="macro", zero_division=0))
        salidas[nombre] = {
            "accuracy": round(float(np.mean(accs)), 4),
            "accuracy_std": round(float(np.std(accs)), 4),
            "f1_macro": round(float(np.mean(f1s)), 4),
        }
        if verbose:
            print(f"  [cv transbordos] {nombre}: {salidas[nombre]}")
    salidas["_folds"] = n_folds
    return salidas


def entrenar(
    df: Any = None,
    seed: int = SEED,
    guardar: bool = True,
    dir_modelos: str = MODELS_DIR,
    muestra: Optional[int] = None,
    modelos_tiempo: Optional[Dict[str, Any]] = None,
    modelos_transbordos: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    cv_folds: int = CV_FOLDS,
) -> Dict[str, Any]:
    """Entrena todos los modelos y (opcionalmente) guarda bundle + métricas.

    El split es **held-out por origen** (`indices_split`) y el modelo ganador
    se elige con validación cruzada sobre train (`cv_folds`, 0 para saltarla).
    """
    if df is None:
        df = cargar_dataset()

    split = indices_split(df, seed=seed, muestra=muestra)
    datos = split["df"]
    X, y_tiempo, y_trans = _preparar_matrices(datos)
    tr, te = split["train"], split["test"]
    X_train, X_test = X[tr], X[te]
    y_t_train, y_t_test = y_tiempo[tr], y_tiempo[te]
    y_c_train, y_c_test = y_trans[tr], y_trans[te]
    grupos_train = datos[COLUMNA_ORIGEN].to_numpy()[tr]

    mt = modelos_tiempo if modelos_tiempo is not None else _modelos_tiempo(seed)
    mc = modelos_transbordos if modelos_transbordos is not None else _modelos_transbordos(seed)

    metrics: Dict[str, Any] = {
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "n_filas": int(len(datos)),
        "n_filas_dataset": int(len(df)),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_descartadas": int(len(split["descartados"])),
        "features": list(FEATURES),
        "evaluacion": {
            "estrategia": "held-out por origen",
            "regla": (
                "train: origen y destino en el grupo de train; test: origen "
                "en el grupo de test; descartados los pares origen(train) -> "
                "destino(test), porque su espejo tendría la misma etiqueta"
            ),
            "origenes": int(split["n_origenes"]),
            "origenes_test": list(split["origenes_test"]),
        },
        "cv": None,
        "tiempo": {},
        "transbordos": {},
    }

    # 1) Selección del mejor modelo con CV sobre train (el test queda intacto)
    mejor_tiempo = mejor_transbordos = None
    if cv_folds and cv_folds >= 2:
        metrics["cv"] = {
            "estrategia": "GroupKFold por origen sobre train",
            "folds": cv_folds,
            "tiempo": _cv_regresion(mt, X_train, y_t_train, grupos_train,
                                    cv_folds, seed, verbose),
            "transbordos": _cv_clasificacion(mc, X_train, y_c_train, grupos_train,
                                             cv_folds, seed, verbose),
        }
        cv_t = {k: v for k, v in metrics["cv"]["tiempo"].items() if k != "_folds"}
        cv_c = {k: v for k, v in metrics["cv"]["transbordos"].items() if k != "_folds"}
        mejor_tiempo = min(cv_t, key=lambda n: cv_t[n]["mae"])
        mejor_transbordos = max(cv_c, key=lambda n: cv_c[n]["accuracy"])

    # 2) Ajuste final sobre todo el train y evaluación una única vez en test
    for nombre, modelo in mt.items():
        modelo.fit(X_train, y_t_train)
        pred = modelo.predict(X_test)
        metrics["tiempo"][nombre] = metricas_regresion(y_t_test, pred)
        if verbose:
            print(f"  [tiempo] {nombre}: {metrics['tiempo'][nombre]}")

    for nombre, modelo in mc.items():
        modelo.fit(X_train, y_c_train)
        pred = modelo.predict(X_test)
        metrics["transbordos"][nombre] = metricas_clasificacion(y_c_test, pred)
        if verbose:
            print(f"  [transbordos] {nombre}: {metrics['transbordos'][nombre]}")

    # Respaldo por si la CV no corrió: en ese caso se elige sobre test
    if mejor_tiempo is None:
        mejor_tiempo = min(metrics["tiempo"], key=lambda n: metrics["tiempo"][n]["mae"])
    if mejor_transbordos is None:
        mejor_transbordos = max(
            metrics["transbordos"], key=lambda n: metrics["transbordos"][n]["accuracy"]
        )
    metrics["mejor_tiempo"] = mejor_tiempo
    metrics["mejor_transbordos"] = mejor_transbordos
    metrics["mejor_seleccionado_por"] = "cv_train" if metrics["cv"] else "test"

    if guardar:
        guardar_modelos(
            {
                "version": "1.0.0",
                "creado": metrics["creado"],
                "seed": seed,
                "features": list(FEATURES),
                "modelos_tiempo": mt,
                "modelos_transbordos": mc,
                "mejor_tiempo": metrics["mejor_tiempo"],
                "mejor_transbordos": metrics["mejor_transbordos"],
            },
            metrics,
            dir_modelos=dir_modelos,
        )

    return metrics


def guardar_modelos(bundle: Dict[str, Any], metrics: Dict[str, Any],
                    dir_modelos: str = MODELS_DIR) -> None:
    """Escribe model.joblib y metrics.json en `dir_modelos`."""
    import joblib

    os.makedirs(dir_modelos, exist_ok=True)
    joblib.dump(bundle, os.path.join(dir_modelos, "model.joblib"))
    with open(os.path.join(dir_modelos, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)


def cargar_modelos(dir_modelos: str = MODELS_DIR) -> Dict[str, Any]:
    """Carga el bundle guardado.

    Antes reentrenaba (y escribía ~93 MB) si faltaba el archivo, sin avisar:
    en un clon limpio eso dejaba el repositorio sucio y tardaba minutos.
    Ahora el error indica exactamente qué comando ejecutar.
    """
    import joblib

    ruta = os.path.join(dir_modelos, "model.joblib")
    if not os.path.exists(ruta):
        raise FileNotFoundError(
            f"No hay modelo entrenado en {ruta}. "
            f"Entrena primero con: python -m supervised entrenar"
        )
    return joblib.load(ruta)


def resumen(metrics: Dict[str, Any]) -> str:
    """Tabla legible de métricas."""
    descartadas = metrics.get("n_descartadas", 0)
    extra = f" / descartadas {descartadas}" if descartadas else ""
    evaluacion = metrics.get("evaluacion") or {}
    lineas = [
        f"Filas: {metrics['n_filas']}  "
        f"(train {metrics['n_train']} / test {metrics['n_test']}{extra})",
        f"Evaluación: {evaluacion.get('estrategia', 'test simple')}  |  "
        f"mejor modelo por: {metrics.get('mejor_seleccionado_por', 'test')}",
        "",
        "TIEMPO (min) — sobre test",
        f"{'modelo':<22}{'MAE':>8}{'RMSE':>8}{'R²':>10}",
    ]
    for nombre, m in metrics["tiempo"].items():
        marca = "  <- mejor" if nombre == metrics["mejor_tiempo"] else ""
        lineas.append(f"{nombre:<22}{m['mae']:>8}{m['rmse']:>8}{m['r2']:>10.3f}{marca}")
    lineas += [
        "",
        "TRANSBORDOS — sobre test",
        f"{'modelo':<22}{'accuracy':>10}{'F1 macro':>10}{'MAE':>8}",
    ]
    for nombre, m in metrics["transbordos"].items():
        marca = "  <- mejor" if nombre == metrics["mejor_transbordos"] else ""
        lineas.append(
            f"{nombre:<22}{m['accuracy']:>10.3f}{m['f1_macro']:>10.3f}{m['mae']:>8.3f}{marca}"
        )
    return "\n".join(lineas)
