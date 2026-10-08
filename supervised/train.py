"""Entrenamiento y evaluación de los modelos supervisados.

Dos tareas sobre los mismos pares origen→destino:

  - **regresión** del `tiempo` total (minutos): Ridge, Random Forest,
    Gradient Boosting + dos baselines (media y velocidad media de la red);
  - **clasificación** de `transbordos`: Regresión Logística, KNN,
    Random Forest + baseline de clase más frecuente.

Split 80/20 con `random_state` fijo, métricas sobre el conjunto de test y
guardado del bundle en `models/model.joblib` + `models/metrics.json`.
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

MODELS_DIR = os.path.join("models")
MODELO_JOBLIB = os.path.join(MODELS_DIR, "model.joblib")
METRICS_JSON = os.path.join(MODELS_DIR, "metrics.json")

SEED = 42
TEST_SIZE = 0.2


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class BaselineVelocidad:
    """Predice tiempo = distancia_km × velocidad_media (ajustada en train)."""

    def __init__(self) -> None:
        self.min_por_km: float = 1.0

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
# Entrenamiento
# ---------------------------------------------------------------------------

def separar(df: Any, seed: int = SEED, muestra: Optional[int] = None):
    """Split 80/20 (estratificado por nº de transbordos) → 6 salidas.

    Si una clase tiene menos de 2 ejemplos (muestras muy pequeñas) se hace
    el split sin estratificar para no fallar.
    """
    from sklearn.model_selection import train_test_split

    if muestra is not None and muestra < len(df):
        df = df.sample(n=muestra, random_state=seed).reset_index(drop=True)

    X = df[FEATURES].to_numpy(dtype=float)
    y_tiempo = df[COLUMNA_TIEMPO].to_numpy(dtype=float)
    y_trans = df[COLUMNA_TRANSBORDOS].to_numpy()

    comun = dict(test_size=TEST_SIZE, random_state=seed)
    _, conteos = np.unique(y_trans, return_counts=True)
    estratificar = y_trans if conteos.min() >= 2 else None
    try:
        return train_test_split(X, y_tiempo, y_trans, stratify=estratificar, **comun)
    except ValueError:
        # p.ej. hay menos ejemplos de test que clases
        return train_test_split(X, y_tiempo, y_trans, **comun)


def entrenar(
    df: Any = None,
    seed: int = SEED,
    guardar: bool = True,
    dir_modelos: str = MODELS_DIR,
    muestra: Optional[int] = None,
    modelos_tiempo: Optional[Dict[str, Any]] = None,
    modelos_transbordos: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Entrena todos los modelos y (opcionalmente) guarda bundle + métricas."""
    if df is None:
        df = cargar_dataset()

    (
        X_train, X_test,
        y_t_train, y_t_test,
        y_c_train, y_c_test,
    ) = separar(df, seed=seed, muestra=muestra)

    mt = modelos_tiempo if modelos_tiempo is not None else _modelos_tiempo(seed)
    mc = modelos_transbordos if modelos_transbordos is not None else _modelos_transbordos(seed)

    metrics: Dict[str, Any] = {
        "creado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "n_filas": int(len(X_train) + len(X_test)),
        "n_filas_dataset": int(len(df)),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(FEATURES),
        "tiempo": {},
        "transbordos": {},
    }

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

    metrics["mejor_tiempo"] = min(
        metrics["tiempo"], key=lambda n: metrics["tiempo"][n]["mae"]
    )
    metrics["mejor_transbordos"] = max(
        metrics["transbordos"], key=lambda n: metrics["transbordos"][n]["accuracy"]
    )

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
    """Carga el bundle guardado (entrena si no existe)."""
    import joblib

    ruta = os.path.join(dir_modelos, "model.joblib")
    if not os.path.exists(ruta):
        entrenar(guardar=True, dir_modelos=dir_modelos)
    return joblib.load(ruta)


def resumen(metrics: Dict[str, Any]) -> str:
    """Tabla legible de métricas."""
    lineas = [
        f"Filas: {metrics['n_filas']}  (train {metrics['n_train']} / test {metrics['n_test']})",        "",
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
