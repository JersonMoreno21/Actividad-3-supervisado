"""Tests de entrenamiento, métricas y guardado de modelos."""

import json
import os

import numpy as np
import pytest

from supervised.dataset import COLUMNA_DESTINO, COLUMNA_ORIGEN, cargar_dataset
from supervised.features import FEATURES
from supervised.train import (
    METRICS_JSON,
    MODELO_JOBLIB,
    cargar_modelos,
    entrenar,
    indices_split,
    metricas_clasificacion,
    metricas_regresion,
    resumen,
    separar,
)

MUESTRA = 1500


@pytest.fixture(scope="module")
def df():
    return cargar_dataset()


@pytest.fixture(scope="module")
def metrics(df):
    return entrenar(df, guardar=False, muestra=MUESTRA)


def test_estructura_de_metricas(metrics):
    assert metrics["n_filas"] == MUESTRA
    # el split por origen descarta pares origen(train)->destino(test):
    # train + test + descartadas tiene que cuadrar con las filas
    assert metrics["n_train"] + metrics["n_test"] + metrics["n_descartadas"] == MUESTRA
    assert metrics["n_train"] > metrics["n_test"] > 0
    assert metrics["features"] == FEATURES
    for grupo in ("tiempo", "transbordos"):
        assert len(metrics[grupo]) >= 3
    assert metrics["mejor_tiempo"] in metrics["tiempo"]
    assert metrics["mejor_transbordos"] in metrics["transbordos"]


def test_mejor_modelo_elegido_por_cv(metrics):
    """El 'mejor' no puede salir de mirar el test (dejaría de ser held-out)."""
    assert metrics["mejor_seleccionado_por"] == "cv_train"
    assert metrics["evaluacion"]["estrategia"].startswith("held-out")
    assert metrics["cv"]["folds"] >= 2
    assert metrics["cv"]["estrategia"].startswith("GroupKFold")
    # las métricas reportadas siguen siendo las del test
    assert metrics["tiempo"][metrics["mejor_tiempo"]]["mae"] >= 0


def test_tiempos_baten_baselines(metrics):
    t = metrics["tiempo"]
    assert t["random_forest"]["mae"] < t["baseline_media"]["mae"]
    assert t["random_forest"]["mae"] < t["baseline_velocidad"]["mae"]
    assert t["random_forest"]["r2"] > 0.75
    assert t["random_forest"]["rmse"] >= t["random_forest"]["mae"]


def test_transbordos_baten_baseline(metrics):
    c = metrics["transbordos"]
    assert c["random_forest"]["accuracy"] > c["baseline_frecuente"]["accuracy"]
    assert c["random_forest"]["accuracy"] > 0.6
    assert 0.0 <= c["random_forest"]["f1_macro"] <= 1.0


def test_split_reproducible(df):
    a = separar(df, seed=42)
    b = separar(df, seed=42)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))
    c = separar(df, seed=7)
    assert not np.array_equal(np.asarray(a[0]), np.asarray(c[0]))


def test_split_sin_fuga_de_etiquetas(df):
    X_train, X_test, y_t_train, y_t_test, y_c_train, y_c_test = separar(df)
    assert X_train.shape[1] == len(FEATURES)
    assert X_train.shape[0] == len(y_t_train) == len(y_c_train)
    assert X_test.shape[0] == len(y_t_test) == len(y_c_test)


def test_split_particiones_disjuntas_y_completas(df):
    info = indices_split(df)
    d = info["df"]
    tr, te, de = info["train"], info["test"], info["descartados"]
    total = set(range(len(d)))
    assert set(tr) | set(te) | set(de) == total
    assert not set(tr) & set(te)
    assert not set(tr) & set(de)
    assert not set(te) & set(de)
    assert len(te) > 0 and len(de) > 0
    assert info["origenes_test"]


def test_split_sin_fuga_por_espejo(df):
    """Regresión: con un split aleatorio por filas, el 78 % del test tenía su
    par espejo (b,a) ya visto en train (las etiquetas de (a,b) y (b,a) son
    idénticas), y las métricas salían infladas."""
    info = indices_split(df)
    d = info["df"]
    train = d.iloc[info["train"]]
    test = d.iloc[info["test"]]

    pares_train = set(zip(train[COLUMNA_ORIGEN], train[COLUMNA_DESTINO]))
    pares_test = set(zip(test[COLUMNA_ORIGEN], test[COLUMNA_DESTINO]))
    assert not {(b, a) for a, b in pares_train} & pares_test

    # ningún origen del test se ha visto nunca como origen en train
    assert not {a for a, _ in pares_train} & {a for a, _ in pares_test}

    # los orígenes del test son exactamente el grupo de test
    assert {a for a, _ in pares_test} == set(info["origenes_test"])


def test_guardar_y_recargar(df, tmp_path):
    metrics = entrenar(df, guardar=True, dir_modelos=str(tmp_path), muestra=600)
    joblib_path = tmp_path / "model.joblib"
    metrics_path = tmp_path / "metrics.json"
    assert joblib_path.exists()
    assert metrics_path.exists()

    with open(metrics_path, encoding="utf-8") as f:
        guardadas = json.load(f)
    assert guardadas["mejor_tiempo"] == metrics["mejor_tiempo"]

    bundle = cargar_modelos(dir_modelos=str(tmp_path))
    assert bundle["features"] == FEATURES
    assert bundle["mejor_tiempo"] in bundle["modelos_tiempo"]
    assert bundle["mejor_transbordos"] in bundle["modelos_transbordos"]
    assert all(hasattr(m, "predict") for m in bundle["modelos_tiempo"].values())


def test_metricas_helpers():
    y = np.array([10.0, 20.0, 30.0])
    reg = metricas_regresion(y, y)
    assert reg["mae"] == 0.0 and reg["r2"] == 1.0
    cls = metricas_clasificacion(np.array([0, 1, 1]), np.array([0, 1, 1]))
    assert cls["accuracy"] == 1.0


def test_resumen_es_legible(metrics):
    texto = resumen(metrics)
    assert "TIEMPO" in texto and "TRANSBORDOS" in texto
    assert metrics["mejor_tiempo"] in texto


def test_archivos_del_repo_existentes():
    # el entrenamiento definitivo del repositorio debe estar versionado
    assert os.path.exists(METRICS_JSON), "falta models/metrics.json"
