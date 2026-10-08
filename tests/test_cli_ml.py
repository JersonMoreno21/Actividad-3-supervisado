"""Tests de la CLI de la versión supervisada (python -m supervised)."""

import subprocess
import sys

import pytest


def run(*args, timeout=180):
    return subprocess.run(
        [sys.executable, "-m", "supervised", *args],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def test_help():
    r = run("--help")
    assert r.returncode == 0
    for c in ("generar", "entrenar", "predecir", "evaluar", "listar"):
        assert c in r.stdout


def test_listar_filtro():
    r = run("listar", "--filtro", "univalle")
    assert r.returncode == 0
    assert "univalle" in r.stdout
    assert "81 estaciones" in r.stdout


def test_evaluar_imprime_metricas():
    r = run("evaluar")
    assert r.returncode == 0
    assert "TIEMPO" in r.stdout and "TRANSBORDOS" in r.stdout


def test_predecir_ok():
    r = run("predecir", "--origen", "Univalle", "--destino", "Chiminangos")
    assert r.returncode == 0
    assert "[OK] Viaje: univalle -> chiminangos" in r.stdout
    assert "Dijkstra" in r.stdout


def test_predecir_estacion_inexistente():
    r = run("predecir", "--origen", "NoExiste", "--destino", "Univalle")
    assert r.returncode == 1
    assert "[ERROR]" in r.stderr


def test_entrenar_muestra_rapida(tmp_path):
    r = run(
        "entrenar", "--muestra", "300", "--sin-guardar",
        "--dir-modelos", str(tmp_path),
    )
    assert r.returncode == 0
    assert "TIEMPO" in r.stdout
    assert not (tmp_path / "model.joblib").exists()
