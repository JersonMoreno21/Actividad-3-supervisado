"""Tests de la CLI simbólica (`python -m mio_router`)."""

import subprocess
import sys


def run_mio(inputs: str = "", args=(), timeout: int = 240):
    return subprocess.run(
        [sys.executable, "-m", "mio_router", *args],
        input=inputs,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def test_modo_interactivo_ruta():
    r = run_mio("Univalle\nChiminangos\n")
    assert r.returncode == 0, r.stderr
    assert "Ruta:" in r.stdout
    assert "Tiempo total:" in r.stdout


def test_interactivo_estacion_inexistente_no_crashea():
    """Regresión: `nombres_disponibles` solo se definía en una rama del if y
    una estación inexistente terminaba en UnboundLocalError."""
    r = run_mio("EstacionInexistente\nUnivalle\nsalir\n")
    assert r.returncode == 0, r.stderr
    assert "[ERROR] Estacion no encontrada" in r.stdout
    assert "Traceback" not in r.stderr
    assert "UnboundLocalError" not in r.stderr


def test_ruta_con_argumentos():
    r = run_mio(args=("--origen", "univalle", "--destino", "chiminangos"))
    assert r.returncode == 0, r.stderr
    assert "[OK] Ruta:" in r.stdout


def test_ruta_con_explicacion():
    r = run_mio(args=("--origen", "univalle", "--destino", "chiminangos",
                      "--explicar"))
    assert r.returncode == 0, r.stderr
    assert "Explicacion del razonamiento" in r.stdout
