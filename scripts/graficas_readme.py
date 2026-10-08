"""Genera las imágenes del README (docs/img/*.png).

Uso:
    python scripts/graficas_readme.py

Requiere `datasets/od.csv` (python -m supervised generar) y
`models/model.joblib` (python -m supervised entrenar).
"""

from __future__ import annotations

import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RAIZ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, RAIZ)

from supervised.dataset import cargar_dataset  # noqa: E402
from supervised.predict import predecir  # noqa: E402
from supervised.train import cargar_modelos, separar  # noqa: E402

IMG_DIR = os.path.join(RAIZ, "docs", "img")

AZUL, NARANJA, VERDE = "steelblue", "darkorange", "seagreen"


def guardar(fig, nombre: str) -> None:
    os.makedirs(IMG_DIR, exist_ok=True)
    ruta = os.path.join(IMG_DIR, nombre)
    fig.savefig(ruta, dpi=110, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {os.path.relpath(ruta, RAIZ)}  ({os.path.getsize(ruta) / 1024:.0f} KB)")


def grafica_eda(df) -> None:
    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.2))

    ejes[0].hist(df["tiempo"], bins=40, color=AZUL, edgecolor="white")
    ejes[0].set_title("Distribución del tiempo del viaje")
    ejes[0].set_xlabel("minutos (Dijkstra)")
    ejes[0].set_ylabel("pares origen-destino")

    ejes[1].hist(df["transbordos"], bins=range(int(df["transbordos"].max()) + 2),
                 color=NARANJA, edgecolor="white", align="left")
    ejes[1].set_title("Distribución de transbordos")
    ejes[1].set_xlabel("número de transbordos")

    ejes[2].scatter(df["dist_km"], df["tiempo"], s=6, alpha=0.25, color=VERDE)
    ejes[2].set_title("Distancia vs tiempo (profesor)")
    ejes[2].set_xlabel("distancia en línea recta (km)")
    ejes[2].set_ylabel("minutos")

    plt.tight_layout()
    guardar(fig, "01_eda.png")


def grafica_modelos(metrics) -> None:
    fig, ejes = plt.subplots(1, 2, figsize=(15, 4.4))

    t = metrics["tiempo"]
    nombres = list(t)
    maes = [t[n]["mae"] for n in nombres]
    colores = [AZUL if n == metrics["mejor_tiempo"] else "#9fb8d4" for n in nombres]
    ejes[0].barh(nombres, maes, color=colores)
    ejes[0].invert_yaxis()
    ejes[0].set_title("Tiempo: MAE sobre el test (menor = mejor)")
    ejes[0].set_xlabel("minutos de error medio")
    for i, v in enumerate(maes):
        ejes[0].text(v + 0.15, i, f"{v:.2f}", va="center", fontsize=9)

    c = metrics["transbordos"]
    nombres_c = list(c)
    accs = [c[n]["accuracy"] for n in nombres_c]
    colores_c = [NARANJA if n == metrics["mejor_transbordos"] else "#f5c08a"
                 for n in nombres_c]
    ejes[1].barh(nombres_c, accs, color=colores_c)
    ejes[1].invert_yaxis()
    ejes[1].set_xlim(0, 1)
    ejes[1].set_title("Transbordos: exactitud sobre el test (mayor = mejor)")
    ejes[1].set_xlabel("accuracy")
    for i, v in enumerate(accs):
        ejes[1].text(v + 0.015, i, f"{v:.3f}", va="center", fontsize=9)

    plt.tight_layout()
    guardar(fig, "02_metricas.png")


def grafica_pred_vs_real(metrics) -> dict:
    df = cargar_dataset()
    X_train, X_test, y_t_train, y_t_test, y_c_train, y_c_test = separar(df)
    bundle = cargar_modelos()

    modelo_t = bundle["modelos_tiempo"][bundle["mejor_tiempo"]]
    modelo_c = bundle["modelos_transbordos"][bundle["mejor_transbordos"]]
    pred_t = modelo_t.predict(X_test)
    pred_c = modelo_c.predict(X_test)
    err_t = np.abs(pred_t - y_t_test)
    err_c = np.abs(np.round(pred_c) - y_c_test)

    fig, ejes = plt.subplots(1, 3, figsize=(15, 4.4))

    limite = max(y_t_test.max(), pred_t.max()) * 1.05
    ejes[0].scatter(y_t_test, pred_t, s=8, alpha=0.4, color=AZUL)
    ejes[0].plot([0, limite], [0, limite], "--", color="#444444", lw=1)
    ejes[0].set_xlim(0, limite)
    ejes[0].set_ylim(0, limite)
    ejes[0].set_title(f"Predicho vs real ({bundle['mejor_tiempo']})")
    ejes[0].set_xlabel("minutos reales (Dijkstra)")
    ejes[0].set_ylabel("minutos predichos")

    ejes[1].hist(err_t, bins=40, color=AZUL, edgecolor="white")
    ejes[1].set_title("Error absoluto en minutos")
    ejes[1].set_xlabel("|predicho − real|")

    ejes[2].hist(err_c, bins=range(int(err_c.max()) + 2), color=NARANJA,
                 edgecolor="white", align="left")
    ejes[2].set_title("Error absoluto en transbordos")
    ejes[2].set_xlabel("|predicho − real|")

    plt.tight_layout()
    guardar(fig, "03_pred_vs_real.png")

    return {
        "mejor_tiempo": bundle["mejor_tiempo"],
        "mejor_transbordos": bundle["mejor_transbordos"],
        "mae": float(np.mean(err_t)),
        "mediana": float(np.median(err_t)),
        "p90": float(np.percentile(err_t, 90)),
        "pct_2min": float(np.mean(err_t <= 2)),
        "exactitud_trans": float(np.mean(np.round(pred_c) == y_c_test)),
        "mae_trans": float(np.mean(err_c)),
    }


def ejemplo_viaje() -> dict:
    res = predecir("Univalle", "Chiminangos")
    return {
        "viaje": f"{res['origen']} -> {res['destino']}",
        "pred": res["tiempo_pred"],
        "real": res["tiempo_real"],
        "trans_pred": res["transbordos_pred"],
        "trans_real": res["transbordos_real"],
        "por_modelo": res["por_modelo"],
    }


def main() -> None:
    import json

    df = cargar_dataset()
    with open(os.path.join(RAIZ, "models", "metrics.json"), encoding="utf-8") as f:
        metrics = json.load(f)

    print("Generando gráficas del README:")
    grafica_eda(df)
    grafica_modelos(metrics)
    stats = grafica_pred_vs_real(metrics)
    viaje = ejemplo_viaje()

    resumen = {**stats, "viaje": viaje, "modelos": {
        "tiempo": metrics["mejor_tiempo"],
        "transbordos": metrics["mejor_transbordos"],
    }}
    print("\nResumen (test held-out):")
    for k, v in resumen.items():
        if k != "viaje":
            print(f"  {k}: {v}")
    print("  viaje:", viaje["viaje"], viaje["pred"], "min vs", viaje["real"], "min")


if __name__ == "__main__":
    main()
