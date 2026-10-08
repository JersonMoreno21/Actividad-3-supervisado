"""Versión supervisada (aprendizaje de máquina) del sistema de ruteo MIO.

El motor simbólico original (Dijkstra sobre el grafo MIO) actúa como
"profesor" que genera las etiquetas; los modelos de scikit-learn aprenden a
predecir el tiempo total y el número de transbordos de un viaje origen→destino
a partir solo de features observables (geografía, corredor, zona, topología).
"""

from __future__ import annotations

from typing import List

__version__ = "1.0.0"

from .features import FEATURES  # noqa: E402

__all__ = ["FEATURES", "__version__"]
