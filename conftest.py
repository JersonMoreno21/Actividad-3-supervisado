"""Configuración de pytest.

- Pone la raíz del repositorio en `sys.path` para que `import mio_router` e
  `import supervised` funcionen aunque pytest se lance desde otro directorio.
- Cambia el directorio de trabajo a la raíz: los tests antiguos referencian
  rutas relativas (`kb/mio.pl`, `datasets/od.csv`).
"""

import os
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))

if RAIZ not in sys.path:
    sys.path.insert(0, RAIZ)

if os.getcwd() != RAIZ:
    os.chdir(RAIZ)
