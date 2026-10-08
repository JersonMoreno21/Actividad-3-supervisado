# Actividad 3 — MIO Router **supervisado** (aprendizaje de máquina)

Versión en **aprendizaje supervisado** del sistema de ruteo del MIO de Cali:
los modelos de `scikit-learn` aprenden a predecir, para un par
**origen → destino**, el **tiempo total** del viaje (regresión) y el **número de
transbordos** (clasificación). Las etiquetas las genera el motor simbólico
original (Dijkstra sobre el grafo MIO), que actúa como **profesor**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JersonMoreno21/Actividad-3-supervisado/blob/master/demo.ipynb)

**Empieza por:** [instalación](#3-instalación) · [ejemplos de uso](#4-ejemplos-de-uso) ·
[resultados](#6-resultados-test-held-out-por-origen-1602-pares) · [pruebas](#8-pruebas)

> **Nota académica:** los datos provienen de los datasets oficiales de
> [MetroCali](https://www.metrocali.gov.co); el experimento es ilustrativo y
> aproximado, con fines académicos.

---

## 1. Idea del experimento

```
data/*.csv (MetroCali) ──► mio_router (grafo MIO) ──► Dijkstra  =  PROFESOR
                                                          │
                                            etiquetas (tiempo, transbordos)
                                                          ▼
   features observables ──────────────────────►  scikit-learn  =  ALUMNO
 (geografía, corredor, zona, topología)          (Random Forest, GBM, Ridge,
                                                  LogReg, KNN + baselines)
                                                          │
                                                          ▼
                              predicción instantánea + comparación con el profesor
```

- **Instancias:** 8010 pares ordenados = 90 nodos (81 estaciones + 9 nodos-zona
  `zona_*`) × 89 destinos distintos.
- **Etiquetas:** `tiempo` (minutos, Dijkstra) y `transbordos` (cambios de ruta o
  tramos de caminata entre plataformas, misma regla que la CLI original).
- **Split: held-out por origen** (`seed=42`), no un muestreo aleatorio de filas:
  - los 90 orígenes se parten en grupo de train (80 %) y grupo de test (20 %);
  - **train** (5112): ambos extremos en el grupo de train;
  - **test** (1602): el origen está en el grupo de test (nunca se ha visto ese
    origen como origen en train);
  - **descartados** (1296): pares origen(train) → destino(test), porque su
    espejo destino(train) → origen(test) tiene **exactamente la misma etiqueta**.
- **Selección del modelo:** validación cruzada `GroupKFold` de 5 pliegues
  (agrupando por origen) **solo sobre train**; el test se toca una única vez al
  final para reportar las métricas.

> Con un split aleatorio por filas, el **78 %** del test tenía su par espejo
> `(b, a)` ya visto en train (las etiquetas de `(a,b)` y `(b,a)` son idénticas)
> y las métricas salían infladas: R² 0,981 y 90 % de acierto, frente a
> **0,962** y **72,9 %** del split honesto.

---

## 2. Estructura

```
Actividad 3 supervisada/
├── data/                      # datasets oficiales de MetroCali (CSV)
├── kb/                        # KB legacy (mio.pl, reglas.pl) del profesor
├── mio_router/                # motor simbólico original (Dijkstra, inferencia, CLI)
├── supervised/                # ← paquete de aprendizaje supervisado
│   ├── dataset.py             # genera datasets/od.csv con Dijkstra como profesor
│   ├── features.py            # 18 features numéricas por par OD
│   ├── train.py               # split, modelos, métricas, guardado (joblib)
│   ├── predict.py             # predice un viaje y lo contrasta con Dijkstra
│   └── __main__.py            # CLI: python -m supervised <comando>
├── datasets/od.csv            # dataset supervisado (8010 × 22)
├── models/metrics.json        # métricas del último entrenamiento (versionado)
│                              # (models/model.joblib se ignora: ~93 MB)
├── demo.ipynb                 # notebook listo para Google Colab
├── conftest.py                # pytest: raíz del repo en sys.path + CWD
├── tests/                     # 71 tests del motor simbólico + 45 del ML
├── requirements.txt
└── README.md
```

---

## 3. Instalación

Requisitos: **Python 3.10+** (probado en 3.14).

```bash
pip install -r requirements.txt
```

> **Windows / Smart App Control:** si `import sklearn` falla con
> *"Application Control policy bloqueó este archivo"*, instala la versión fijada
> en `requirements.txt` (`scikit-learn==1.9.0`); los `.pyd` de 1.9.1 pueden
> quedar bloqueados por la política de integridad de código de Windows 11.

---

## 4. Ejemplos de uso

### 4.1 Motor simbólico — `python -m mio_router`

Ruta óptima (Dijkstra) entre dos estaciones; acepta tildes, mayúsculas y
nombres legibles o slugs:

```bash
python -m mio_router --origen univalle --destino chiminangos
```

```text
[OK] Ruta: Univalle -> Chiminangos
Tiempo total: 67 minutos
Transbordos: 8
Tramos:
  Univalle -> Buitrera (carrera_100, 1 min)
  Buitrera -> Meléndez (caminata transbordo, 7 min)
  Meléndez -> Capri (calle_5, 4 min)
  ...
  Flora Industrial -> Chiminangos (carrera_1, 2 min)
```

Otros criterios de optimización, la explicación del razonamiento y el modo
interactivo con sugerencias:

```bash
# menos transbordos aunque tarde más (92 min, 6 transbordos)
python -m mio_router --origen univalle --destino chiminangos --criterio transbordos

# misma ruta + cadena de razonamiento (algoritmo, hechos usados, tramos)
python -m mio_router --origen univalle --destino chiminangos --explicar

python -m mio_router --listar-estaciones
python -m mio_router            # interactivo: "Origen:" / "Destino:" / "salir"
```

### 4.2 Aprendizaje supervisado — `python -m supervised`

```bash
# 1) Genera datasets/od.csv con Dijkstra como profesor (≈ 40 s)
python -m supervised generar

# 2) Entrena los 9 modelos con CV de 5 pliegues (≈ 45 s)
python -m supervised entrenar
```

```text
Filas: 8010  (train 5112 / test 1602 / descartadas 1296)
Evaluación: held-out por origen  |  mejor modelo por: cv_train

TIEMPO (min) — sobre test
modelo                     MAE    RMSE        R²
baseline_media          11.669   14.37    -0.035
baseline_velocidad       5.683   7.897     0.687
ridge                     4.57   6.139     0.811
random_forest            2.059    2.76     0.962  <- mejor
gradient_boosting        3.183   4.046     0.918
```

Predice un viaje y compáralo con la ruta real del profesor:

```bash
python -m supervised predecir --origen Univalle --destino Chiminangos --todos
```

```text
[OK] Viaje: univalle -> chiminangos
Predicción (modelo random_forest): 67 min, 8 transbordos
Real (Dijkstra)  : 67 min, 8 transbordos
Error            : 0 min, 0 transbordos

Por modelo:
  tiempo       baseline_media            29.9 min
  tiempo       baseline_velocidad        74.7 min
  tiempo       gradient_boosting         66.5 min
  tiempo       random_forest             66.7 min
  tiempo       ridge                     65.1 min
  transbordos  baseline_frecuente           4
  transbordos  knn                          7
  transbordos  logreg                       6
  transbordos  random_forest                8
```

También por nombre de zona (nodo virtual `zona_*`), y las métricas o el
listado de estaciones:

```bash
python -m supervised predecir --origen "Paso del Comercio" --destino Universidades
python -m supervised evaluar
python -m supervised listar --filtro univalle
```

```text
univalle                           corredor=carrera_100          zona=universidades

81 estaciones
```

> `predecir` necesita el bundle entrenado (`models/model.joblib`, ~93 MB y
> **no** versionado). Si falta, el comando no reentrena en silencio: avisa de
> que hay que ejecutar `python -m supervised entrenar`.
> Opciones útiles de `entrenar`: `--muestra N` (entrena con N filas; el modelo
> y las métricas guardados quedan marcados como parciales), `--sin-cv` (omite
> la validación cruzada) y `--sin-guardar`.

### 4.3 Como librería

Ruta óptima con el profesor:

```python
from mio_router.builder import construir_relations
from mio_router.graph import construir_grafo, dijkstra

g = construir_grafo(construir_relations())
tiempo, tramos = dijkstra(g, "univalle", "chiminangos")
print(tiempo, "min")                      # 67 min
print(tramos[0])                          # ('buitrera', 'carrera_100', 1)
```

Inferencia simbólica (backward chaining sobre las 8 reglas de `kb/reglas.pl`):

```python
from mio_router.builder import construir_relations
from mio_router.inference import backward_chain, derivar_requiere_transbordo

rel = construir_relations()

viajes = backward_chain(("viaje_directo", "univalle", "Y", "R"), rel)
print(len(viajes), "viajes directos")     # 2 viajes directos
print(viajes[0])                          # {'Y': 'universidades', 'R': 'carrera_100', 'X': 'univalle'}

tb = derivar_requiere_transbordo("univalle", "chiminangos", rel)
print(len(tb), "transbordos en el viaje") # 12 transbordos en el viaje
print(tb[0]["estacion_transbordo"])       # buitrera
```

Entrenar y predecir con los modelos supervisados:

```python
from supervised.train import entrenar
from supervised.predict import predecir

entrenar()                                # la primera vez crea models/model.joblib
res = predecir("Univalle", "Chiminangos")

print(res["tiempo_pred"], res["tiempo_real"], res["error_tiempo"])        # 67 67 0
print(res["transbordos_pred"], res["transbordos_real"])                  # 8 8
print(res["modelos"])                                                   # {'tiempo': 'random_forest', ...}
```

---

## 5. Features (18) — sin fuga de etiquetas

| Grupo | Features |
|---|---|
| Geografía | `lat_o, lon_o, lat_d, lon_d, dist_km, dlat, dlon` |
| Corredor / zona | `idx_corredor_o, idx_corredor_d, idx_zona_o, idx_zona_d, mismo_corredor, misma_zona` |
| Topología | `arista_directa` (¿adyacentes en el grafo?), `grado_o`, `grado_d` |
| Categoría | `tipo_o`, `tipo_d` (estación / terminal / parada / zona) |

Ninguna feature proviene de la ruta calculada: todo es observable **antes** de
correr Dijkstra, evitando *data leakage*.

---

## 6. Resultados (test held-out por origen: 1602 pares)

Las métricas de abajo se calculan **una sola vez** sobre el test; el modelo
ganador se elige con la CV sobre train (GroupKFold por origen, 5 pliegues).

### Tiempo total (regresión)

| Modelo | MAE (min) | RMSE | R² |
|---|---|---|---|
| baseline_media | 11,669 | 14,370 | −0,035 |
| baseline_velocidad | 5,683 | 7,897 | 0,687 |
| ridge | 4,570 | 6,139 | 0,811 |
| gradient_boosting | 3,183 | 4,046 | 0,918 |
| **random_forest** | **2,059** | **2,760** | **0,962** |

### Transbordos (clasificación)

| Modelo | Accuracy | F1 macro | MAE |
|---|---|---|---|
| baseline_frecuente | 0,250 | 0,040 | 1,677 |
| logreg | 0,451 | 0,344 | 1,059 |
| knn | 0,661 | 0,698 | 0,433 |
| **random_forest** | **0,729** | **0,766** | **0,303** |

Sobre el test, Random Forest acierta el tiempo en **±2 min el 59,6 %** de las
veces (mediana de error 1,61 min) y acierta el número exacto de transbordos en
el **72,9 %** de los casos.

La CV sobre train elige `random_forest` para las dos tareas (MAE 2,335 ± 0,141;
accuracy 0,677 ± 0,091), coherente con el test: no hay atajo mirando el test.

---

## 7. Demo (`demo.ipynb`)

El notebook [`demo.ipynb`](demo.ipynb) se puede abrir tal cual en GitHub o en
[Google Colab](https://colab.research.google.com/github/JersonMoreno21/Actividad-3-supervisado/blob/master/demo.ipynb)
(*Runtime → Run all*). Es **autocontenido**: clona este repositorio, instala
`requirements.txt`, genera el dataset si hace falta, entrena, grafica (EDA +
métricas), predice viajes concretos y corre los 116 tests.

---

## 8. Pruebas

```bash
python -m pytest tests/ -q
```

Resultado: **116 tests** — 71 del motor simbólico original (parser, inferencia,
grafo, loaders y CLI) y 45 del pipeline supervisado (dataset, features,
entrenamiento, predicción y CLI). `conftest.py` deja la raíz del repo en
`sys.path` y como directorio de trabajo, así que el comando funciona desde
cualquier carpeta.

---

## 9. Relación con la versión simbólica

| | Versión simbólica (Actividad 2) | Versión supervisada (esta) |
|---|---|---|
| Método | reglas + búsqueda en grafos | aprendizaje supervisado |
| Respuesta | ruta óptima **garantizada** (Dijkstra) | **estimación** aprendida del profesor |
| Explicación | cadena de inferencia | importancia de features / métricas |
| Coste | cálculo por consulta | inferencia instantánea tras entrenar |
| Dependencia | ninguna | `scikit-learn`, `pandas`, `numpy` |

---

## 10. Referencias

- Fuente oficial de datos del MIO: [https://www.metrocali.gov.co](https://www.metrocali.gov.co)
- Russell, S. & Norvig, P. — *Artificial Intelligence: A Modern Approach*
- Documentación de scikit-learn: https://scikit-learn.org
- Documentación de Python: https://docs.python.org
