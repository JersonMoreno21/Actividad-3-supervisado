# Actividad 3 — MIO Router **supervisado** (aprendizaje de máquina)

Versión en **aprendizaje supervisado** del sistema de ruteo del MIO de Cali:
los modelos de `scikit-learn` aprenden a predecir, para un par
**origen → destino**, el **tiempo total** del viaje (regresión) y el **número de
transbordos** (clasificación). Las etiquetas las genera el motor simbólico
original (Dijkstra sobre el grafo MIO), que actúa como **profesor**.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JersonMoreno21/Actividad-3-supervisado/blob/master/demo.ipynb)

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
- **Split:** 80/20, `random_state=42`, estratificado por nº de transbordos.

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
├── demo.ipynb                 # notebook listo para Google Colab
├── tests/                     # 44 tests del profesor + 36 del pipeline ML
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

## 4. Uso (CLI)

```bash
# 1) Genera el dataset con Dijkstra como profesor (≈ 15 s)
python -m supervised generar

# 2) Entrena los 9 modelos y guarda bundle + métricas (≈ 30 s)
python -m supervised entrenar

# 3) Predice un viaje y compáralo con la ruta real
python -m supervised predecir --origen Univalle --destino Chiminangos --todos

# También por nombre de zona (nodo virtual zona_*)
python -m supervised predecir --origen "Paso del Comercio" --destino Universidades

# Métricas del último entrenamiento / listado de estaciones
python -m supervised evaluar
python -m supervised listar --filtro cali
```

Ejemplo de salida:

```text
[OK] Viaje: univalle -> chiminangos
Predicción (modelo random_forest): 67 min, 8 transbordos
Real (Dijkstra)  : 67 min, 8 transbordos
Error            : 0 min, 0 transbordos
```

### Como librería

```python
from supervised.dataset import cargar_dataset
from supervised.train import entrenar, cargar_modelos
from supervised.predict import predecir

df = cargar_dataset()
metrics = entrenar(df, guardar=True, verbose=True)
res = predecir("Univalle", "Chiminangos")
print(res["tiempo_pred"], res["tiempo_real"], res["error_tiempo"])
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

## 6. Resultados (test 20 %, 1602 pares)

### Tiempo total (regresión)

| Modelo | MAE (min) | RMSE | R² |
|---|---|---|---|
| baseline_media | 12,649 | 15,484 | −0,001 |
| baseline_velocidad | 5,865 | 8,293 | 0,713 |
| ridge | 4,636 | 6,451 | 0,826 |
| gradient_boosting | 2,870 | 3,827 | 0,939 |
| **random_forest** | **1,443** | **2,162** | **0,981** |

### Transbordos (clasificación)

| Modelo | Accuracy | F1 macro | MAE |
|---|---|---|---|
| baseline_frecuente | 0,212 | 0,032 | 1,729 |
| logreg | 0,449 | 0,345 | 0,978 |
| knn | 0,737 | 0,655 | 0,313 |
| **random_forest** | **0,900** | **0,800** | **0,110** |

Sobre el test completo, Random Forest acierta el tiempo en **±2 min el 76 %**
de las veces (mediana de error 0,99 min) y acierta el número exacto de
transbordos en el **90 %** de los casos.

---

## 7. Demo en Google Colab

Abre `demo.ipynb` en Colab con *Runtime → Run all*. El notebook es
**autocontenido**: clona este repositorio, instala `requirements.txt`, genera el
dataset si hace falta, entrena, grafica (EDA + métricas), predice viajes
concretos y corre los 80 tests.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/JersonMoreno21/Actividad-3-supervisado/blob/master/demo.ipynb)

Para que el enlace anterior funcione, sube el proyecto a GitHub:

```bash
git init
git add .
git commit -m "Actividad 3: versión supervisada del ruteo MIO"
git branch -M master
git remote add origin https://github.com/JersonMoreno21/Actividad-3-supervisado.git
git push -u origin master
```

---

## 8. Pruebas

```bash
python -m pytest tests/ -q
```

Resultado: **80 tests** — 44 del motor simbólico original (parser, inferencia,
grafo, loaders) y 36 del pipeline supervisado (dataset, features, entrenamiento,
predicción y CLI).

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
