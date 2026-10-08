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

## 4. Uso (CLI)

```bash
# 1) Genera el dataset con Dijkstra como profesor (≈ 40 s)
python -m supervised generar

# 2) Entrena los 9 modelos y guarda bundle + métricas (≈ 45 s, con CV de 5 pliegues)
python -m supervised entrenar

# 3) Predice un viaje y compáralo con la ruta real
python -m supervised predecir --origen Univalle --destino Chiminangos --todos

# También por nombre de zona (nodo virtual zona_*)
python -m supervised predecir --origen "Paso del Comercio" --destino Universidades

# Métricas del último entrenamiento / listado de estaciones
python -m supervised evaluar
python -m supervised listar --filtro cali
```

> `predecir` necesita el bundle entrenado (`models/model.joblib`, ~93 MB y
> **no** versionado). Si falta, el comando no reentrena en silencio: avisa de
> que hay que ejecutar `python -m supervised entrenar`.
> Opciones útiles de `entrenar`: `--muestra N` (entrena con N filas; el modelo
> y las métricas guardados quedan marcados como parciales), `--sin-cv` (omite
> la validación cruzada) y `--sin-guardar`.

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

## 7. Demo en Google Colab

Abre `demo.ipynb` en Colab con *Runtime → Run all*. El notebook es
**autocontenido**: clona este repositorio, instala `requirements.txt`, genera el
dataset si hace falta, entrena, grafica (EDA + métricas), predice viajes
concretos y corre los 116 tests.

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
