# DISICO — Detección de Anomalías en Consumo Eléctrico (EMCALI)

Pipeline completo para detectar irregularidades en el consumo de energía eléctrica (fraude / hurto de energía) a partir de datos de EMCALI. Combina métodos de detección no supervisados y los evalúa contra verdad de campo proveniente de inspecciones físicas.

---

## Requisitos previos

### Python y UV

Este proyecto usa **[uv](https://docs.astral.sh/uv/)** como gestor de entornos y dependencias. Es más rápido que pip y resuelve dependencias de forma determinista.

**Instalar uv** (si no lo tienes):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# O con pip
pip install uv
```

**Instalar dependencias del proyecto:**

```bash
cd DISICO_anomalies
uv sync
```

Esto crea automáticamente el entorno virtual en `.venv/` e instala todas las dependencias definidas en `pyproject.toml` (pandas, scikit-learn, prophet, tsfresh, lightgbm, statsmodels, etc.).

---

## Datos necesarios (estructura mínima)

Antes de correr cualquier paso debes tener los siguientes archivos crudos de EMCALI en la ruta indicada. **Sin estos archivos el pipeline no puede ejecutarse.** 
La carpeta completa con los datos necesarios, la estructura y resultados se puede descargar en el siguiente link: [/data](https://drive.google.com/file/d/1jQ5VKnxVQZfb55cw2JfO_mroeOYJ7oSZ/view?usp=sharing) 

```
DISICO_anomalies/
└── data/
    └── EMCALI/
        ├── lv_data_comercial.csv                          # Información comercial de clientes
        ├── lv_data_consumo.csv                            # Histórico de consumo mensual
        ├── lv_data_medidor.csv                            # Información de medidores
        ├── lv_lega_insp_irregularidad_report02102025.xlsx # Reporte de irregularidades (lv_lega)
        └── INSPECCIONES_REALIZADAS.xlsx                   # Inspecciones de campo realizadas
```

El resto de carpetas (`datos_procesados/`, `Resultados/`) se crean automáticamente durante el pipeline.

> **Nota sobre tamaños:** `lv_data_consumo.csv` puede superar los 600 MB. El archivo procesado `consumo_*.parquet` (~35 MB) es el que usan los notebooks de modelos, por lo que el parquet es el que más se lee en el día a día.

---

## Paso a paso para correr el pipeline completo

### Paso 1 — Construcción de la base de consumo (corta)

Procesa y cruza los datos comerciales con el histórico de consumo. Genera el archivo base que usan todos los modelos.

```bash
uv run python src/construccion_base.py
```

**Salidas en `data/EMCALI/datos_procesados/`:**
- `consumo_<fecha_inicio>_<fecha_fin>.csv` — consumos con info comercial (~607 MB)
- `info_anomalos.csv` — irregularidades lv_lega con etiqueta `anomalo`
- `info_inspecciones.csv` — inspecciones realizadas con etiqueta `anomalo`
- `info_inspecciones_integrada.csv` — integración de ambas fuentes (ground truth principal)

---

### Paso 2 — Construcción de la serie completa

Extiende la base de consumo para completar la serie temporal de cada cliente (rellena meses faltantes, genera el parquet filtrado que usan los modelos).

Abrir y ejecutar el notebook construccion_serie_completa.ipynb **en orden de arriba a abajo**.

**Salida principal:** `data/EMCALI/datos_procesados/consumo_<fechas>.parquet`

---

### Paso 3 — Construcción del ground truth de inspecciones

Este paso ya se ejecuta dentro de `construccion_base.py` (Paso 1), pero si necesitas regenerar solo el ground truth, puedes correr directamente desde Python:

```bash
uv run python build_anomalos.py
uv run python build_inspecciones.py
```

**Salida:** `data/EMCALI/datos_procesados/info_anomalos.csv`

---

### Paso 4 — Modelos no supervisados

Con la base de consumo ya construida, puedes entrenar los modelos de detección no supervisados. Abrir y ejecutar:

```bash
jupyter notebook notebooks/ans.ipynb
```

Este notebook corre los siguientes métodos en secuencia:

| Método | Descripción |
|--------|-------------|
| **Gaussiano (Mahalanobis)** | Compara consumo vs. mes anterior del año dentro del grupo socioeconómico |
| **Isolation Forest (tsfresh)** | Features estadísticas de la serie + IsolationForest por grupo |
| **Series de tiempo ML** | Consenso ETS / ARIMA / Prophet (anomalía si ≥ 2/3 acuerdan) |
| **CUSUM** | Detección de quiebres estructurales en la serie (α = 0.01) |
| **Markov Regimes** | Regresión con cambio de régimen oculto (Markov switching) |

**Salidas en `data/Resultados/`:** rankings por cliente de cada método (`.csv` y `.parquet`).

> Los métodos con mayor costo computacional (Isolation Forest, Prophet) pueden tardar varias horas o días en el dataset completo. Todos tienen checkpointing: si se interrumpen, reanudan desde el último paso completado.

---

### Paso 5 — Modelos supervisados

Requiere haber completado los Pasos 1–3 (necesita el ground truth de inspecciones y la base de consumo).

```bash
jupyter notebook notebooks/as.ipynb
```

Entrena y evalúa tres modelos de clasificación binaria (`anomalo = 1`):

- **Gradient Boosting** (baseline)
- **Random Forest**
- **LightGBM**

Genera curvas ROC comparativas y predicciones sobre el dataset completo.

**Salidas en `data/Resultados/`:**
- `roc_curves_supervisados.png`
- `supervisado_predicciones_completo.csv`

---

### Paso 6 — Evaluación de modelos no supervisados

Evalúa qué tan bien los rankings no supervisados recuperan los casos inspeccionados usando métricas de recuperación de información (Precision@k, Recall@k, Average Precision).

```bash
jupyter notebook notebooks/evaluacion_no_supervisado.ipynb
```

Usa las funciones de `src/metricas_no_supervisado.py` con k = 100, 500, 1000, 2000, 5000.

**Salida:** `data/Resultados/evaluacion_no_supervisado.png`

---

## Resumen del flujo

```
Datos crudos (data/EMCALI/)
        │
        ▼
[Paso 1] python src/construccion_base.py
        │   → consumo_*.csv  +  info_inspecciones_integrada.csv
        ▼
[Paso 2] notebooks/construccion_serie_completa.ipynb
        │   → consumo_*.parquet  (serie temporal completa)
        ▼
[Paso 3] src/build_anomalos.py
         src/build_inspecciones.py
        │
        ├──► [Paso 4] notebooks/ans.ipynb          (modelos no supervisados)
        │                   │
        │                   ▼
        │            [Paso 6] notebooks/evaluacion_no_supervisado.ipynb
        │
        └──► [Paso 5] notebooks/as.ipynb           (modelos supervisados)
```

---

## Referencia de módulos

| Módulo | Descripción |
|--------|-------------|
| `src/construccion_base.py` | ETL: cruza CSVs crudos, clasifica anomalías |
| `src/metricas_no_supervisado.py` | Métricas de evaluación (Precision@k, Recall@k, AP) |
| `src/pipe_expres/loader.py` | Cargador con caché Excel → pickle |
| `src/pipe_expres/limpieza.py` | Filtrado de clientes, interpolación spline |
| `src/pipe_expres/features.py` | Features simples: log, lag 4 meses, año-mes |
| `src/pipe_expres/features_based.py` | tsfresh + IsolationForest por grupo |
| `src/pipe_expres/gaussiana_method.py` | Distancia de Mahalanobis + visualización |
| `src/pipe_expres/time_series_ml.py` | Consenso ETS / ARIMA / Prophet |
| `src/pipe_expres/cusum_test.py` | Test CUSUM para quiebres estructurales |
| `src/pipe_expres/markov_regimes_anomaly.py` | Markov switching regression |
| `src/pipe_expres/series_anomalas.py` | Identifica series inválidas (varianza cero, >40% nulos) |

### Paso 7 — Dashboard interactivo (DASH-DISICO)

El archivo `src/Dash.py` contiene el tablero interactivo para la visualización
y exploración de los resultados del modelo.

### Pasos para ejecutar el Dash

Desde la raíz del repositorio, correr:
```bash
uv run python src/Dash.py
```

Luego abrir en el navegador el link que aparece en la terminal
(por defecto `http://127.0.0.1:8050`).

> **Nota:** el tiempo de carga inicial puede ser de 5 a 10 minutos,
> ya que al arrancar se pre-computan los análisis de Mahalanobis
> para todos los clientes disponibles. El dashboard estará listo
> cuando aparezca el mensaje `Dash is running on http://127.0.0.1:8050/`
> en la terminal.
