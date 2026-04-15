import os
import hmac
import time as _time
from pathlib import Path
from collections import defaultdict as _defaultdict
import pickle
import pandas as pd
import numpy as np

from dash import Dash, html, dcc, dash_table, Input, Output, State, no_update
from dash.exceptions import PreventUpdate
import plotly.express as px
import plotly.graph_objects as go
from flask import (session as _flask_session,
                   request as _flask_request,
                   redirect as _flask_redirect)

# =========================================================
# 1) RUTAS
# =========================================================
# Permitir configurar ruta de datos via variable de entorno
# Por defecto: misma carpeta que el script (modo local)
# En Docker: /app/data (via volumen)
DATA_PATH = Path(os.environ.get("DATA_PATH", Path(__file__).resolve().parent))
CACHE_PATH = Path(os.environ.get("CACHE_PATH", DATA_PATH / "cache"))

FILE_SUP          = DATA_PATH / "top_5000_supervisado.xlsx"
FILE_UNSUP        = DATA_PATH / "top_5000_no_supervisado.xlsx"
FILE_CONSUMO      = DATA_PATH / "consumo_011223_01062025_filtrado.parquet"
FILE_ROC          = DATA_PATH / "informacion_curva_roc.pickle"
FILE_GEO          = DATA_PATH / "georeferencias.csv"
FILE_RESULTADOS_1 = DATA_PATH / "resultados_1.parquet"
FILE_RESDF_LAST   = DATA_PATH / "resultados_df_last.csv"
FILE_MAH_CACHE    = CACHE_PATH / "mahalanobis_cache.pkl"

# =========================================================
# 2) FUNCIONES DE CARGA
# =========================================================
def load_supervised(path):
    df = pd.read_excel(path, engine="openpyxl")
    id_col    = "producto" if "producto" in df.columns else df.columns[1]
    score_col = ("prediccion_lgbm" if "prediccion_lgbm" in df.columns else
                 next((c for c in ["prediccion","score_supervisado","score","pred"]
                       if c in df.columns),
                      [c for c in df.select_dtypes("number").columns if c != id_col][0]))
    df = df[[id_col, score_col]].copy()
    df = df.rename(columns={id_col:"producto", score_col:"score_supervisado"})
    df["producto"]          = pd.to_numeric(df["producto"],          errors="coerce")
    df["score_supervisado"] = pd.to_numeric(df["score_supervisado"], errors="coerce")
    df = df.dropna(subset=["producto","score_supervisado"])
    return df.groupby("producto", as_index=False)["score_supervisado"].max()


def load_unsupervised(path):
    df = pd.read_excel(path, engine="openpyxl")
    id_col    = ("producto" if "producto" in df.columns else
                 next((c for c in ["CLIENTE_ID","CLIENTES_ID","cliente_id"] if c in df.columns),
                      df.columns[1]))
    score_col = ("anomaly_score" if "anomaly_score" in df.columns else
                 next((c for c in ["fraud_score","score_no_supervisado","score"]
                       if c in df.columns),
                      [c for c in df.select_dtypes("number").columns if c != id_col][0]))
    df = df[[id_col, score_col]].copy()
    df = df.rename(columns={id_col:"producto", score_col:"score_no_supervisado"})
    df["producto"]             = pd.to_numeric(df["producto"],             errors="coerce")
    df["score_no_supervisado"] = pd.to_numeric(df["score_no_supervisado"], errors="coerce")
    df = df.dropna(subset=["producto","score_no_supervisado"])
    return df.groupby("producto", as_index=False)["score_no_supervisado"].max()


def load_consumo(path, valid_ids=None):
    wanted    = ["fecha","consumo","CLIENTE_ID","localidad","barrio",
                 "tipo_producto","categoria","subcategoria",
                 "consumo_prev","consumo_actual","mes"]
    # Leer solo columnas disponibles sin cargar datos completos primero
    import pyarrow.parquet as pq
    schema = pq.read_schema(path)
    cols = [c for c in wanted if c in schema.names]
    df = pd.read_parquet(path, columns=cols)
    df["fecha"]      = pd.to_datetime(df["fecha"],      errors="coerce")
    df["consumo"]    = pd.to_numeric(df["consumo"],     errors="coerce")
    df["CLIENTE_ID"] = pd.to_numeric(df["CLIENTE_ID"],  errors="coerce")
    df = df.dropna(subset=["fecha","consumo","CLIENTE_ID"])
    if valid_ids is not None:
        df = df[df["CLIENTE_ID"].isin(valid_ids)]
    for col in ["localidad","barrio","tipo_producto","categoria","subcategoria"]:
        if col in df.columns:
            df[col] = (df[col].astype(str).str.strip()
                               .str.rstrip(".0")
                               .replace({"nan":pd.NA,"<NA>":pd.NA}))
    return df


def load_roc(path):
    try:
        with open(path,"rb") as f: return pickle.load(f), None
    except Exception as e:         return None, str(e)


def load_geo(path, valid_ids=None):
    try:
        df = pd.read_csv(path, usecols=lambda c: c.strip().lower() in
                         {"producto","cliente_id","clientes_id","lat","lon"})
        df.columns = [c.strip().lower() for c in df.columns]
        for alt in ["cliente_id","clientes_id"]:
            if alt in df.columns and "producto" not in df.columns:
                df = df.rename(columns={alt:"producto"})
        df["producto"] = pd.to_numeric(df["producto"], errors="coerce")
        df["lat"]      = pd.to_numeric(df["lat"],      errors="coerce")
        df["lon"]      = pd.to_numeric(df["lon"],      errors="coerce")
        if valid_ids is not None:
            df = df[df["producto"].isin(valid_ids)]
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)



# =========================================================
# 3) CARGA INICIAL (optimizada para memoria)
# =========================================================
import gc as _gc

print("[Dash] Cargando Excel de scores...", flush=True)
df_sup = df_unsup = pd.DataFrame()
_load_errors = []

try:
    df_sup = load_supervised(FILE_SUP)
except Exception as e:
    _load_errors.append(f"supervisado: {e}")

try:
    df_unsup = load_unsupervised(FILE_UNSUP)
except Exception as e:
    _load_errors.append(f"no supervisado: {e}")

DATA_OK    = not (df_sup.empty and df_unsup.empty)
LOAD_ERROR = "\n".join(_load_errors)

# Extraer IDs del top 5000 para filtrar archivos grandes
ids_sup   = sorted(df_sup["producto"].dropna().astype(int).tolist())   if DATA_OK else []
ids_unsup = sorted(df_unsup["producto"].dropna().astype(int).tolist()) if DATA_OK else []
_all_ids  = set(ids_sup) | set(ids_unsup) if DATA_OK else None

# Cargar consumo filtrado por IDs conocidos
print("[Dash] Cargando consumo (filtrado)...", flush=True)
df_consumo_sup = df_consumo_unsup = pd.DataFrame()
try:
    df_consumo = load_consumo(FILE_CONSUMO, valid_ids=_all_ids)
    if DATA_OK and not df_consumo.empty and "CLIENTE_ID" in df_consumo.columns:
        _s_ids = set(ids_sup);   _u_ids = set(ids_unsup)
        df_consumo_sup   = df_consumo[df_consumo["CLIENTE_ID"].isin(_s_ids)]
        df_consumo_unsup = df_consumo[df_consumo["CLIENTE_ID"].isin(_u_ids)]
    else:
        df_consumo_sup = df_consumo_unsup = df_consumo
    del df_consumo; _gc.collect()
except Exception as e:
    _load_errors.append(f"consumo: {e}")
LOAD_ERROR = "\n".join(_load_errors)

# Cargar ROC (pequeño, 18KB)
print("[Dash] Cargando ROC...", flush=True)
roc_data, roc_error = load_roc(FILE_ROC)

# Cargar geo filtrado por IDs
print("[Dash] Cargando georeferencias (filtrado)...", flush=True)
df_geo_raw, geo_error = load_geo(FILE_GEO, valid_ids=_all_ids)
if DATA_OK and not df_geo_raw.empty:
    _s_ids = set(ids_sup);   _u_ids = set(ids_unsup)
    df_geo_sup   = df_geo_raw[df_geo_raw["producto"].isin(_s_ids)]
    df_geo_unsup = df_geo_raw[df_geo_raw["producto"].isin(_u_ids)]
else:
    df_geo_sup = df_geo_unsup = df_geo_raw
del df_geo_raw; _gc.collect()

# ── Mahalanobis ───────────────────────────────────────────
def _compute_mahalanobis(df_r1, df_last):
    """Calcula distancias de Mahalanobis por grupo. Retorna dict {cliente_id: info}."""
    if df_r1.empty or df_last.empty:
        return {}
    GC    = ["localidad","barrio","tipo_producto","categoria","subcategoria"]
    av_gc = [c for c in GC if c in df_r1.columns]
    has_ym= "year_month" in df_r1.columns and "year_month" in df_last.columns
    kc    = av_gc + (["year_month"] if has_ym else [])
    need  = [c for c in kc+["CLIENTE_ID","consumo_prev","consumo_actual"] if c in df_r1.columns]
    r1    = df_r1[need].copy()
    for c in kc:
        r1[c]      = r1[c].astype(str)
        df_last[c] = df_last[c].astype(str)
    r1["_gk"] = r1[kc].agg("||".join, axis=1)
    clean     = r1.dropna(subset=["consumo_prev","consumo_actual"])
    grouped   = {k:(g[["consumo_prev","consumo_actual"]].values, g["CLIENTE_ID"].values)
                 for k,g in clean.groupby("_gk", sort=False)}
    fc = "fraud_score" if "fraud_score" in df_last.columns else None
    seen, cache = {}, {}
    for _, row in df_last.iterrows():
        cid = int(row["CLIENTE_ID"])
        if cid in cache:
            continue
        key = "||".join(str(row[c]) for c in kc)
        if key in seen:
            ge = seen[key]
        else:
            pair = grouped.get(key)
            if pair is None:
                continue
            X, ids = pair
            if X.shape[0] < 2:
                continue
            mu  = X.mean(axis=0)
            cov = np.cov(X, rowvar=False, ddof=0)
            if np.linalg.det(cov) == 0:
                cov += np.eye(2)*1e-6
            inv = np.linalg.inv(cov)
            md  = np.sqrt(((X-mu)@inv*(X-mu)).sum(axis=1))
            ge  = {"X":X,"ids_group":ids,"mu":mu,"cov":cov,"mah_dist":md,
                   "es_anomalo":md>2,"filter_desc":{c:row[c] for c in kc}}
            seen[key] = ge
        cache[cid] = {**ge, "fraud_score": float(row[fc]) if fc else None}
    return cache

def _load_mahalanobis():
    """Intenta cargar cache, si no existe computa desde archivos de datos."""
    import traceback as _tb
    # 1. Intentar cache pre-computado
    print(f"[Dash] Mahalanobis: cache path = {FILE_MAH_CACHE}", flush=True)
    print(f"[Dash] Mahalanobis: cache existe = {FILE_MAH_CACHE.exists()}", flush=True)
    if FILE_MAH_CACHE.exists():
        try:
            with open(FILE_MAH_CACHE, "rb") as f:
                cache = pickle.load(f)
            print(f"[Dash] ✓ Cache Mahalanobis cargado: {len(cache)} clientes.", flush=True)
            return cache
        except Exception as e:
            print(f"[Dash] ⚠ Error cargando cache: {e}", flush=True)
            _tb.print_exc()
    # 2. Computar desde datos si existen
    print(f"[Dash] Mahalanobis: resultados_1 path = {FILE_RESULTADOS_1}", flush=True)
    print(f"[Dash] Mahalanobis: resultados_1 existe = {FILE_RESULTADOS_1.exists()}", flush=True)
    print(f"[Dash] Mahalanobis: resultados_df_last path = {FILE_RESDF_LAST}", flush=True)
    print(f"[Dash] Mahalanobis: resultados_df_last existe = {FILE_RESDF_LAST.exists()}", flush=True)
    if FILE_RESULTADOS_1.exists() and FILE_RESDF_LAST.exists():
        try:
            print("[Dash] Computando Mahalanobis desde datos...", flush=True)
            r1 = pd.read_parquet(FILE_RESULTADOS_1)
            print(f"[Dash] Mahalanobis: r1 shape={r1.shape}, cols={list(r1.columns)[:6]}", flush=True)
            if _all_ids is not None and "CLIENTE_ID" in r1.columns:
                r1 = r1[r1["CLIENTE_ID"].isin(_all_ids)]
                print(f"[Dash] Mahalanobis: r1 filtrado={len(r1)} filas ({len(_all_ids)} IDs)", flush=True)
            last = pd.read_csv(FILE_RESDF_LAST)
            print(f"[Dash] Mahalanobis: last shape={last.shape}, cols={list(last.columns)[:6]}", flush=True)
            if _all_ids is not None and "CLIENTE_ID" in last.columns:
                last = last[last["CLIENTE_ID"].isin(_all_ids)]
                print(f"[Dash] Mahalanobis: last filtrado={len(last)} filas", flush=True)
            for d in [r1, last]:
                if "year_month" in d.columns:
                    d["year_month"] = pd.to_datetime(d["year_month"], errors="coerce").dt.strftime("%Y-%m-%d")
            cache = _compute_mahalanobis(r1, last)
            del r1, last
            print(f"[Dash] ✓ Mahalanobis computado: {len(cache)} clientes.", flush=True)
            return cache
        except Exception as e:
            print(f"[Dash] ⚠ Error computando Mahalanobis: {e}", flush=True)
            _tb.print_exc()
    else:
        print("[Dash] ℹ Sin archivos para Mahalanobis.", flush=True)
    return {}

_MAH_CACHE = _load_mahalanobis()
_mah_ids_ordered = sorted(_MAH_CACHE.keys(),
                          key=lambda c: -(_MAH_CACHE[c].get("fraud_score") or 0)) if _MAH_CACHE else []
_gc.collect()
print("[Dash] ✓ Carga completa.", flush=True)

# =========================================================
# 3b) AUTENTICACIÓN
# =========================================================
_AUTH_USERS = {}  # key (username/email lowercase) → {username, email, password}

def _init_auth():
    i = 1
    users = []
    while True:
        name = os.environ.get(f"AUTH_USER{i}_NAME")
        if not name:
            break
        pwd = os.environ.get(f"AUTH_USER{i}_PASS", "")
        email = os.environ.get(f"AUTH_USER{i}_EMAIL", "")
        users.append({"username": name, "email": email, "password": pwd})
        i += 1
    for u in users:
        _AUTH_USERS[u["username"].strip().lower()] = u
        if u["email"]:
            _AUTH_USERS[u["email"].strip().lower()] = u
    if users:
        print(f"[Dash] ✓ Auth: {len(users)} usuarios configurados.", flush=True)
    else:
        print("[Dash] ℹ Auth deshabilitado (sin AUTH_USER*_NAME).", flush=True)

_init_auth()
_AUTH_ENABLED = bool(_AUTH_USERS)
_FAIL_LOG = _defaultdict(list)  # ip → [timestamps]
_MAX_FAILS = 5
_LOCKOUT_SEC = 300

# =========================================================
# 4) PALETA SW DISICO — Light Theme (Design System Biblia v3.0)
# =========================================================
_FONT = "ui-sans-serif, system-ui, -apple-system, sans-serif"
_MONO = "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace"

# Primarios
C_NAVY      = "#1B5E7E"
C_NAVY_DEEP = "#0D3548"
C_NAVY_LT   = "#2A7A9F"
C_STEEL     = "#4A8BA8"
C_STEEL_LT  = "#7FB3C8"
C_GOLD      = "#E8B649"
C_GOLD_LT   = "#F5D080"
C_ORANGE    = "#FF8C1A"

# Superficies Light
C_BG        = "#F4F3F0"
C_BG1       = "#FAFAFA"
C_BG2       = "#FFFFFF"
C_BG3       = "#EDF2F4"
C_BG4       = "#D8E2E8"
C_BORDER    = "rgba(13,53,72,0.10)"
C_BORDER_HI = "rgba(13,53,72,0.22)"

# Texto
C_TEXT      = "#0D1E28"
C_TEXT2     = "#3D6478"
C_TEXT3     = "#7A9BAD"

# Semánticos
C_RED       = "#EF476F"
C_GREEN     = "#06D6A0"
C_BLUE      = C_STEEL

# Colores de método
C_SUP_ACCENT   = C_STEEL      # acento supervisado
C_UNSUP_ACCENT = C_GOLD       # acento no supervisado

# Paleta de gráficos
BAR_COLORS = [
    "#4A8BA8",  # Steel
    "#FF8C1A",  # Orange
    "#E8B649",  # Gold
    "#7FB3C8",  # Steel light
    "#2A7A9F",  # Navy light
    "#F5D080",  # Gold light
    "#06D6A0",  # Green
    "#EF476F",  # Red/Pink
    "#4361EE",  # Blue
    "#FFD166",  # Yellow
]

# ── Tipografía ─────────────────────────────────────────────
T_H1   = {"fontSize": "22px", "fontWeight": "700", "color": C_TEXT,
          "fontFamily": _FONT, "letterSpacing": "-0.01em"}
T_H2   = {"fontSize": "14px", "fontWeight": "700", "color": "#FFFFFF",
          "fontFamily": _FONT, "letterSpacing": "0.02em"}
T_H3   = {"fontSize": "14px", "fontWeight": "600", "color": C_TEXT,
          "fontFamily": _FONT}
T_BODY = {"fontSize": "13px", "fontWeight": "400", "color": C_TEXT2,
          "fontFamily": _FONT, "lineHeight": "1.5"}
T_CAP  = {"fontSize": "11px", "fontWeight": "400", "color": C_TEXT3,
          "fontFamily": _FONT}

LABEL = {"color": C_TEXT3, "fontSize": "11px", "fontWeight": "600",
         "textTransform": "uppercase", "letterSpacing": "0.06em",
         "marginBottom": "6px", "display": "block", "fontFamily": _FONT}
DROP  = {"fontSize": "13px", "fontFamily": _FONT}

# ── Estilos de plot ────────────────────────────────────────
PLOT = dict(
    paper_bgcolor=C_BG2, plot_bgcolor=C_BG1,
    font={"color": C_TEXT2, "family": _FONT, "size": 12},
    margin={"t": 36, "b": 48, "l": 56, "r": 24},
    colorway=BAR_COLORS,
    xaxis={"gridcolor": C_BORDER, "zerolinecolor": C_BORDER},
    yaxis={"gridcolor": C_BORDER, "zerolinecolor": C_BORDER},
)

TOP_N_OPTIONS = [{"label": "25", "value": 25}, {"label": "50", "value": 50},
                 {"label": "100", "value": 100}, {"label": "500", "value": 500},
                 {"label": "1000", "value": 1000}, {"label": "Todos", "value": 999999}]

# Tabs — styled via CSS (.custom-tab), minimal inline needed
TAB_STYLE = {"padding": "12px 22px", "border": "none",
             "borderBottom": "2px solid transparent",
             "background": "transparent", "color": C_TEXT3,
             "fontSize": "13px", "fontWeight": "600", "fontFamily": _FONT}
TAB_SEL   = {"padding": "12px 22px", "border": "none",
             "borderBottom": f"2px solid {C_GOLD}",
             "background": "rgba(232,182,73,0.06)", "color": C_GOLD,
             "fontSize": "13px", "fontWeight": "700", "fontFamily": _FONT}


def title_bar(text, accent_color=C_NAVY):
    """Barra de título premium con gradiente sutil."""
    return html.Div(text, className="title-bar", style={
        **T_H2,
        "background": f"linear-gradient(135deg, {accent_color} 0%, {C_NAVY_DEEP} 100%)",
    })


def card_wrap(children, **extra):
    """Card body debajo de title_bar."""
    return html.Div(className="card-body", children=children,
                    style={**extra})


def plot_legend(fig):
    """Aplica estilo de leyenda consistente."""
    fig.update_layout(legend={
        "bgcolor": "rgba(255,255,255,0.95)",
        "bordercolor": C_BORDER, "borderwidth": 1,
        "font": {"size": 11, "color": C_TEXT2, "family": _FONT},
    })
    return fig

# =========================================================
# 5) LAYOUTS
# =========================================================

def _data_table(tid, columns, data=None, **kw):
    """DataTable con estilo consistente (CSS maneja colores)."""
    return dash_table.DataTable(
        id=tid, columns=columns, data=data or [],
        page_size=kw.get("page_size", 12),
        sort_action="native", filter_action="native",
        row_selectable=kw.get("row_selectable"),
        selected_rows=kw.get("selected_rows", []),
        style_table={"overflowX": "auto"},
        style_cell={"textAlign": "center", "fontFamily": _FONT,
                     "fontSize": "13px", "padding": "10px 14px",
                     "border": "none"},
        style_header={"fontWeight": "600", "textAlign": "center",
                       "border": "none"},
        style_as_list_view=True,
    )


# ── Pestaña 1: Scores & Series ────────────────────────────
tab1_layout = html.Div([

    html.Div(id="error_box_t1", className="error-box",
             style={"display": "none" if DATA_OK else "block"},
             children="" if DATA_OK else f"Error cargando datos:\n{LOAD_ERROR}"),

    # Controles compartidos
    html.Div(className="controls-row", children=[
        html.Div([
            html.Label("Buscar producto:", style=LABEL),
            dcc.Input(id="search_producto", type="text", placeholder="Ej: 41",
                      style={"width": "160px"}),
        ]),
        html.Div([
            html.Label("Top N:", style=LABEL),
            dcc.Dropdown(id="top_n", options=TOP_N_OPTIONS, value=100,
                         clearable=False, style={**DROP, "width": "120px"}),
        ]),
    ]),

    # Dos tablas de scores lado a lado
    html.Div(className="two-col", children=[
        html.Div([
            title_bar("Modelo Supervisado", C_SUP_ACCENT),
            card_wrap([
                _data_table("tabla_sup",
                    columns=[{"name": "Producto", "id": "producto"},
                             {"name": "Prediccion Supervisada", "id": "score_supervisado",
                              "type": "numeric", "format": {"specifier": ".4f"}}],
                    data=df_sup.to_dict("records") if DATA_OK else [],
                    row_selectable="multi", selected_rows=[]),
            ]),
        ]),
        html.Div([
            title_bar("Modelo No Supervisado", C_UNSUP_ACCENT),
            card_wrap([
                _data_table("tabla_unsup",
                    columns=[{"name": "Producto", "id": "producto"},
                             {"name": "Score No Supervisado", "id": "score_no_supervisado",
                              "type": "numeric", "format": {"specifier": ".4f"}}],
                    data=df_unsup.to_dict("records") if DATA_OK else [],
                    row_selectable="multi", selected_rows=[]),
            ]),
        ]),
    ]),

    # Estadísticas lado a lado
    html.Div(className="two-col", children=[
        html.Div([
            title_bar("Estadisticas – Supervisado", C_SUP_ACCENT),
            card_wrap([
                _data_table("tabla_stats_sup",
                    columns=[{"name": "Estadistico", "id": "Estadístico"},
                             {"name": "Prediccion Supervisada", "id": "Predicción Supervisada",
                              "type": "numeric", "format": {"specifier": ".4f"}}]),
            ]),
        ]),
        html.Div([
            title_bar("Estadisticas – No Supervisado", C_UNSUP_ACCENT),
            card_wrap([
                _data_table("tabla_stats_unsup",
                    columns=[{"name": "Estadistico", "id": "Estadístico"},
                             {"name": "Score No Supervisado", "id": "Score No Supervisado",
                              "type": "numeric", "format": {"specifier": ".4f"}}]),
            ]),
        ]),
    ]),

    # Series de tiempo lado a lado
    html.Div(className="two-col", children=[
        html.Div([
            title_bar("Series de Tiempo – Supervisado", C_SUP_ACCENT),
            card_wrap([dcc.Graph(id="grafica_consumo_sup")]),
        ]),
        html.Div([
            title_bar("Series de Tiempo – No Supervisado", C_UNSUP_ACCENT),
            card_wrap([dcc.Graph(id="grafica_consumo_unsup")]),
        ]),
    ]),
])


# ── Helper: layout descriptivas ───────────────────────────
def _desc_layout(sfx, accent, client_list):
    return html.Div([
        # Controles
        html.Div(className="controls-row", children=[
            html.Div([
                html.Label("Filtrar por Cliente ID:", style=LABEL),
                dcc.Dropdown(id=f"filter_cliente_{sfx}",
                             options=[{"label": str(c), "value": c} for c in client_list],
                             value=None, placeholder="Todos", clearable=True,
                             style={**DROP, "width": "200px"}),
            ]),
            html.Div([
                html.Label("Top N clientes:", style=LABEL),
                dcc.Dropdown(id=f"top_n_desc_{sfx}", options=TOP_N_OPTIONS, value=10,
                             clearable=False, style={**DROP, "width": "150px"}),
            ]),
        ]),
        # Categoría (fila completa)
        html.Div(style={"marginBottom": "16px"}, children=[
            title_bar("Categoria", accent),
            card_wrap([dcc.Graph(id=f"graf_categoria_{sfx}", style={"height": "300px"})]),
        ]),
        # Grid 2×2
        html.Div(className="grid-2x2", children=[
            html.Div([title_bar("Subcategoria", accent),
                       card_wrap([dcc.Graph(id=f"graf_subcategoria_{sfx}", style={"height": "260px"})])]),
            html.Div([title_bar("Localidad", accent),
                       card_wrap([dcc.Graph(id=f"graf_localidad_{sfx}", style={"height": "260px"})])]),
            html.Div([title_bar("Barrio", accent),
                       card_wrap([dcc.Graph(id=f"graf_barrio_{sfx}", style={"height": "260px"})])]),
            html.Div([title_bar("Tipo de Producto", accent),
                       card_wrap([dcc.Graph(id=f"graf_tipo_{sfx}", style={"height": "260px"})])]),
        ]),
        # Mapa + KPIs
        html.Div(className="map-kpi-row", children=[
            html.Div(className="map-col disico-card", style={"padding": "20px"}, children=[
                html.P("Mapa de Geolocalizacion", style={**T_H3, "margin": "0 0 10px 0"}),
                dcc.Graph(id=f"graf_mapa_{sfx}", style={"height": "400px"}),
            ]),
            html.Div(className="kpi-col", children=[
                html.P("Resumen", style={**T_H3, "margin": "0 0 6px 0"}),
                html.Div(id=f"kpi_total_{sfx}", className="kpi-card", children=[
                    html.Span("Total clientes", className="kpi-label"),
                    html.Span("--", className="kpi-value"),
                ]),
                html.Div(id=f"kpi_sindir_{sfx}", className="kpi-card danger", children=[
                    html.Span("Sin direccion", className="kpi-label"),
                    html.Span("--", className="kpi-value"),
                ]),
            ]),
        ]),
    ])

tab2_layout = _desc_layout("unsup", C_UNSUP_ACCENT, ids_unsup)
tab3_layout = _desc_layout("sup",   C_SUP_ACCENT,   ids_sup)

# ── Pestaña 4: Métricas de los Modelos ────────────────────
METRICS = {"Exactitud": "67.4 %", "AUC": "0.712", "Recall (anomalias)": "41.3 %",
           "Precision (anomalias)": "64.5 %", "Perfil": "Mejor AUC"}

tab4_layout = html.Div([
    html.Div(className="grid-3", children=[
        html.Div(className="disico-card", style={"padding": "20px"}, children=[
            html.P("Curva ROC – LightGBM", style={**T_H3, "margin": "0 0 10px 0"}),
            dcc.Graph(id="graf_roc", style={"height": "360px"}),
            html.Div(id="roc_error_box", className="error-box",
                     style={"display": "block" if roc_error else "none", "marginTop": "8px"},
                     children=f"Error ROC: {roc_error}" if roc_error else ""),
        ]),
        html.Div(className="disico-card", style={"padding": "20px"}, children=[
            html.P("Metricas – LightGBM", style={**T_H3, "margin": "0 0 16px 0"}),
            html.Div(style={"display": "flex", "flexDirection": "column", "gap": "8px"},
                     children=[
                         html.Div(className="metric-row", children=[
                             html.Span(k, style={**T_BODY}),
                             html.Span(v, style={**T_H3, "color": C_GOLD})])
                         for k, v in METRICS.items()
                     ]),
            html.Div(className="insight-box", children=
                html.P("LightGBM obtuvo el mayor AUC (0.712) con buen balance entre "
                       "recall y precision para deteccion de anomalias energeticas.",
                       style={**T_CAP, "margin": "0", "color": C_TEXT2})),
        ]),
        html.Div(className="disico-card", style={"padding": "20px"}, children=[
            html.P("Matriz de Confusion – LightGBM", style={**T_H3, "margin": "0 0 10px 0"}),
            dcc.Graph(id="graf_confmat", style={"height": "360px"}),
        ]),
    ]),
    html.Div(className="disico-card mah-section", style={"padding": "24px", "marginTop": "16px"}, children=[
        html.P("Analisis No Supervisado – Distancia de Mahalanobis",
               style={**T_H3, "margin": "0 0 4px 0"}),
        html.P("Ingresa un Cliente ID para compararlo frente a su grupo. Ordenado por fraud score descendente.",
               style={**T_CAP, "margin": "0 0 16px 0"}),
        html.Div(className="controls-row",
                 style={"marginBottom": "16px", "padding": "12px 16px"},
                 children=[
                     html.Div([
                         html.Label("Buscar Cliente ID:", style=LABEL),
                         dcc.Input(id="dd_cliente_mah", type="number",
                                   placeholder="Ej: 11116848", debounce=True,
                                   value=_mah_ids_ordered[0] if _mah_ids_ordered else None,
                                   style={"width": "200px"}),
                     ]),
                     html.Div(className="info-badge",
                              style={"alignSelf": "flex-end", "marginBottom": "2px"},
                              children=(f"Top: {_mah_ids_ordered[0] if _mah_ids_ordered else '--'}"
                                        f"  |  {len(_mah_ids_ordered)} clientes")),
                 ]),
        html.Div(id="label_cliente_activo", className="mah-label",
                 style={"marginBottom": "12px"}),
        dcc.Graph(id="graf_mahalanobis", style={"height": "520px"}),
    ]),
])

# =========================================================
# 6) APP
# =========================================================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "DISICO – Anomalias Energia"
server = app.server
server.secret_key = os.environ.get("SECRET_KEY", "dev-key-cambiar-en-produccion")

# Favicon personalizado
app.index_string = '''<!DOCTYPE html>
<html>
<head>
{%metas%}
<title>{%title%}</title>
<link rel="icon" type="image/png" href="/assets/icon-disico.png">
{%css%}
</head>
<body>
{%app_entry%}
<footer>
{%config%}
{%scripts%}
{%renderer%}
</footer>
</body>
</html>'''


# ── Login layout ──────────────────────────────────────────
def _login_layout():
    return html.Div(className="login-bg", children=[
        html.Div(className="login-card", children=[
            # Logos
            html.Div(style={"display": "flex", "alignItems": "center",
                            "justifyContent": "center", "gap": "16px",
                            "marginBottom": "28px"}, children=[
                html.Img(src="/assets/logo_disico.png",
                         style={"height": "40px", "width": "auto"}),
                html.Div(style={"width": "1px", "height": "32px",
                                "background": C_BORDER_HI, "flexShrink": "0"}),
                html.Img(src="/assets/SW-Disico.png",
                         style={"height": "34px", "width": "auto"}),
            ]),
            # Título
            html.H2("Iniciar Sesión", style={
                "textAlign": "center", "color": C_TEXT, "fontSize": "20px",
                "fontWeight": "700", "fontFamily": _FONT, "margin": "0 0 6px 0"}),
            html.P("Detección de Anomalías de Consumo Energético", style={
                "textAlign": "center", "color": C_TEXT3, "fontSize": "12px",
                "fontFamily": _FONT, "margin": "0 0 24px 0"}),
            # Formulario
            html.Div(children=[
                html.Label("Usuario o correo electrónico", style={
                    "color": C_TEXT2, "fontSize": "12px", "fontWeight": "600",
                    "fontFamily": _FONT, "display": "block", "marginBottom": "6px"}),
                html.Input(id="login-user", type="text",
                           placeholder="usuario o correo",
                           className="login-field",
                           style={"marginBottom": "16px"}),
                html.Label("Contraseña", style={
                    "color": C_TEXT2, "fontSize": "12px", "fontWeight": "600",
                    "fontFamily": _FONT, "display": "block", "marginBottom": "6px"}),
                html.Div(style={"position": "relative", "marginBottom": "20px"}, children=[
                    html.Input(id="login-pass", type="password",
                               placeholder="••••••••",
                               className="login-field login-field--pass"),
                    html.Button(id="toggle-pass", n_clicks=0,
                                className="login-eye-btn",
                                title="Mostrar/ocultar contraseña",
                                children="👁"),
                ]),
                html.Button("Ingresar", id="login-btn", n_clicks=0,
                            className="login-btn"),
                html.Div(id="login-error", style={
                    "color": C_RED, "fontSize": "12px", "fontFamily": _FONT,
                    "textAlign": "center", "marginTop": "14px", "minHeight": "18px"}),
            ]),
        ]),
        dcc.Location(id="login-redirect", refresh=True),
    ])


# ── Dashboard layout ─────────────────────────────────────
def _dashboard_layout():
    logout_link = html.A("Cerrar sesión", href="/logout", style={
        "color": C_TEXT3, "fontSize": "12px", "fontFamily": _FONT,
        "textDecoration": "none", "padding": "4px 12px",
        "border": f"1px solid {C_BORDER}", "borderRadius": "6px",
    }) if _AUTH_ENABLED else None

    return html.Div(
        style={"background": C_BG, "minHeight": "100vh", "fontFamily": _FONT},
        children=[
            # ── Topbar ──
            html.Div(className="topbar", style={
                "background": "rgba(255,255,255,0.92)", "padding": "0 28px",
                "display": "flex", "alignItems": "center", "justifyContent": "space-between",
                "height": "56px",
            }, children=[
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "14px"}, children=[
                    html.Img(src="/assets/logo_disico.png",
                             style={"height": "36px", "width": "auto"}),
                    html.Div(style={"width": "1px", "height": "28px",
                                    "background": C_BORDER_HI, "flexShrink": "0"}),
                    html.Img(src="/assets/SW-Disico.png",
                             style={"height": "30px", "width": "auto"}),
                ]),
                html.Div(style={"display": "flex", "alignItems": "center", "gap": "16px"}, children=[
                    html.Span("Deteccion de Anomalias de Consumo Energetico", style={
                        "color": C_TEXT3, "fontSize": "13px", "fontFamily": _FONT}),
                    logout_link,
                ]) if logout_link else html.Span(
                    "Deteccion de Anomalias de Consumo Energetico", style={
                        "color": C_TEXT3, "fontSize": "13px", "fontFamily": _FONT}),
            ]),
            # ── Contenido principal ──
            html.Div(style={"padding": "24px 28px"}, children=[
                dcc.Tabs(id="main_tabs", value="tab1",
                         style={"marginBottom": "20px", "borderBottom": f"1px solid {C_BORDER}"},
                         children=[
                             dcc.Tab(label="Scores y Series",              value="tab1", style=TAB_STYLE, selected_style=TAB_SEL),
                             dcc.Tab(label="Descriptivas No Supervisado",  value="tab2", style=TAB_STYLE, selected_style=TAB_SEL),
                             dcc.Tab(label="Descriptivas Supervisado",     value="tab3", style=TAB_STYLE, selected_style=TAB_SEL),
                             dcc.Tab(label="Metricas de los Modelos",      value="tab4", style=TAB_STYLE, selected_style=TAB_SEL),
                         ]),
                html.Div(id="tab_content"),
            ]),
        ],
    )


# ── Serve layout (login o dashboard según sesión) ────────
def _serve_layout():
    if _AUTH_ENABLED and "user" not in _flask_session:
        return _login_layout()
    return _dashboard_layout()

app.layout = _serve_layout


# ── Flask middleware: proteger rutas y manejar logout ─────
@server.before_request
def _auth_guard():
    if not _AUTH_ENABLED:
        return None
    path = _flask_request.path
    # Permitir assets estáticos y favicon
    if path.startswith("/assets/") or path == "/favicon.ico":
        return None
    # Logout
    if path == "/logout":
        _flask_session.pop("user", None)
        return _flask_redirect("/")
    # Página principal (serve_layout decide qué mostrar)
    if path == "/":
        return None
    # JS/CSS de Dash y rutas necesarias para renderizar el login
    _DASH_PUBLIC = (
        "/_dash-component-suites/",
        "/_dash-layout",
        "/_dash-dependencies",
        "/_reload-hash",
        "/_dash-error",
    )
    if any(path.startswith(p) if p.endswith("/") else path == p
           for p in _DASH_PUBLIC):
        return None
    # Bloquear callbacks del dashboard si no autenticado
    if path == "/_dash-update-component":
        if "user" in _flask_session:
            return None
        try:
            import json
            body = _flask_request.get_data(as_text=True)
            data = json.loads(body)
            outputs = str(data.get("output", ""))
            if any(k in outputs for k in ("login-error", "login-redirect", "login-pass")):
                return None
        except Exception:
            pass
        from flask import abort
        abort(401)
    # Otros endpoints internos de Dash → redirigir a login si no autenticado
    if path.startswith("/_dash"):
        if "user" not in _flask_session:
            return _flask_redirect("/")
        return None
    return None

# =========================================================
# 7) CALLBACKS
# =========================================================

# ── Login callback ────────────────────────────────────────
@app.callback(
    Output("login-error", "children"),
    Output("login-redirect", "href"),
    Input("login-btn", "n_clicks"),
    State("login-user", "value"),
    State("login-pass", "value"),
    prevent_initial_call=True,
)
def _do_login(n_clicks, user_input, password):
    if not n_clicks or not user_input or not password:
        raise PreventUpdate
    ip = _flask_request.remote_addr or "unknown"
    now = _time.time()
    # Limpiar intentos viejos
    _FAIL_LOG[ip] = [t for t in _FAIL_LOG[ip] if now - t < _LOCKOUT_SEC]
    if len(_FAIL_LOG[ip]) >= _MAX_FAILS:
        wait = int(_LOCKOUT_SEC - (now - _FAIL_LOG[ip][0]))
        return f"Demasiados intentos. Intente en {wait}s.", no_update
    key = user_input.strip().lower()
    record = _AUTH_USERS.get(key)
    if record and hmac.compare_digest(record["password"], password):
        _flask_session["user"] = record["username"]
        _FAIL_LOG.pop(ip, None)
        return "", "/"
    _FAIL_LOG[ip].append(now)
    remaining = _MAX_FAILS - len(_FAIL_LOG[ip])
    if remaining <= 2:
        return f"Credenciales inválidas. {remaining} intentos restantes.", no_update
    return "Usuario o contraseña incorrectos.", no_update


@app.callback(
    Output("login-pass", "type"),
    Input("toggle-pass", "n_clicks"),
    prevent_initial_call=True,
)
def _toggle_password(n):
    return "text" if n % 2 == 1 else "password"


@app.callback(Output("tab_content","children"), Input("main_tabs","value"))
def render_tab(tab):
    return {"tab1":tab1_layout,"tab2":tab2_layout,
            "tab3":tab3_layout,"tab4":tab4_layout}.get(tab, html.Div())


# ── Tab 1: tablas + estadísticas ─────────────────────────
@app.callback(
    Output("tabla_sup",         "data"),
    Output("tabla_unsup",       "data"),
    Output("tabla_stats_sup",   "data"),
    Output("tabla_stats_unsup", "data"),
    Input("search_producto", "value"),
    Input("top_n",           "value"),
)
def update_tables(search, top_n):
    if not DATA_OK: return [], [], [], []
    n = int(top_n) if int(top_n) < 999999 else None

    def _filt(df, score_col):
        d = df.copy()
        if search and str(search).strip():
            try:    d = d[d["producto"] == float(search)]
            except: pass
        d = d.sort_values(score_col, ascending=False)
        return d.head(n) if n else d

    ds = _filt(df_sup,   "score_supervisado")
    du = _filt(df_unsup, "score_no_supervisado")

    def _stats(d, col, label):
        s = d[col].dropna()
        return [{"Estadístico":st, label: round(fn(s),4) if not s.empty else None}
                for st,fn in [("Promedio",lambda x:x.mean()),("Desv. Estándar",lambda x:x.std()),
                               ("Mínimo",lambda x:x.min()),("Máximo",lambda x:x.max())]]

    return (ds.to_dict("records"), du.to_dict("records"),
            _stats(ds,"score_supervisado","Predicción Supervisada"),
            _stats(du,"score_no_supervisado","Score No Supervisado"))


# ── Tab 1: series supervisado ─────────────────────────────
@app.callback(
    Output("grafica_consumo_sup","figure"),
    Input("tabla_sup","derived_virtual_data"),
    Input("tabla_sup","selected_rows"),
)
def graph_sup(derived, selected):
    def empty(m): return go.Figure().update_layout(title=m, height=360, **PLOT)
    if not DATA_OK or df_consumo_sup.empty or not derived: return empty("Sin datos.")
    tdf = pd.DataFrame(derived)
    ids = (tdf.iloc[selected]["producto"].dropna().unique().tolist()
           if selected else tdf["producto"].dropna().unique().tolist()[:25])
    if not ids: return empty("Sin IDs.")
    dff = df_consumo_sup[df_consumo_sup["CLIENTE_ID"].isin(ids)].sort_values(["CLIENTE_ID","fecha"])
    if dff.empty: return empty("Sin registros de consumo.")
    fig = px.line(dff, x="fecha", y="consumo", color=dff["CLIENTE_ID"].astype(str),
                  title=f"Consumo supervisado – {len(ids)} cliente(s)",
                  color_discrete_sequence=BAR_COLORS, template=None)
    fig.update_layout(height=380, legend_title_text="CLIENTE_ID",
                      title={"font": {"size": 13, "color": C_TEXT2, "family": _FONT}},
                      **PLOT)
    plot_legend(fig)
    return fig


# ── Tab 1: series no supervisado ─────────────────────────
@app.callback(
    Output("grafica_consumo_unsup","figure"),
    Input("tabla_unsup","derived_virtual_data"),
    Input("tabla_unsup","selected_rows"),
)
def graph_unsup(derived, selected):
    def empty(m): return go.Figure().update_layout(title=m, height=360, **PLOT)
    if not DATA_OK or df_consumo_unsup.empty or not derived: return empty("Sin datos.")
    tdf = pd.DataFrame(derived)
    ids = (tdf.iloc[selected]["producto"].dropna().unique().tolist()
           if selected else tdf["producto"].dropna().unique().tolist()[:25])
    if not ids: return empty("Sin IDs.")
    dff = df_consumo_unsup[df_consumo_unsup["CLIENTE_ID"].isin(ids)].sort_values(["CLIENTE_ID","fecha"])
    if dff.empty: return empty("Sin registros de consumo.")
    fig = px.line(dff, x="fecha", y="consumo", color=dff["CLIENTE_ID"].astype(str),
                  title=f"Consumo no supervisado – {len(ids)} cliente(s)",
                  color_discrete_sequence=BAR_COLORS, template=None)
    fig.update_layout(height=380, legend_title_text="CLIENTE_ID",
                      title={"font": {"size": 13, "color": C_TEXT2, "family": _FONT}},
                      **PLOT)
    plot_legend(fig)
    return fig


# ── Helpers descriptivas ──────────────────────────────────
def _cat_fig(col, df_cons, df_scores, score_col, fc, tn):
    _pe = {**PLOT, "margin": {"t": 30, "b": 20, "l": 30, "r": 10}}
    _pb = {**PLOT, "margin": {"t": 20, "b": 55, "l": 50, "r": 10}}
    def empty(m): return go.Figure().update_layout(title=m, **_pe)
    if df_cons.empty or col not in df_cons.columns: return empty(f"'{col}' no disponible.")
    n = int(tn) if int(tn) < 999999 else len(df_scores)
    top_ids = df_scores.sort_values(score_col, ascending=False).head(n)["producto"].tolist()
    dff = df_cons[df_cons["CLIENTE_ID"].isin(top_ids)]
    if fc is not None: dff = dff[dff["CLIENTE_ID"] == fc]
    if dff.empty: return empty("Sin datos.")
    dff = dff[[col, "CLIENTE_ID"]].copy()
    dff[col] = dff[col].astype(str).str.strip()
    dff = dff[~dff[col].isin(["nan", "<NA>", "None", ""])]
    if dff.empty: return empty("Sin valores validos.")
    counts = (dff.groupby(col, observed=True)["CLIENTE_ID"].nunique()
                 .reset_index().rename(columns={"CLIENTE_ID": "clientes"})
                 .sort_values("clientes", ascending=False))
    fig = px.bar(counts, x=col, y="clientes", color=col,
                 color_discrete_sequence=BAR_COLORS,
                 labels={col: col.capitalize(), "clientes": "Clientes unicos"})
    fig.update_traces(marker_line_width=0, marker_opacity=0.9)
    fig.update_xaxes(type="category", tickangle=-35, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=C_BORDER)
    fig.update_layout(showlegend=False, **_pb)
    return fig


def _map_fig(df_geo, df_scores, score_col, fc, tn):
    _pm = {**PLOT, "margin": {"t": 10, "b": 10, "l": 10, "r": 10}}
    def kpi(label, val):
        return [html.Span(label, className="kpi-label"),
                html.Span(str(val), className="kpi-value")]
    def empty_m(m): return go.Figure().update_layout(title=m, **_pm)
    if df_geo.empty:
        return empty_m("Sin datos de geolocalizacion."), kpi("Total clientes", "--"), kpi("Sin direccion", "--")
    n = int(tn) if int(tn) < 999999 else len(df_scores)
    top_ids = df_scores.sort_values(score_col, ascending=False).head(n)["producto"].tolist()
    dff = df_geo[df_geo["producto"].isin(top_ids)].copy()
    if fc is not None: dff = dff[dff["producto"] == fc]
    total   = len(dff)
    sin_dir = int((dff["lat"].isna() | dff["lon"].isna()).sum())
    dff_p   = dff.dropna(subset=["lat", "lon"]).copy()
    if dff_p.empty:
        return empty_m("Sin coordenadas validas."), kpi("Total clientes", total), kpi("Sin direccion", sin_dir)
    dff_p["producto_str"] = dff_p["producto"].astype(int).astype(str)
    fig = px.scatter_map(dff_p, lat="lat", lon="lon", hover_name="producto_str",
                         color_discrete_sequence=[C_ORANGE], zoom=11, height=400,
                         map_style="carto-positron")
    fig.update_traces(marker={"size": 9, "opacity": 0.9})
    fig.update_layout(**_pm)
    return fig, kpi("Total clientes", total), kpi("Sin direccion", sin_dir)


# ── Tab 2: descriptivas no supervisado ───────────────────
@app.callback(
    Output("graf_categoria_unsup",   "figure"),
    Output("graf_subcategoria_unsup","figure"),
    Output("graf_localidad_unsup",   "figure"),
    Output("graf_barrio_unsup",      "figure"),
    Output("graf_tipo_unsup",        "figure"),
    Input("filter_cliente_unsup","value"),
    Input("top_n_desc_unsup",    "value"),
)
def desc_unsup(fc, tn):
    tn = tn if tn is not None else 10
    return tuple(_cat_fig(c, df_consumo_unsup, df_unsup, "score_no_supervisado", fc, tn)
                 for c in ["categoria","subcategoria","localidad","barrio","tipo_producto"])

@app.callback(
    Output("graf_mapa_unsup",  "figure"),
    Output("kpi_total_unsup",  "children"),
    Output("kpi_sindir_unsup", "children"),
    Input("filter_cliente_unsup","value"),
    Input("top_n_desc_unsup",   "value"),
)
def mapa_unsup(fc, tn):
    tn = tn if tn is not None else 10
    return _map_fig(df_geo_unsup, df_unsup, "score_no_supervisado", fc, tn)


# ── Tab 3: descriptivas supervisado ──────────────────────
@app.callback(
    Output("graf_categoria_sup",   "figure"),
    Output("graf_subcategoria_sup","figure"),
    Output("graf_localidad_sup",   "figure"),
    Output("graf_barrio_sup",      "figure"),
    Output("graf_tipo_sup",        "figure"),
    Input("filter_cliente_sup","value"),
    Input("top_n_desc_sup",   "value"),
)
def desc_sup(fc, tn):
    tn = tn if tn is not None else 10
    return tuple(_cat_fig(c, df_consumo_sup, df_sup, "score_supervisado", fc, tn)
                 for c in ["categoria","subcategoria","localidad","barrio","tipo_producto"])

@app.callback(
    Output("graf_mapa_sup",  "figure"),
    Output("kpi_total_sup",  "children"),
    Output("kpi_sindir_sup", "children"),
    Input("filter_cliente_sup","value"),
    Input("top_n_desc_sup",   "value"),
)
def mapa_sup(fc, tn):
    tn = tn if tn is not None else 10
    return _map_fig(df_geo_sup, df_sup, "score_supervisado", fc, tn)


# ── Tab 4: confmat ────────────────────────────────────────
@app.callback(Output("graf_confmat","figure"), Input("main_tabs","value"))
def confmat(tab):
    if tab != "tab4": return go.Figure()
    fig = go.Figure(go.Heatmap(
        z=[[923, 165], [426, 300]],
        x=["Normal (0)", "Anomalo (1)"], y=["Normal (0)", "Anomalo (1)"],
        text=[["TN = 923", "FP = 165"], ["FN = 426", "TP = 300"]],
        texttemplate="%{text}",
        colorscale=[[0, C_NAVY_DEEP], [0.35, C_NAVY], [0.65, C_GOLD], [1, C_ORANGE]],
        showscale=True, textfont={"size": 15, "color": "white"}, hoverongaps=False))
    fig.update_layout(xaxis_title="Prediccion", yaxis_title="Valor Real",
                      yaxis_autorange="reversed",
                      **{**PLOT, "margin": {"t": 20, "b": 50, "l": 70, "r": 20},
                         "xaxis": {**PLOT["xaxis"], "side": "bottom"}})
    return fig


# ── Tab 4: ROC ────────────────────────────────────────────
@app.callback(Output("graf_roc","figure"), Input("main_tabs","value"))
def roc(tab):
    def empty(m): return go.Figure().update_layout(title=m, **PLOT)
    if tab != "tab4" or roc_data is None:
        return empty("No se pudo cargar la curva ROC." if roc_data is None else "")
    try:
        if isinstance(roc_data,(list,tuple)) and len(roc_data)>=2:
            fpr,tpr = np.array(roc_data[0]),np.array(roc_data[1])
            auc_val = float(np.trapz(tpr,fpr))
        elif isinstance(roc_data,dict):
            fpr = np.array(roc_data.get("fpr",roc_data.get("FPR",[])))
            tpr = np.array(roc_data.get("tpr",roc_data.get("TPR",[])))
            auc_val = float(roc_data.get("auc",roc_data.get("AUC",np.trapz(tpr,fpr))))
        else: return empty("Formato no reconocido.")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"LightGBM (AUC={auc_val:.3f})",
                                 line={"color": C_ORANGE, "width": 2.5},
                                 fill="tozeroy", fillcolor="rgba(255,140,26,0.08)"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Aleatorio",
                                 line={"color": C_TEXT3, "dash": "dash", "width": 1.5}))
        fig.update_layout(
            xaxis_title="Tasa de Falsos Positivos",
            yaxis_title="Tasa de Verdaderos Positivos",
            **{**PLOT,
               "xaxis": {**PLOT["xaxis"], "showgrid": True, "range": [0, 1]},
               "yaxis": {**PLOT["yaxis"], "showgrid": True, "range": [0, 1]}})
        plot_legend(fig)
        fig.update_layout(legend={"x": 0.55, "y": 0.08})
        return fig
    except Exception as ex: return empty(f"Error: {ex}")


# ── Tab 4: Mahalanobis ────────────────────────────────────
@app.callback(
    Output("graf_mahalanobis",     "figure"),
    Output("label_cliente_activo", "children"),
    Input("dd_cliente_mah",        "value"),
    Input("main_tabs",             "value"),
)
def mahalanobis(cliente_id, tab):
    _pm = {**PLOT,"margin":{"t":40,"b":60,"l":70,"r":30}}
    def empty(m): return go.Figure().update_layout(title=m,**_pm)
    if tab != "tab4": return empty(""), ""
    if cliente_id is None: return empty("Ingresa un Cliente ID."), ""
    entry = _MAH_CACHE.get(int(cliente_id))
    if entry is None: return empty(f"Cliente {cliente_id} no encontrado en el pre-cómputo."), ""
    X,ids_g,mu,cov   = entry["X"],entry["ids_group"],entry["mu"],entry["cov"]
    md,es_an,fd,fval = entry["mah_dist"],entry["es_anomalo"],entry["filter_desc"],entry["fraud_score"]
    et = (ids_g == cliente_id)
    fig = go.Figure()
    nm = (~es_an) & (~et)
    if nm.any(): fig.add_trace(go.Scatter(
        x=X[nm, 0], y=X[nm, 1], mode="markers", name="Normal",
        opacity=0.75, marker={"color": C_STEEL, "size": 8,
                              "line": {"color": "white", "width": 0.5}}))
    am = es_an & (~et)
    if am.any(): fig.add_trace(go.Scatter(
        x=X[am, 0], y=X[am, 1], mode="markers", name="Anomalo grupo (>2s)",
        opacity=0.85, marker={"color": C_RED, "size": 9,
                              "line": {"color": "white", "width": 0.5}}))
    if et.any(): fig.add_trace(go.Scatter(
        x=X[et, 0], y=X[et, 1], mode="markers+text",
        name=f"Cliente {cliente_id}",
        text=[str(int(cliente_id))] * int(et.sum()),
        textposition="top right",
        textfont={"size": 10, "color": C_GOLD},
        marker={"color": C_GOLD, "size": 14, "symbol": "star",
                "line": {"color": C_ORANGE, "width": 1.5}}))
    ax0, ax1 = float(X.min()) * 0.95, float(X.max()) * 1.05
    fig.add_trace(go.Scatter(x=[ax0, ax1], y=[ax0, ax1], mode="lines", name="y = x",
                             line={"color": C_TEXT3, "dash": "dash", "width": 1.5}))
    ve,ve2 = np.linalg.eigh(cov)
    o = ve.argsort()[::-1]; ve,ve2 = ve[o],ve2[:,o]
    tr = np.arctan2(*ve2[:,0][::-1])
    ae,be = 2*np.sqrt(ve[0]),2*np.sqrt(ve[1])
    t = np.linspace(0,2*np.pi,200)
    ex = mu[0]+ae*np.cos(t)*np.cos(tr)-be*np.sin(t)*np.sin(tr)
    ey = mu[1]+ae*np.cos(t)*np.sin(tr)+be*np.sin(t)*np.cos(tr)
    fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", name="2s",
                             line={"color": C_GREEN, "width": 2}))
    fig.update_layout(
        title={"text": " | ".join(f"{k}: {v}" for k, v in fd.items()),
               "font": {"size": 11, "color": C_TEXT3}},
        xaxis_title="Consumo periodo anterior", yaxis_title="Consumo periodo actual",
        **{**_pm,
           "xaxis": {**PLOT["xaxis"], "showgrid": True},
           "yaxis": {**PLOT["yaxis"], "showgrid": True}})
    plot_legend(fig)
    fig.update_layout(legend={"x": 0.01, "y": 0.99})
    dc = float(md[et].mean()) if et.any() else float("nan")
    fs = f"{fval:.4f}" if fval is not None else "N/A"
    label = (f"Cliente {cliente_id}  |  Fraud score: {fs}  |  "
             f"Grupo: {X.shape[0]} obs.  |  Anómalos: {es_an.sum()}  |  "
             f"Dist. Mahalanobis: {dc:.2f}σ")
    return fig, label


# =========================================================
# 8) RUN
# =========================================================
if __name__ == "__main__":
    app.run(debug=True)
