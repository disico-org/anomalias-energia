import os
from pathlib import Path
import pickle
import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor as _TPE

from dash import Dash, html, dcc, dash_table, Input, Output
import plotly.express as px
import plotly.graph_objects as go

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


def load_consumo(path):
    wanted    = ["fecha","consumo","CLIENTE_ID","localidad","barrio",
                 "tipo_producto","categoria","subcategoria",
                 "consumo_prev","consumo_actual","mes"]
    available = pd.read_parquet(path, columns=None).columns.tolist()
    cols = [c for c in wanted if c in available]
    df = pd.read_parquet(path, columns=cols).copy()
    df["fecha"]      = pd.to_datetime(df["fecha"],      errors="coerce")
    df["consumo"]    = pd.to_numeric(df["consumo"],     errors="coerce")
    df["CLIENTE_ID"] = pd.to_numeric(df["CLIENTE_ID"],  errors="coerce")
    df = df.dropna(subset=["fecha","consumo","CLIENTE_ID"])
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


def load_geo(path):
    try:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]
        for alt in ["cliente_id","clientes_id"]:
            if alt in df.columns and "producto" not in df.columns:
                df = df.rename(columns={alt:"producto"})
        df["producto"] = pd.to_numeric(df["producto"], errors="coerce")
        df["lat"]      = pd.to_numeric(df["lat"],      errors="coerce")
        df["lon"]      = pd.to_numeric(df["lon"],      errors="coerce")
        return df, None
    except Exception as e:
        return pd.DataFrame(), str(e)


def load_resultados(path_r1, path_last):
    try:
        r1   = pd.read_parquet(path_r1) if str(path_r1).endswith(".parquet") else pd.read_csv(path_r1)
        last = pd.read_csv(path_last)
        for d in [r1, last]:
            if "year_month" in d.columns:
                d["year_month"] = (pd.to_datetime(d["year_month"], errors="coerce")
                                     .dt.strftime("%Y-%m-%d"))
        return r1, last, None
    except Exception as e:
        return pd.DataFrame(), pd.DataFrame(), str(e)


# =========================================================
# 3) CARGA INICIAL
# =========================================================
df_sup = df_unsup = df_consumo = pd.DataFrame()
_load_errors = []

try:
    df_sup = load_supervised(FILE_SUP)
except Exception as e:
    _load_errors.append(f"supervisado: {e}")

try:
    df_unsup = load_unsupervised(FILE_UNSUP)
except Exception as e:
    _load_errors.append(f"no supervisado: {e}")

try:
    df_consumo = load_consumo(FILE_CONSUMO)
except Exception as e:
    _load_errors.append(f"consumo: {e}")

DATA_OK    = not (df_sup.empty and df_unsup.empty)
LOAD_ERROR = "\n".join(_load_errors)

with _TPE(max_workers=3) as _ex:
    _f_roc = _ex.submit(load_roc,        FILE_ROC)
    _f_geo = _ex.submit(load_geo,        FILE_GEO)
    _f_res = _ex.submit(load_resultados, FILE_RESULTADOS_1, FILE_RESDF_LAST)
    roc_data,   roc_error  = _f_roc.result()
    df_geo_raw, geo_error  = _f_geo.result()
    df_res1, df_res_last, res_error = _f_res.result()

# IDs únicos por método (sin merge entre sí)
ids_sup   = sorted(df_sup["producto"].dropna().astype(int).tolist())   if DATA_OK else []
ids_unsup = sorted(df_unsup["producto"].dropna().astype(int).tolist()) if DATA_OK else []

# Pre-filtrar consumo y geo independientemente por método
if DATA_OK:
    _s_ids = set(ids_sup);   _u_ids = set(ids_unsup)
    df_consumo_sup   = df_consumo[df_consumo["CLIENTE_ID"].isin(_s_ids)].copy()
    df_consumo_unsup = df_consumo[df_consumo["CLIENTE_ID"].isin(_u_ids)].copy()
    df_geo_sup   = df_geo_raw[df_geo_raw["producto"].isin(_s_ids)].copy() if not df_geo_raw.empty else pd.DataFrame()
    df_geo_unsup = df_geo_raw[df_geo_raw["producto"].isin(_u_ids)].copy() if not df_geo_raw.empty else pd.DataFrame()
else:
    df_consumo_sup = df_consumo_unsup = df_consumo.copy()
    df_geo_sup     = df_geo_unsup     = df_geo_raw.copy()

df_res1_full = df_res1.copy()

if not df_res_last.empty and "CLIENTE_ID" in df_res_last.columns:
    _sc = "fraud_score" if "fraud_score" in df_res_last.columns else df_res_last.columns[0]
    _mah_ids_ordered = (df_res_last.sort_values(_sc, ascending=False)["CLIENTE_ID"]
                        .drop_duplicates().astype(int).tolist())
else:
    _mah_ids_ordered = ids_unsup

# ── Pre-cómputo Mahalanobis ────────────────────────────────
def _precompute_mahalanobis(df_r1, df_last):
    if df_r1.empty or df_last.empty: return {}
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
        if cid in cache: continue
        key = "||".join(str(row[c]) for c in kc)
        if key in seen:
            ge = seen[key]
        else:
            pair = grouped.get(key)
            if pair is None: continue
            X, ids = pair
            if X.shape[0] < 2: continue
            mu  = X.mean(axis=0)
            cov = np.cov(X, rowvar=False, ddof=0)
            if np.linalg.det(cov) == 0: cov += np.eye(2)*1e-6
            inv = np.linalg.inv(cov)
            md  = np.sqrt(((X-mu)@inv*(X-mu)).sum(axis=1))
            ge  = {"X":X,"ids_group":ids,"mu":mu,"cov":cov,"mah_dist":md,
                   "es_anomalo":md>2,"filter_desc":{c:row[c] for c in kc}}
            seen[key] = ge
        cache[cid] = {**ge, "fraud_score": float(row[fc]) if fc else None}
    return cache

# ── Cargar o calcular cache de Mahalanobis ─────────────────
def _load_or_compute_mahalanobis(df_r1, df_last):
    """
    Intenta cargar cache pre-calculado, si no existe lo calcula.
    """
    # Intentar cargar cache si existe
    if FILE_MAH_CACHE.exists():
        try:
            print(f"Cargando cache de Mahalanobis desde {FILE_MAH_CACHE}…", flush=True)
            with open(FILE_MAH_CACHE, "rb") as f:
                cache = pickle.load(f)
            print(f"✓ Cache cargado: {len(cache)} clientes.", flush=True)
            return cache
        except Exception as e:
            print(f"⚠ Error cargando cache: {e}. Recalculando…", flush=True)
    
    # Calcular si no hay cache o está corrupto
    print("Pre-computando Mahalanobis…", flush=True)
    cache = _precompute_mahalanobis(df_r1, df_last)
    print(f"✓ Cálculo completado: {len(cache)} clientes.", flush=True)
    return cache

_MAH_CACHE = _load_or_compute_mahalanobis(df_res1_full.copy(), df_res_last.copy())

# =========================================================
# 4) PALETA SW DISICO
# =========================================================
# Primarios
C_NAVY      = "#1B5E7E"   # Navy   — fondos, texto primario, autoridad
C_STEEL     = "#4A8BA8"   # Steel  — soporte, iconos, etiquetas
C_GOLD      = "#E8B649"   # Gold   — KPIs, alertas, destacados
C_ORANGE    = "#FF8C1A"   # Orange — acción, CTA, estados activos

# Superficies (fondo claro, inspirado en las pantallas del producto)
C_BG        = "#f5f7f9"   # fondo general — muy claro, casi blanco
C_WHITE     = "#ffffff"
C_BORDER    = "#dde3ea"
C_TEXT      = "#1B2E3C"   # texto principal — navy oscuro
C_SUBTEXT   = "#5a7080"   # texto secundario — steel grisáceo
C_RED       = "#e05252"   # error / alerta negativa

# Alias para uso general
C_BLUE      = C_STEEL
C_LIGHTBLUE = "#7ab8cc"

# Colores de método
# Supervisado  → Steel (azul medio)   header columnas
# No supervisado → Gold oscuro        header columnas
# Ambos comparten Navy para la barra de título del card
C_SUP_COL_HDR   = C_STEEL          # header columnas tabla supervisado
C_UNSUP_COL_HDR = "#b07a1a"        # header columnas tabla no supervisado (gold oscuro)
C_SUP_HDR       = C_NAVY           # título card supervisado (mismo para ambos)
C_UNSUP_HDR     = C_NAVY           # título card no supervisado
C_SUP_ROW       = "#e8f2f8"        # fila alterna supervisado — Navy scale muy pálido
C_UNSUP_ROW     = "#fdf5e0"        # fila alterna no supervisado — Gold scale muy pálido
C_SUP_BG        = "#f2f8fc"        # fondo card supervisado
C_UNSUP_BG      = "#fdfaf0"        # fondo card no supervisado

# Paleta de barras: Navy scale + Orange/Gold scale alternados
BAR_COLORS = [
    "#1B5E7E",  # Navy 950
    "#2a7fa8",  # Navy 700
    "#4A8BA8",  # Steel
    "#7ab8cc",  # Navy 300
    "#b8dae8",  # Navy 100
    "#FF8C1A",  # Orange base
    "#E8B649",  # Gold base
    "#c0760f",  # Orange oscuro
    "#b07a1a",  # Gold oscuro
    "#ffd080",  # Gold claro
]

# ── Tipografía jerárquica ──────────────────────────────────
# H1: 20px 700  — título de sección principal
# H2: 15px 700  — título de card / barra de color
# H3: 13px 600  — subtítulo dentro de card
# Body: 13px 400
# Caption: 11px 400 — notas, subtext
T_H1     = {"fontSize":"20px","fontWeight":"700","color":C_NAVY,
            "fontFamily":"ui-sans-serif, system-ui, sans-serif"}
T_H2     = {"fontSize":"15px","fontWeight":"700","color":"white",
            "fontFamily":"ui-sans-serif, system-ui, sans-serif"}
T_H3     = {"fontSize":"13px","fontWeight":"600","color":C_NAVY,
            "fontFamily":"ui-sans-serif, system-ui, sans-serif"}
T_BODY   = {"fontSize":"13px","fontWeight":"400","color":C_TEXT,
            "fontFamily":"ui-sans-serif, system-ui, sans-serif"}
T_CAP    = {"fontSize":"11px","fontWeight":"400","color":C_SUBTEXT,
            "fontFamily":"ui-sans-serif, system-ui, sans-serif"}

def CARD(bg=C_WHITE, mb="14px"):
    return {"background":bg,"borderRadius":"10px","padding":"20px","marginBottom":mb,
            "boxShadow":"0 2px 12px rgba(27,94,126,0.10)","border":f"1px solid {C_BORDER}"}

def TH(hdr): return {"backgroundColor":hdr,"color":"white","fontWeight":"700",
                     "fontSize":"13px","textAlign":"center","padding":"10px 12px",
                     "fontFamily":"ui-sans-serif, system-ui, sans-serif"}
def TD(bg=C_WHITE): return {"backgroundColor":bg,"color":C_TEXT,"padding":"9px 12px",
                             "fontSize":"13px","border":f"1px solid {C_BORDER}","textAlign":"center",
                             "fontFamily":"ui-sans-serif, system-ui, sans-serif"}

LABEL = {"color":C_SUBTEXT,"fontSize":"12px","fontWeight":"600","marginBottom":"4px",
         "display":"block","fontFamily":"ui-sans-serif, system-ui, sans-serif"}
DROP  = {"fontSize":"13px","fontFamily":"ui-sans-serif, system-ui, sans-serif"}

PLOT = dict(paper_bgcolor=C_WHITE, plot_bgcolor="#f8fafc",
            font={"color":C_TEXT,
                  "family":"ui-sans-serif, system-ui, sans-serif",
                  "size":12},
            margin={"t":36,"b":48,"l":56,"r":24}, colorway=BAR_COLORS)

TOP_N_OPTIONS = [{"label":"25","value":25},{"label":"50","value":50},
                 {"label":"100","value":100},{"label":"500","value":500},
                 {"label":"1000","value":1000},{"label":"Todos","value":999999}]

TAB_STYLE = {"color":C_SUBTEXT,"background":C_WHITE,"border":f"1px solid {C_BORDER}",
             "padding":"10px 20px","fontSize":"13px","fontWeight":"600",
             "fontFamily":"ui-sans-serif, system-ui, sans-serif"}
TAB_SEL   = {"color":C_NAVY,"background":C_BG,
             "borderTop":f"3px solid {C_ORANGE}",
             "padding":"10px 20px","fontSize":"13px","fontWeight":"700",
             "fontFamily":"ui-sans-serif, system-ui, sans-serif"}

def title_bar(text, color):
    """Barra de título H2 para cards."""
    return html.Div(text, style={
        **T_H2,
        "background":color,"padding":"8px 16px","borderRadius":"10px 10px 0 0"})

def section_title(text):
    """Título H3 dentro de un card (sin barra de color)."""
    return html.P(text, style={**T_H3, "margin":"0 0 10px 0"})

def card_wrap(children, bg=C_WHITE):
    return html.Div(style={**CARD(bg),"borderRadius":"0 0 10px 10px",
                            "marginBottom":"0","paddingTop":"14px"},
                    children=children)

def plot_with_legend_bg(fig):
    """Aplica fondo semitransparente a la leyenda de una figura."""
    fig.update_layout(legend={"bgcolor":"rgba(255,255,255,0.85)",
                               "bordercolor":C_BORDER,"borderwidth":1,
                               "font":{"size":12}})
    return fig

# =========================================================
# 5) LAYOUTS
# =========================================================

# ── Pestaña 1: Scores & Series ────────────────────────────
tab1_layout = html.Div([

    html.Div(id="error_box_t1",
             style={"color":C_RED,"whiteSpace":"pre-wrap","marginBottom":"10px","fontSize":"13px"},
             children="" if DATA_OK else f"Error cargando datos:\n{LOAD_ERROR}"),

    # Controles compartidos
    html.Div(style={**CARD(),"display":"flex","gap":"20px","flexWrap":"wrap","alignItems":"flex-end"},
             children=[
                 html.Div([
                     html.Label("Buscar producto:", style=LABEL),
                     dcc.Input(id="search_producto", type="text", placeholder="Ej: 41",
                               style={"width":"150px","padding":"7px 10px","borderRadius":"5px",
                                      "border":f"1px solid {C_BORDER}","fontSize":"13px","color":C_TEXT}),
                 ]),
                 html.Div([
                     html.Label("Top N:", style=LABEL),
                     dcc.Dropdown(id="top_n", options=TOP_N_OPTIONS, value=100,
                                  clearable=False, style={**DROP,"width":"120px"}),
                 ]),
             ]),

    # Dos tablas de scores lado a lado
    html.Div(style={"display":"flex","gap":"14px","marginBottom":"14px","alignItems":"flex-start"},
             children=[
                 html.Div(style={"flex":"1","minWidth":"0"}, children=[
                     title_bar("Modelo Supervisado", C_SUP_HDR),
                     card_wrap(bg=C_SUP_BG, children=[
                         dash_table.DataTable(
                             id="tabla_sup",
                             columns=[{"name":"Producto","id":"producto"},
                                      {"name":"Predicción Supervisada","id":"score_supervisado",
                                       "type":"numeric","format":{"specifier":".4f"}}],
                             data=df_sup.to_dict("records") if DATA_OK else [],
                             page_size=12, sort_action="native", filter_action="native",
                             row_selectable="multi", selected_rows=[],
                             style_table={"overflowX":"auto"},
                             style_cell=TD(), style_header=TH(C_SUP_COL_HDR),
                             style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":C_SUP_ROW}]),
                     ]),
                 ]),
                 html.Div(style={"flex":"1","minWidth":"0"}, children=[
                     title_bar("Modelo No Supervisado", C_UNSUP_HDR),
                     card_wrap(bg=C_UNSUP_BG, children=[
                         dash_table.DataTable(
                             id="tabla_unsup",
                             columns=[{"name":"Producto","id":"producto"},
                                      {"name":"Score No Supervisado","id":"score_no_supervisado",
                                       "type":"numeric","format":{"specifier":".4f"}}],
                             data=df_unsup.to_dict("records") if DATA_OK else [],
                             page_size=12, sort_action="native", filter_action="native",
                             row_selectable="multi", selected_rows=[],
                             style_table={"overflowX":"auto"},
                             style_cell=TD(), style_header=TH(C_UNSUP_COL_HDR),
                             style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":C_UNSUP_ROW}]),
                     ]),
                 ]),
             ]),

    # Estadísticas lado a lado
    html.Div(style={"display":"flex","gap":"14px","marginBottom":"14px","alignItems":"flex-start"},
             children=[
                 html.Div(style={"flex":"1","minWidth":"0"}, children=[
                     title_bar("Estadísticas Descriptivas – Modelo Supervisado", C_SUP_HDR),
                     card_wrap(bg=C_SUP_BG, children=[
                         dash_table.DataTable(
                             id="tabla_stats_sup",
                             columns=[{"name":"Estadístico","id":"Estadístico"},
                                      {"name":"Predicción Supervisada","id":"Predicción Supervisada",
                                       "type":"numeric","format":{"specifier":".4f"}}],
                             data=[], style_table={"overflowX":"auto"},
                             style_cell=TD(), style_header=TH(C_SUP_COL_HDR),
                             style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":C_SUP_ROW}]),
                     ]),
                 ]),
                 html.Div(style={"flex":"1","minWidth":"0"}, children=[
                     title_bar("Estadísticas Descriptivas – Modelo No Supervisado", C_UNSUP_HDR),
                     card_wrap(bg=C_UNSUP_BG, children=[
                         dash_table.DataTable(
                             id="tabla_stats_unsup",
                             columns=[{"name":"Estadístico","id":"Estadístico"},
                                      {"name":"Score No Supervisado","id":"Score No Supervisado",
                                       "type":"numeric","format":{"specifier":".4f"}}],
                             data=[], style_table={"overflowX":"auto"},
                             style_cell=TD(), style_header=TH(C_UNSUP_COL_HDR),
                             style_data_conditional=[{"if":{"row_index":"odd"},"backgroundColor":C_UNSUP_ROW}]),
                     ]),
                 ]),
             ]),

    # Series de tiempo lado a lado
    html.Div(style={"display":"flex","gap":"14px","marginBottom":"14px","alignItems":"flex-start"},
             children=[
                 html.Div(style={"flex":"1","minWidth":"0"}, children=[
                     title_bar("Series de Tiempo – Modelo Supervisado", C_SUP_HDR),
                     card_wrap(bg=C_SUP_BG, children=[dcc.Graph(id="grafica_consumo_sup")]),
                 ]),
                 html.Div(style={"flex":"1","minWidth":"0"}, children=[
                     title_bar("Series de Tiempo – Modelo No Supervisado", C_UNSUP_HDR),
                     card_wrap(bg=C_UNSUP_BG, children=[dcc.Graph(id="grafica_consumo_unsup")]),
                 ]),
             ]),
])


# ── Helper: layout descriptivas ───────────────────────────
def _desc_layout(sfx, hdr, row_bg, card_bg, client_list):
    return html.Div([
        # Controles
        html.Div(style={**CARD(),"display":"flex","gap":"20px","flexWrap":"wrap","alignItems":"flex-end"},
                 children=[
                     html.Div([
                         html.Label("Filtrar por Cliente ID:", style=LABEL),
                         dcc.Dropdown(id=f"filter_cliente_{sfx}",
                                      options=[{"label":str(c),"value":c} for c in client_list],
                                      value=None, placeholder="Todos", clearable=True,
                                      style={**DROP,"width":"200px"}),
                     ]),
                     html.Div([
                         html.Label("Top N clientes:", style=LABEL),
                         dcc.Dropdown(id=f"top_n_desc_{sfx}", options=TOP_N_OPTIONS, value=10,
                                      clearable=False, style={**DROP,"width":"150px"}),
                     ]),
                 ]),
        # Categoría (fila completa)
        html.Div(style={"marginBottom":"14px"}, children=[
            title_bar("Categoría  (código)", hdr),
            card_wrap(bg=card_bg, children=[
                dcc.Graph(id=f"graf_categoria_{sfx}", style={"height":"300px"})]),
        ]),
        # Grid 2×2
        html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr","gap":"14px",
                        "marginBottom":"14px"},
                 children=[
                     html.Div([title_bar("Subcategoría  (código)", hdr),
                                card_wrap(bg=card_bg, children=[
                                    dcc.Graph(id=f"graf_subcategoria_{sfx}",style={"height":"240px"})])]),
                     html.Div([title_bar("Localidad  (código)", hdr),
                                card_wrap(bg=card_bg, children=[
                                    dcc.Graph(id=f"graf_localidad_{sfx}",style={"height":"240px"})])]),
                     html.Div([title_bar("Barrio  (código)", hdr),
                                card_wrap(bg=card_bg, children=[
                                    dcc.Graph(id=f"graf_barrio_{sfx}",style={"height":"240px"})])]),
                     html.Div([title_bar("Tipo de Producto  (código)", hdr),
                                card_wrap(bg=card_bg, children=[
                                    dcc.Graph(id=f"graf_tipo_{sfx}",style={"height":"240px"})])]),
                 ]),
        # Mapa + KPIs
        html.Div(style={"display":"flex","gap":"14px","alignItems":"flex-start"}, children=[
            html.Div(style={**CARD(),"flex":"3","minWidth":"0","marginBottom":"0"}, children=[
                html.P("Mapa de Geolocalización",
                       style={**T_H3,"margin":"0 0 8px 0"}),
                dcc.Graph(id=f"graf_mapa_{sfx}", style={"height":"400px"}),
            ]),
            html.Div(style={**CARD(),"flex":"1","minWidth":"180px","marginBottom":"0"}, children=[
                html.P("Resumen",style={**T_H3,"margin":"0 0 14px 0"}),
                html.Div(id=f"kpi_total_{sfx}",
                         style={"background":"#f0f5fb","borderRadius":"6px","padding":"12px 14px",
                                "marginBottom":"10px","borderLeft":f"4px solid {C_ORANGE}"},
                         children=[html.Span("Total clientes",style={"color":C_SUBTEXT,"fontSize":"12px","display":"block"}),
                                   html.Span("–",style={"color":C_NAVY,"fontWeight":"700","fontSize":"22px"})]),
                html.Div(id=f"kpi_sindir_{sfx}",
                         style={"background":"#fff5f5","borderRadius":"6px","padding":"12px 14px",
                                "borderLeft":f"4px solid {C_RED}"},
                         children=[html.Span("Sin dirección",style={"color":C_SUBTEXT,"fontSize":"12px","display":"block"}),
                                   html.Span("–",style={"color":C_RED,"fontWeight":"700","fontSize":"22px"})]),
            ]),
        ]),
    ])

tab2_layout = _desc_layout("unsup", C_UNSUP_HDR, C_UNSUP_ROW, C_UNSUP_BG, ids_unsup)
tab3_layout = _desc_layout("sup",   C_SUP_HDR,   C_SUP_ROW,   C_SUP_BG,   ids_sup)

# ── Pestaña 4: Métricas de los Modelos ────────────────────
METRICS = {"Exactitud":"67.4 %","AUC":"0.712","Recall (anomalías)":"41.3 %",
           "Precisión (anomalías)":"64.5 %","Perfil":"Mejor AUC · RECOMENDADO"}

tab4_layout = html.Div([
    html.Div(style={"display":"grid","gridTemplateColumns":"1fr 1fr 1fr","gap":"14px","marginBottom":"14px"},
             children=[
                 html.Div(style=CARD(), children=[
                     html.P("Curva ROC – LightGBM",
                            style={**T_H3,"margin":"0 0 8px 0"}),
                     dcc.Graph(id="graf_roc", style={"height":"360px"}),
                     html.Div(id="roc_error_box",style={"color":C_RED,"fontSize":"12px","marginTop":"6px"},
                              children=f"Error ROC: {roc_error}" if roc_error else ""),
                 ]),
                 html.Div(style=CARD(), children=[
                     html.P("Métricas – LightGBM",
                            style={**T_H3,"margin":"0 0 14px 0"}),
                     html.Div(style={"display":"flex","flexDirection":"column","gap":"9px"},
                              children=[
                                  html.Div(style={"background":"#f0f5fb","borderRadius":"6px","padding":"9px 12px",
                                                  "display":"flex","justifyContent":"space-between","alignItems":"center",
                                                  "borderLeft":f"4px solid {C_ORANGE}"},
                                           children=[html.Span(k,style={**T_BODY,"color":C_SUBTEXT}),
                                                     html.Span(v,style={**T_H3})])
                                  for k,v in METRICS.items()
                              ]),
                     html.Div(style={"marginTop":"14px","padding":"9px 12px","background":"#e8f4fb",
                                     "borderRadius":"6px","border":f"1px solid {C_GOLD}"},
                              children=html.P("LightGBM obtuvo el mayor AUC (0.712) con buen balance entre "
                                              "recall y precisión para detección de anomalías energéticas.",
                                             style={"color":C_SUBTEXT,"fontSize":"11px","margin":"0"})),
                 ]),
                 html.Div(style=CARD(), children=[
                     html.P("Matriz de Confusión – LightGBM",
                            style={**T_H3,"margin":"0 0 8px 0"}),
                     dcc.Graph(id="graf_confmat", style={"height":"360px"}),
                 ]),
             ]),
    html.Div(style=CARD(), children=[
        html.P("Análisis No Supervisado – Distancia de Mahalanobis",
               style={**T_H3,"margin":"0 0 4px 0"}),
        html.P("Ingresa un Cliente ID para compararlo frente a su grupo. Ordenado por fraud score descendente.",
               style={**T_CAP,"margin":"0 0 12px 0"}),
        html.Div(style={"display":"flex","gap":"10px","alignItems":"flex-end","marginBottom":"14px","flexWrap":"wrap"},
                 children=[
                     html.Div([
                         html.Label("Buscar Cliente ID:", style=LABEL),
                         dcc.Input(id="dd_cliente_mah", type="number",
                                   placeholder="Ej: 11116848", debounce=True,
                                   value=_mah_ids_ordered[0] if _mah_ids_ordered else None,
                                   style={"width":"200px","padding":"7px 10px","borderRadius":"5px",
                                          "border":f"1px solid {C_BORDER}","fontSize":"13px","color":C_TEXT}),
                     ]),
                     html.Div(style={"color":C_SUBTEXT,"fontSize":"11px","paddingBottom":"8px"},
                              children=(f"Top anómalo por defecto: {_mah_ids_ordered[0] if _mah_ids_ordered else '–'}"
                                        f"  ·  {len(_mah_ids_ordered)} clientes disponibles")),
                 ]),
        html.Div(id="label_cliente_activo",
                 style={"color":C_SUBTEXT,"fontSize":"12px","marginBottom":"8px"}),
        dcc.Graph(id="graf_mahalanobis", style={"height":"520px"}),
    ]),
])

# =========================================================
# 6) APP
# =========================================================
app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "DISICO – Anomalías Energía"

app.layout = html.Div(
    style={"background":C_BG,"minHeight":"100vh","fontFamily":"ui-sans-serif, system-ui, sans-serif"},
    children=[
        html.Div(style={"background":C_NAVY,"padding":"0 28px","display":"flex","alignItems":"center",
                        "gap":"14px","height":"62px","boxShadow":"0 3px 12px rgba(27,94,126,0.30)"},
                 children=[
                     html.Span("SW DISICO",style={"color":"white","fontWeight":"800",
                                                   "fontSize":"20px","letterSpacing":"1.5px",
                                                   "fontFamily":"ui-sans-serif, system-ui, sans-serif"}),
                     html.Span("·",style={"color":C_GOLD,"fontSize":"22px","marginLeft":"6px"}),
                     html.Span("Detección de Anomalías de Consumo Energético",
                               style={"color":C_STEEL,"fontSize":"14px","marginLeft":"4px",
                                      "fontFamily":"ui-sans-serif, system-ui, sans-serif"}),
                 ]),
        html.Div(style={"padding":"20px 24px"}, children=[
            dcc.Tabs(id="main_tabs", value="tab1", style={"marginBottom":"16px"},
                     children=[
                         dcc.Tab(label="Scores y Series",            value="tab1", style=TAB_STYLE, selected_style=TAB_SEL),
                         dcc.Tab(label="Descriptivas No Supervisado", value="tab2", style=TAB_STYLE, selected_style=TAB_SEL),
                         dcc.Tab(label="Descriptivas Supervisado",    value="tab3", style=TAB_STYLE, selected_style=TAB_SEL),
                         dcc.Tab(label="Metricas de los Modelos",     value="tab4", style=TAB_STYLE, selected_style=TAB_SEL),
                     ]),
            html.Div(id="tab_content"),
        ]),
    ],
)

# =========================================================
# 7) CALLBACKS
# =========================================================

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
    _plot_sup = {**PLOT, "plot_bgcolor":"#eef5fa"}
    fig.update_layout(height=380, legend_title_text="CLIENTE_ID",
                      legend={"bgcolor":"rgba(255,255,255,0.85)",
                              "bordercolor":C_BORDER,"borderwidth":1,
                              "font":{"size":12,"family":"ui-sans-serif, system-ui, sans-serif"}},
                      title={"font":{"size":13,"color":C_NAVY,
                                     "family":"ui-sans-serif, system-ui, sans-serif"}},
                      **_plot_sup)
    fig.update_xaxes(showgrid=True, gridcolor=C_BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=C_BORDER)
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
    _plot_unsup = {**PLOT, "plot_bgcolor":"#fdfaf0"}
    fig.update_layout(height=380, legend_title_text="CLIENTE_ID",
                      legend={"bgcolor":"rgba(255,255,255,0.85)",
                              "bordercolor":C_BORDER,"borderwidth":1,
                              "font":{"size":12,"family":"ui-sans-serif, system-ui, sans-serif"}},
                      title={"font":{"size":13,"color":C_NAVY,
                                     "family":"ui-sans-serif, system-ui, sans-serif"}},
                      **_plot_unsup)
    fig.update_xaxes(showgrid=True, gridcolor=C_BORDER)
    fig.update_yaxes(showgrid=True, gridcolor=C_BORDER)
    return fig


# ── Helpers descriptivas ──────────────────────────────────
def _cat_fig(col, df_cons, df_scores, score_col, fc, tn):
    _pe = {**PLOT,"margin":{"t":30,"b":20,"l":30,"r":10}}
    _pb = {**PLOT,"margin":{"t":20,"b":55,"l":50,"r":10}}
    def empty(m): return go.Figure().update_layout(title=m, **_pe)
    if df_cons.empty or col not in df_cons.columns: return empty(f"'{col}' no disponible.")
    n = int(tn) if int(tn) < 999999 else len(df_scores)
    top_ids = df_scores.sort_values(score_col, ascending=False).head(n)["producto"].tolist()
    dff = df_cons[df_cons["CLIENTE_ID"].isin(top_ids)]
    if fc is not None: dff = dff[dff["CLIENTE_ID"] == fc]
    if dff.empty: return empty("Sin datos.")
    dff = dff[[col,"CLIENTE_ID"]].copy()
    dff[col] = dff[col].astype(str).str.strip()
    dff = dff[~dff[col].isin(["nan","<NA>","None",""])]
    if dff.empty: return empty("Sin valores válidos.")
    counts = (dff.groupby(col, observed=True)["CLIENTE_ID"].nunique()
                 .reset_index().rename(columns={"CLIENTE_ID":"clientes"})
                 .sort_values("clientes", ascending=False))
    fig = px.bar(counts, x=col, y="clientes", color=col,
                 color_discrete_sequence=BAR_COLORS,
                 labels={col:f"Código {col}","clientes":"Clientes únicos"})
    fig.update_traces(marker_line_width=0)
    fig.update_xaxes(type="category", tickangle=-35, showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor=C_BORDER)
    fig.update_layout(showlegend=False, **_pb)
    return fig


def _map_fig(df_geo, df_scores, score_col, fc, tn):
    _pm = {**PLOT,"margin":{"t":10,"b":10,"l":10,"r":10}}
    def kpi(label, val, color):
        return [html.Span(label,style={"color":C_SUBTEXT,"fontSize":"12px","display":"block"}),
                html.Span(str(val),style={"color":color,"fontWeight":"700","fontSize":"22px"})]
    def empty_m(m): return go.Figure().update_layout(title=m,**_pm)
    if df_geo.empty:
        return empty_m("Sin datos de geolocalización."), kpi("Total clientes","–",C_NAVY), kpi("Sin dirección","–",C_RED)
    n = int(tn) if int(tn) < 999999 else len(df_scores)
    top_ids = df_scores.sort_values(score_col, ascending=False).head(n)["producto"].tolist()
    dff = df_geo[df_geo["producto"].isin(top_ids)].copy()
    if fc is not None: dff = dff[dff["producto"] == fc]
    total   = len(dff)
    sin_dir = int((dff["lat"].isna() | dff["lon"].isna()).sum())
    dff_p   = dff.dropna(subset=["lat","lon"]).copy()
    if dff_p.empty:
        return empty_m("Sin coordenadas válidas."), kpi("Total clientes",total,C_NAVY), kpi("Sin dirección",sin_dir,C_RED)
    dff_p["producto_str"] = dff_p["producto"].astype(int).astype(str)
    fig = px.scatter_map(dff_p, lat="lat", lon="lon", hover_name="producto_str",
                         color_discrete_sequence=[C_BLUE], zoom=11, height=400,
                         map_style="open-street-map")
    fig.update_traces(marker={"size":9,"opacity":0.85})
    fig.update_layout(**_pm)
    return fig, kpi("Total clientes",total,C_NAVY), kpi("Sin dirección",sin_dir,C_RED)


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
        z=[[923,165],[426,300]], x=["Normal (0)","Anómalo (1)"], y=["Normal (0)","Anómalo (1)"],
        text=[["TN = 923","FP = 165"],["FN = 426","TP = 300"]],
        texttemplate="%{text}",
        colorscale=[[0,"#1B5E7E"],[0.35,"#4A8BA8"],[0.65,"#E8B649"],[1,"#FF8C1A"]],
        showscale=True, textfont={"size":15,"color":"white"}, hoverongaps=False))
    fig.update_layout(xaxis_title="Predicción", yaxis_title="Valor Real",
                      yaxis_autorange="reversed", xaxis={"side":"bottom"},
                      **{**PLOT,"margin":{"t":20,"b":50,"l":70,"r":20}})
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
        fig.add_trace(go.Scatter(x=fpr,y=tpr,mode="lines",name=f"LightGBM (AUC={auc_val:.3f})",
                                 line={"color":C_BLUE,"width":2.5},
                                 fill="tozeroy",fillcolor="rgba(30,111,165,0.08)"))
        fig.add_trace(go.Scatter(x=[0,1],y=[0,1],mode="lines",name="Clasificador aleatorio",
                                 line={"color":C_SUBTEXT,"dash":"dash","width":1.5}))
        fig.update_layout(
            xaxis_title="Tasa de Falsos Positivos (1 – Especificidad)",
            yaxis_title="Tasa de Verdaderos Positivos (Sensibilidad)",
            legend={"x":0.55,"y":0.08,"bgcolor":"rgba(255,255,255,0.85)",
                    "bordercolor":C_BORDER,"borderwidth":1},
            xaxis={"showgrid":True,"gridcolor":C_BORDER,"range":[0,1]},
            yaxis={"showgrid":True,"gridcolor":C_BORDER,"range":[0,1]}, **PLOT)
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
    nm = (~es_an)&(~et)
    if nm.any(): fig.add_trace(go.Scatter(x=X[nm,0],y=X[nm,1],mode="markers",name="Normal",
                                           opacity=0.75,marker={"color":"royalblue","size":8,"line":{"color":"black","width":0.5}}))
    am = es_an&(~et)
    if am.any(): fig.add_trace(go.Scatter(x=X[am,0],y=X[am,1],mode="markers",name="Anómalo grupo (>2σ)",
                                           opacity=0.85,marker={"color":"crimson","size":9,"line":{"color":"black","width":0.5}}))
    if et.any(): fig.add_trace(go.Scatter(x=X[et,0],y=X[et,1],mode="markers+text",
                                           name=f"Cliente {cliente_id}",
                                           text=[str(int(cliente_id))]*int(et.sum()),
                                           textposition="top right",
                                           textfont={"size":10,"color":"darkgoldenrod"},
                                           marker={"color":"gold","size":14,"symbol":"star",
                                                   "line":{"color":"darkorange","width":1.5}}))
    ax0,ax1 = float(X.min())*0.95, float(X.max())*1.05
    fig.add_trace(go.Scatter(x=[ax0,ax1],y=[ax0,ax1],mode="lines",name="y = x",
                             line={"color":"gray","dash":"dash","width":1.5}))
    ve,ve2 = np.linalg.eigh(cov)
    o = ve.argsort()[::-1]; ve,ve2 = ve[o],ve2[:,o]
    tr = np.arctan2(*ve2[:,0][::-1])
    ae,be = 2*np.sqrt(ve[0]),2*np.sqrt(ve[1])
    t = np.linspace(0,2*np.pi,200)
    ex = mu[0]+ae*np.cos(t)*np.cos(tr)-be*np.sin(t)*np.sin(tr)
    ey = mu[1]+ae*np.cos(t)*np.sin(tr)+be*np.sin(t)*np.cos(tr)
    fig.add_trace(go.Scatter(x=ex,y=ey,mode="lines",name="2σ",line={"color":"forestgreen","width":2}))
    fig.update_layout(
        title={"text":" | ".join(f"{k}: {v}" for k,v in fd.items()),"font":{"size":11}},
        xaxis_title="Consumo año anterior", yaxis_title="Consumo año actual",
        legend={"x":0.01,"y":0.99,"bgcolor":"rgba(255,255,255,0.85)","bordercolor":C_BORDER,
                "borderwidth":1,"font":{"size":11}},
        xaxis={"showgrid":True,"gridcolor":C_BORDER},
        yaxis={"showgrid":True,"gridcolor":C_BORDER}, **_pm)
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
