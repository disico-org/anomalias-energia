import re
import os
import argparse
import pandas as pd
from multiprocessing import Pool, cpu_count
from geopy.geocoders import ArcGIS, Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm


# =========================
# Normalización de direcciones
# =========================

def normalizar_direccion(dir_oficial, dir_instalacion=None, ciudad="Cali", pais="Colombia"):
    """Limpia y normaliza direcciones colombianas."""
    direccion = dir_oficial if isinstance(dir_oficial, str) else dir_instalacion
    if not isinstance(direccion, str):
        return None

    direccion = direccion.upper().strip()
    for corte in ['APTO', 'APARTAMENTO', 'LC', 'BLOQUE', 'INTERIOR', 'CASA']:
        direccion = direccion.split(corte)[0].strip()
    direccion = direccion.replace('MACROMEDICION', '').replace('MACRO', '').strip()

    reemplazos = {
        r'\bK\b': 'CARRERA', r'\bCR\b': 'CARRERA', r'\bCARRERA\b': 'CARRERA',
        r'\bCALLE\b': 'CALLE', r'\bCL\b': 'CALLE',
        r'\bAVENIDA\b': 'AVENIDA', r'\bAV\b': 'AVENIDA',
        r'\bDIAGONAL\b': 'DIAGONAL',
        r'\bTRANSVERSAL\b': 'TRANSVERSAL', r'\bTR\b': 'TRANSVERSAL',
        r'\bVD\b': 'VEREDA',
    }
    for patron, reemplazo in reemplazos.items():
        direccion = re.sub(patron, reemplazo, direccion)

    direccion = re.sub(r'\b0+(\d+)\b', r'\1', direccion)
    direccion = re.sub(r'\s+0000\s*$', '', direccion)
    direccion = re.sub(r'\s+', ' ', direccion).strip()

    patron = r'((?:CARRERA|CALLE|AVENIDA|DIAGONAL|TRANSVERSAL)\s+[\w]+(?:\s+[A-Z]+)*)\s+(\d+\s*[A-Z]?\s*-\s*\d+)'
    direccion = re.sub(patron, r'\1 # \2', direccion)

    return f"{direccion}, {ciudad}, {pais}"


def geocode_con_fallback(row, ciudad="Cali"):
    """Normaliza la dirección oficial; si falla usa la de instalación."""
    dir_limpia = normalizar_direccion(row["direccion_oficial"], ciudad=ciudad)
    if dir_limpia:
        return pd.Series([dir_limpia, "direccion_oficial"])

    dir_limpia2 = normalizar_direccion(row["direccion_de_instalacion"], ciudad=ciudad)
    if dir_limpia2:
        return pd.Series([dir_limpia2, "instalacion"])

    return pd.Series([None, "fallido"])


# =========================
# Geocoding (ArcGIS + Nominatim fallback)
# =========================

def init_worker():
    global arcgis
    global geocode_nominatim

    arcgis = ArcGIS(timeout=10)
    nominatim = Nominatim(user_agent="disico_anomalias")
    geocode_nominatim = RateLimiter(nominatim.geocode, min_delay_seconds=1)


def geocode_gratis(direccion):
    if pd.isna(direccion):
        return (None, None, "sin_direccion")

    # Intento 1: ArcGIS
    try:
        loc = arcgis.geocode(direccion)
        if loc:
            return (loc.latitude, loc.longitude, "arcgis")
    except Exception:
        pass

    # Intento 2: Nominatim
    try:
        loc = geocode_nominatim(direccion)
        if loc:
            return (loc.latitude, loc.longitude, "nominatim")
    except Exception:
        pass

    return (None, None, "fallido")


# =========================
# Procesamiento por chunk
# =========================

def procesar_chunk(df_chunk, n_workers):
    direcciones = df_chunk["direccion_limpia"].tolist()

    with Pool(processes=n_workers, initializer=init_worker) as pool:
        resultados = list(tqdm(pool.imap(geocode_gratis, direcciones), total=len(direcciones)))

    df_chunk = df_chunk.copy()
    df_chunk["lat"]    = [r[0] for r in resultados]
    df_chunk["lon"]    = [r[1] for r in resultados]
    df_chunk["source"] = [r[2] for r in resultados]

    return df_chunk


# =========================
# Main
# =========================

def main(input_file, output_file, chunk_size, n_workers):
    df = pd.read_csv(input_file)

    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)

    if not os.path.exists(output_file):
        df.to_csv(output_file, index=False)

    print(f"Total registros sin lat: {df['lat'].isna().sum()}")
    indices_nan = df[df["lat"].isna()].index

    for start in range(0, len(indices_nan), chunk_size):
        chunk_indices = indices_nan[start:start + chunk_size]
        print(f"\nProcesando chunk {start} - {start + chunk_size}")

        df_chunk_procesado = procesar_chunk(df.loc[chunk_indices].copy(), n_workers)

        df.loc[chunk_indices, ["lat", "lon", "source"]] = \
            df_chunk_procesado[["lat", "lon", "source"]].values

        df.to_csv(output_file, index=False)
        print("Archivo actualizado")

    print("Proceso terminado.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      required=True, help="CSV de entrada con columna direccion_limpia")
    parser.add_argument("--output",     required=True, help="CSV de salida con lat/lon/source")
    parser.add_argument("--chunk_size", type=int, default=5000)
    parser.add_argument("--workers",    type=int, default=max(1, cpu_count() - 1))
    args = parser.parse_args()

    main(args.input, args.output, args.chunk_size, args.workers)
