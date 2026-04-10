import re
import pandas as pd

def estandarizar_direccion(dir_text: str) -> str:
    """
    Estandariza un texto de dirección a un formato unificado y limpio.

    Esta función realiza una serie de transformaciones y limpiezas sobre el texto de la dirección:
    - Convierte todo a mayúsculas.
    - Elimina paréntesis y su contenido.
    - Elimina sufijos desde "CAS", "APT" o "LOC" en adelante.
    - Estandariza abreviaciones comunes (CLL, AVE, MNZ, LTE).
    - Limpia espacios múltiples.
    - Reestructura y da formato a patrones comunes de direcciones como:
      "CALLE 7 NORTE 5A-66" → "CALLE 7 NORTE #5A-66"
      "CRR BELEN 23 103" → "CARRERA BELEN #23-103"
      y otros casos específicos.
    - Si ningún patrón conocido aplica, retorna la dirección limpiada básica.

    Parámetros
    ----------
    dir_text : str
        Texto original de la dirección a estandarizar.

    Retorna
    -------
    str
        Dirección estandarizada y limpia.
    """

    dir_text = dir_text.upper()

    # Quitar todo lo que esté entre paréntesis
    dir_text = re.sub(r'\([^)]*\)', '', dir_text)

    # Eliminar todo desde "CAS", "APT" o "LOC" en adelante
    dir_text = re.sub(r'\s+(CAS\s|APT\s|LOC\s).*$', '', dir_text)

    # Estandarizar abreviaciones
    dir_text = re.sub(r'\bCLL\b', 'CALLE', dir_text)
    dir_text = re.sub(r'\bAVE\b', 'AVENIDA', dir_text)
    dir_text = re.sub(r'\bMNZ\b', 'MANZANA', dir_text)
    dir_text = re.sub(r'\bLTE\b', 'LOTE', dir_text)

    # Limpiar espacios múltiples
    dir_text = re.sub(r'\s+', ' ', dir_text).strip()

    # Buscar patrones frecuentes de direcciones y formatear
    match = re.search(r'\b([A-ZÁÉÍÓÚÑ]+)\s+(\d+[A-Z]{0,2})\s+(\d+[A-Z]{0,2})\s*-\s*(\d+)', dir_text)
    if match:
        tipo_via = match.group(1)
        numero1 = match.group(2)
        numero2 = match.group(3)
        numero3 = match.group(4)
        return f"{tipo_via} {numero1} #{numero2}-{numero3}"

    match_2 = re.search(r'\b(CALLE|AVENIDA|MANZANA|LOTE)\s+(\d+[A-Z]{0,2})\s+(\d+[A-Z]{0,2})\s+(\d+)', dir_text)
    if match_2:
        tipo_via, numero1, numero2, numero3 = match_2.groups()
        return f"{tipo_via} {numero1} #{numero2}-{numero3}"


    # Casos específicos (muy estructurados)
    match = re.search(r'\bAVENIDA CANAL BOGOTA\s+(\d+)\s+(\d+)\b', dir_text)
    if match:
        n1, n2 = match.groups()
        return f"AVENIDA CANAL BOGOTA #{n1}-{n2}"

    match = re.search(r'\b(CALLE \d+ NORTE)\s+(\d+[A-Z]?)\s*-\s*(\d+)\b', dir_text)
    if match:
        via, n1, n2 = match.groups()
        return f"{via} #{n1}-{n2}"

    match = re.search(r'\bCRR ANT BELEN\s+(\d+[A-Z]?)\s*-\s*(\d+)\b', dir_text)
    if match:
        n1, n2 = match.groups()
        return f"CARRERA ANTIGUA BELEN #{n1}-{n2}"

    match = re.search(r'\bAVENIDA CANAL BOGOTA\s+(\d+[A-Z]?)\s+(\d+)\b', dir_text)
    if match:
        n1, n2 = match.groups()
        return f"AVENIDA CANAL BOGOTA #{n1}-{n2}"

    match = re.search(r'\b(AVENIDA LIBERTADORES)\s+(\d+[A-Z]?)\s*-\s*(\d+)\b', dir_text)
    if match:
        via, n1, n2 = match.groups()
        return f"{via} #{n1}-{n2}"

    match = re.search(r'\bCRR BELEN\s+(\d+[A-Z]?)\s+(\d+)\b', dir_text)
    if match:
        n1, n2 = match.groups()
        return f"CARRERA BELEN #{n1}-{n2}"

    return dir_text

def feature_direccion_limpia(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['DIRECCION_LIMPIA'] = df['DIRECCION'].progress_apply(estandarizar_direccion)
    df['DIRECCION_LIMPIA'] = (
        df['DIRECCION_LIMPIA'] + ', ' +
        df['BARRIO'].astype(str) + ', ' +
        df['MUNICIPIO'].str.title() + ', NORTE DE SANTANDER, COLOMBIA'
    )
    return df