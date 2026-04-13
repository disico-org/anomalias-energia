"""
Entrypoint para DISICO Dashboard en Docker.

Antes de iniciar el dashboard, verifica si los archivos de datos
complementarios existen. Si no, genera datos de ejemplo automáticamente.

Control via variable de entorno DUMMY_DATA:
  auto     (default) — genera datos de ejemplo SOLO si faltan archivos
  generate           — siempre regenera datos de ejemplo (sobreescribe)
  clean              — elimina datos de ejemplo y arranca con lo que haya
  off                — no toca nada, arranca el dashboard tal cual
"""

import os
import sys
from pathlib import Path

# Archivos que el dashboard necesita además de los 2 Excel
REQUIRED_FILES = [
    "consumo_011223_01062025_filtrado.parquet",
    "georeferencias.csv",
    "informacion_curva_roc.pickle",
    "resultados_1.parquet",
    "resultados_df_last.csv",
]

EXCEL_FILES = [
    "top_5000_supervisado.xlsx",
    "top_5000_no_supervisado.xlsx",
]


def check_missing(data_path: Path) -> list[str]:
    """Retorna lista de archivos requeridos que no existen."""
    return [f for f in REQUIRED_FILES if not (data_path / f).exists()]


def has_excel(data_path: Path) -> bool:
    """Verifica que al menos 1 Excel de rankings exista."""
    return any((data_path / f).exists() for f in EXCEL_FILES)


def run_generate(data_path: Path):
    """Ejecuta el generador de datos de ejemplo."""
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    from generate_dummy_data import cmd_generate
    cmd_generate(data_path)


def run_clean(data_path: Path):
    """Ejecuta la limpieza de datos de ejemplo."""
    sys.path.insert(0, str(Path(__file__).parent / "scripts"))
    from generate_dummy_data import cmd_clean
    cmd_clean(data_path, force=True)


def main():
    data_path = Path(os.environ.get("DATA_PATH", "/app/data"))
    mode = os.environ.get("DUMMY_DATA", "auto").lower().strip()

    print(f"[entrypoint] DATA_PATH={data_path}")
    print(f"[entrypoint] DUMMY_DATA={mode}")

    if mode == "off":
        print("[entrypoint] Modo OFF — sin verificación de datos.")

    elif mode == "clean":
        print("[entrypoint] Modo CLEAN — eliminando datos de ejemplo...")
        run_clean(data_path)

    elif mode == "generate":
        if not has_excel(data_path):
            print("[entrypoint] ⚠ No se encontraron los Excel de rankings. No se puede generar.")
        else:
            print("[entrypoint] Modo GENERATE — regenerando datos de ejemplo...")
            run_generate(data_path)

    elif mode == "auto":
        missing = check_missing(data_path)
        if not missing:
            print("[entrypoint] Todos los archivos presentes — no se genera nada.")
        elif not has_excel(data_path):
            print(f"[entrypoint] Faltan {len(missing)} archivos pero no hay Excel de rankings.")
            print("[entrypoint] El dashboard arrancará con datos parciales.")
        else:
            print(f"[entrypoint] Faltan {len(missing)} archivos: {', '.join(missing)}")
            print("[entrypoint] Generando datos de ejemplo automáticamente...")
            run_generate(data_path)
    else:
        print(f"[entrypoint] ⚠ Valor DUMMY_DATA='{mode}' no reconocido. Usando 'auto'.")
        # Recurse with auto
        os.environ["DUMMY_DATA"] = "auto"
        main()
        return

    # Iniciar dashboard
    print("\n[entrypoint] Iniciando dashboard...")
    sys.path.insert(0, str(Path(__file__).parent / "src"))
    from Dash import app
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False,
    )


if __name__ == "__main__":
    main()
