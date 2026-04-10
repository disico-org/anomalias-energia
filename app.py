"""
Entry point para DISICO Dashboard en EasyPanel.
Este archivo es el punto de entrada para Gunicorn.
"""
import os
import sys
from pathlib import Path

# Añadir directorio src al path para imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Importar la aplicación Dash
from Dash import app

# Obtener el servidor Flask subyacente para Gunicorn
server = app.server

if __name__ == "__main__":
    # Modo producción con Gunicorn
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 8050)),
        debug=False
    )