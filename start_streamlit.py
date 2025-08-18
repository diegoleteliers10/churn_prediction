#!/usr/bin/env python3
"""
Script para iniciar la aplicación Streamlit
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Función principal"""
    print("🚀 Iniciando Churn Prediction App...")
    
    # Verificar que streamlit esté instalado
    try:
        import streamlit
        print("✅ Streamlit está instalado")
    except ImportError:
        print("❌ Streamlit no está instalado")
        print("💡 Instala con: pip install streamlit")
        return
    
    # Obtener ruta del archivo principal
    current_dir = Path.cwd()
    app_file = current_dir / "src" / "streamlit_app.py"
    
    if not app_file.exists():
        print(f"❌ Archivo de aplicación no encontrado: {app_file}")
        return
    
    print(f"📁 Aplicación encontrada en: {app_file}")
    print("�� Iniciando servidor web...")
    print("📖 La aplicación estará disponible en: http://localhost:8501")
    print("🔄 Presiona Ctrl+C para detener")
    
    # Iniciar Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", str(app_file),
            "--server.port", "8501",
            "--server.address", "localhost"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Aplicación detenida por el usuario")
    except Exception as e:
        print(f"❌ Error al iniciar la aplicación: {e}")

if __name__ == "__main__":
    main()