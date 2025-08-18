import pandas as pd
from pathlib import Path

# Definir paths
input_path = Path("data/processed")
output_path = Path("data/external")
input_files = list(input_path.glob("*.csv"))

if not input_files:
    raise FileNotFoundError(f"No se encontró ningún archivo CSV en {input_path}")

# Usar el primer archivo encontrado
input_csv = input_files[0]
df = pd.read_csv(input_csv)

# Cortar a los primeros 10 registros
df_cut = df.head(10)

# Crear carpeta de salida si no existe
output_path.mkdir(parents=True, exist_ok=True)

# Guardar el nuevo CSV
output_csv = output_path / input_csv.name
df_cut.to_csv(output_csv, index=False)

print(f"✅ Archivo generado: {output_csv} ({len(df_cut)} filas)")
