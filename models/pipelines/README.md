# Pipeline de Predicción de Churn

Este directorio contiene el script ejecutable para predecir la probabilidad de churn de nuevos clientes usando el modelo XGBoost entrenado.

## 📁 Archivos

- `pipeline_script.py` - Script principal ejecutable
- `example_customer.csv` - Ejemplo de datos de cliente para pruebas
- `README.md` - Este archivo de documentación

## 🚀 Uso del Script

### Verificar Instalación

```bash
python pipeline_script.py --check
```

### Predicción Simple

```bash
python pipeline_script.py --customer_data example_customer.csv
```

### Con Threshold Personalizado

```bash
python pipeline_script.py --customer_data example_customer.csv --threshold 0.5
```

### Guardar Resultados en Archivo Específico

```bash
python pipeline_script.py --customer_data example_customer.csv --output results.json
```

### Modo Batch para Múltiples Clientes

```bash
python pipeline_script.py --customer_data customers_batch.csv --batch
```

### Rutas Personalizadas

```bash
python pipeline_script.py --customer_data customer.csv --model_path /path/to/model.pkl --encoders_path /path/to/encoders.pkl
```

## 📊 Formato de Datos de Entrada

El script acepta archivos en los siguientes formatos:

### CSV
```csv
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
TEST001,Male,0,No,No,2,Yes,No,Fiber optic,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,85.0,170.0
```

### Excel (.xlsx, .xls)
Mismo formato que CSV pero en archivo Excel.

### JSON
```json
{
  "customerID": "TEST001",
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "No",
  "Dependents": "No",
  "tenure": 2,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 85.0,
  "TotalCharges": 170.0
}
```

## 📋 Campos Requeridos

Todos los campos del dataset original son requeridos:

- **Identificación**: `customerID`
- **Demográficos**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Servicios**: `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`
- **Contrato**: `Contract`, `PaperlessBilling`, `PaymentMethod`
- **Financieros**: `tenure`, `MonthlyCharges`, `TotalCharges`

## 🎯 Valores Aceptados

### Campos Categóricos
- `gender`: "Male", "Female"
- `SeniorCitizen`: 0, 1
- `Partner`: "Yes", "No"
- `Dependents`: "Yes", "No"
- `PhoneService`: "Yes", "No"
- `MultipleLines`: "Yes", "No", "No phone service"
- `InternetService`: "DSL", "Fiber optic", "No"
- `OnlineSecurity`: "Yes", "No", "No internet service"
- `OnlineBackup`: "Yes", "No", "No internet service"
- `DeviceProtection`: "Yes", "No", "No internet service"
- `TechSupport`: "Yes", "No", "No internet service"
- `StreamingTV`: "Yes", "No", "No internet service"
- `StreamingMovies`: "Yes", "No", "No internet service"
- `Contract`: "Month-to-month", "One year", "Two year"
- `PaperlessBilling`: "Yes", "No"
- `PaymentMethod`: "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"

### Campos Numéricos
- `tenure`: Entero (meses de antigüedad)
- `MonthlyCharges`: Float (cargo mensual)
- `TotalCharges`: Float o String (cargo total)

## 📤 Formato de Salida

El script genera un archivo JSON con los resultados:

```json
{
  "customer_id": "TEST001",
  "churn_probability": 0.9103,
  "will_churn": true,
  "risk_level": "High Risk",
  "confidence": "High",
  "threshold_used": 0.4,
  "prediction_timestamp": "2024-01-15T10:30:00.123456"
}
```

### Campos de Salida

- `customer_id`: ID del cliente
- `churn_probability`: Probabilidad de churn (0.0 - 1.0)
- `will_churn`: Predicción binaria (true/false)
- `risk_level`: Nivel de riesgo ("Low Risk", "Medium Risk", "High Risk")
- `confidence`: Nivel de confianza ("Low", "Medium", "High")
- `threshold_used`: Threshold utilizado para la decisión
- `prediction_timestamp`: Timestamp de la predicción

## ⚙️ Configuración

### Threshold por Defecto
El threshold por defecto es **0.4**, lo que significa:
- Probabilidad ≥ 0.4: Cliente predicho como CHURN
- Probabilidad < 0.4: Cliente predicho como NO CHURN

### Niveles de Riesgo
- **High Risk**: Probabilidad ≥ 0.7
- **Medium Risk**: Probabilidad entre 0.4 y 0.7
- **Low Risk**: Probabilidad < 0.4

### Niveles de Confianza
- **High**: Probabilidad muy alejada de 0.5 (|prob - 0.5| > 0.3)
- **Medium**: Probabilidad moderadamente alejada de 0.5 (0.15 < |prob - 0.5| ≤ 0.3)
- **Low**: Probabilidad cercana a 0.5 (|prob - 0.5| ≤ 0.15)

## 🔧 Requisitos

- Python 3.7+
- pandas
- numpy
- scikit-learn
- joblib
- xgboost

## 📝 Ejemplos de Uso

### 1. Predicción Individual
```bash
python pipeline_script.py --customer_data customer.csv
```

### 2. Predicción con Threshold Alto (Más Conservador)
```bash
python pipeline_script.py --customer_data customer.csv --threshold 0.6
```

### 3. Predicción con Threshold Bajo (Más Sensible)
```bash
python pipeline_script.py --customer_data customer.csv --threshold 0.3
```

### 4. Procesamiento en Lote
```bash
python pipeline_script.py --customer_data customers_batch.csv --batch --output batch_results.json
```

## 🚨 Manejo de Errores

El script maneja automáticamente:
- Valores desconocidos en campos categóricos
- Archivos de entrada corruptos o mal formateados
- Errores en la carga del modelo
- Timeouts en predicciones

## 📞 Soporte

Para problemas o preguntas, revisar:
1. Los logs de error en la consola
2. El formato de los datos de entrada
3. La existencia de los archivos del modelo y encoders
4. Los permisos de escritura en el directorio de salida
