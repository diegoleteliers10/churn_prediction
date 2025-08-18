# 🚀 Predicción de Abandono de Clientes (Churn) - Telecomunicaciones

## 📋 Descripción del Proyecto

Sistema completo de **Machine Learning en Producción** para predecir qué clientes tienen mayor probabilidad de abandonar un servicio de telecomunicaciones. El proyecto incluye desde análisis exploratorio hasta un **pipeline ejecutable** para predicciones en tiempo real.

## 🎯 Objetivos del Proyecto

- **Principal**: Identificar clientes con alta probabilidad de churn
- **Métrica Clave**: Maximizar Recall (≥85%)
- **Impacto**: Reducir tasa de abandono mediante acciones preventivas
- **Producción**: Pipeline ejecutable para uso empresarial

## 🏗️ Estructura del Proyecto (Actualizada)

```
churn_prediction/
├── 📁 data/
│   ├── raw/                    # Dataset original Telco-Customer-Churn.csv
│   └── processed/              # Datos con features engineering
├── 📁 notebooks/
│   ├── exploratory/            # 01_data_overview.ipynb - Análisis exploratorio
│   ├── modeling/               # 01_modeling.ipynb - Desarrollo y evaluación de modelos
│   └── pipelines/              # 01_pipelines.ipynb - Pipeline de predicción
├── 📁 src/
│   ├── data/                   # Scripts de carga de datos
│   ├── utils.py                # Utilidades centralizadas (logging, config, etc.)
│   └── __init__.py
├── 📁 models/
│   ├── trained/                # churn_model_xgb.pkl - Modelo XGBoost entrenado
│   ├── encoders/               # label_encoders.pkl - Encoders para variables categóricas
│   ├── pipelines/              # 🚀 PIPELINE DE PRODUCCIÓN
│   │   ├── pipeline_script.py  # Script principal ejecutable
│   │   ├── quick_test.py       # Test rápido del modelo
│   │   ├── test_single_customer.py # Test completo con menú
│   │   ├── example_customer.csv # Datos de ejemplo para pruebas
│   │   └── README.md           # Documentación del pipeline
│   └── model_metrics.json      # Métricas del modelo final
├── 📁 config/
│   └── config.yaml             # Configuración centralizada del proyecto
├── 📁 reports/
│   └── figures/                # Visualizaciones y dashboards
├── requirements.txt             # Dependencias Python
└── README.md                   # Este archivo
```

## 🚀 Características Implementadas

### ✅ **Análisis Exploratorio Completo**
- Análisis de distribución de variables
- Identificación de patrones de churn
- Creación de features derivadas (tenure_group, total_services)
- Visualizaciones informativas

### ✅ **Modelado Avanzado**
- **XGBoost** como modelo final (mejor performance)
- Feature engineering automático
- Optimización de thresholds
- Evaluación con múltiples métricas

### ✅ **Pipeline de Producción**
- Script ejecutable desde línea de comandos
- Soporte para múltiples formatos (CSV, Excel, JSON)
- Modo batch para múltiples clientes
- Manejo automático de errores
- Logging completo de operaciones

### ✅ **Utilidades Centralizadas**
- Sistema de logging unificado
- Configuración centralizada (config.yaml)
- Funciones de carga/guardado con logging
- Manejo automático de paths del proyecto

## 📊 Dataset y Features

**Fuente**: Telco Customer Churn Dataset (Kaggle)
- **Registros**: 7,043 clientes
- **Variables Originales**: 21 features
- **Features Derivadas**: 
  - `tenure_group`: Segmentación por antigüedad
  - `total_services`: Conteo de servicios activos
- **Target**: Churn (Yes/No)

### 🎯 **Variables Clave Identificadas**
1. **Contract** (Mayor predictor)
2. **InternetService**
3. **OnlineSecurity**
4. **TechSupport**
5. **tenure** (Antigüedad)

## 🏆 Modelo Final

### **XGBoost Classifier**
- **Performance**: AUC = 0.752
- **Recall**: 76.74% (detecta churn)
- **Precision**: 51.34% (evita falsas alarmas)
- **F1-Score**: 61.52%
- **Threshold óptimo**: 0.4

### **Configuración del Modelo**
```yaml
model:
  target_column: "Churn"
  test_size: 0.15
  random_state: 42
  target_recall: 0.85
  target_precision: 0.60
  target_auc: 0.80
```

## 🚀 Cómo Usar el Modelo

### **1. Instalación y Setup**

```bash
# Clonar/descargar el proyecto
cd churn_prediction

# Crear entorno virtual
python -m venv venv
venv\Scripts\activate     # Windows
source venv/bin/activate  # Linux/Mac

# Instalar dependencias
pip install -r requirements.txt
```

### **2. Verificar Instalación**

```bash
cd models/pipelines
python pipeline_script.py --check
```

### **3. Predicción Individual**

```bash
# Usar cliente de ejemplo
python pipeline_script.py --customer_data example_customer.csv

# Con threshold personalizado
python pipeline_script.py --customer_data customer.csv --threshold 0.5

# Guardar resultados
python pipeline_script.py --customer_data customer.csv --output results.json
```

### **4. Predicción en Lote**

```bash
python pipeline_script.py --customer_data customers_batch.csv --batch
```

### **5. Tests Rápidos**

```bash
# Test automático con cliente predefinido
python quick_test.py

# Test interactivo con menú
python test_single_customer.py
```

## 📊 Formato de Datos de Entrada

### **CSV Requerido (sin columna Churn)**
```csv
customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
CLIENT001,Male,0,No,No,5,Yes,No,DSL,No,No,No,No,No,No,Month-to-month,Yes,Electronic check,55.0,275.0
```

### **Valores Aceptados**
- **gender**: "Male", "Female"
- **SeniorCitizen**: 0, 1
- **Partner/Dependents**: "Yes", "No"
- **Contract**: "Month-to-month", "One year", "Two year"
- **PaymentMethod**: "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"

## 📤 Formato de Salida

### **JSON con Predicción**
```json
{
  "customer_id": "CLIENT001",
  "churn_probability": 0.8234,
  "will_churn": true,
  "risk_level": "High Risk",
  "confidence": "High",
  "threshold_used": 0.4,
  "prediction_timestamp": "2024-01-15T10:30:00.123456"
}
```

### **Interpretación de Resultados**
- **High Risk** (≥70%): Acción inmediata requerida
- **Medium Risk** (40-70%): Intervención urgente
- **Low Risk** (<40%): Monitoreo rutinario

## 🔧 Configuración Avanzada

### **Rutas Personalizadas**
```bash
python pipeline_script.py \
  --customer_data customer.csv \
  --model_path /path/to/model.pkl \
  --encoders_path /path/to/encoders.pkl
```

### **Modificar Configuración**
Editar `config/config.yaml`:
```yaml
model:
  test_size: 0.20          # Cambiar split de datos
  target_recall: 0.90      # Ajustar métrica objetivo
  random_state: 123        # Cambiar semilla
```

## 📈 Métricas de Evaluación

### **Modelo Final (XGBoost)**
- **AUC-ROC**: 0.752
- **Recall**: 76.74% ✅
- **Precision**: 51.34% ✅
- **F1-Score**: 61.52%
- **Threshold**: 0.4

### **Comparación con Objetivos**
- **Recall objetivo**: 85% ❌ (76.74% actual)
- **Precision objetivo**: 60% ❌ (51.34% actual)
- **AUC objetivo**: 80% ❌ (75.2% actual)

## 🚨 Solución de Problemas

### **Error: "Invalid columns: Churn: object"**
**Causa**: El CSV contiene la columna `Churn` (variable objetivo)
**Solución**: Usar solo datos de nuevos clientes sin la columna `Churn`

### **Error: "Modelo no encontrado"**
**Causa**: Paths incorrectos
**Solución**: Ejecutar desde `models/pipelines/` o usar rutas absolutas

### **Error: "Valor desconocido en columna"**
**Causa**: Valores no vistos durante entrenamiento
**Solución**: El pipeline asigna automáticamente valor por defecto

## 🎯 Casos de Uso

### **1. Predicción Individual**
- Agente de atención al cliente
- Sistema de alertas en tiempo real
- Dashboard de gestión

### **2. Predicción en Lote**
- Análisis mensual de base de clientes
- Campañas de retención
- Reportes ejecutivos

### **3. Integración con Sistemas**
- API REST (desarrollar)
- Base de datos empresarial
- CRM/ERP systems

## 🔮 Próximos Pasos

### **Mejoras del Modelo**
- [ ] Aumentar Recall a 85% objetivo
- [ ] Implementar ensemble methods
- [ ] Feature selection automático
- [ ] Hyperparameter tuning avanzado

### **Funcionalidades Adicionales**
- [ ] API REST para predicciones
- [ ] Dashboard web interactivo
- [ ] Sistema de alertas automáticas
- [ ] Integración con bases de datos

### **Deployment**
- [ ] Containerización (Docker)
- [ ] CI/CD pipeline
- [ ] Monitoreo de performance
- [ ] A/B testing

## 👥 Contribución

1. **Fork** el proyecto
2. **Crear** branch de feature (`git checkout -b feature/AmazingFeature`)
3. **Commit** cambios (`git commit -m 'Add AmazingFeature'`)
4. **Push** al branch (`git push origin feature/AmazingFeature`)
5. **Abrir** Pull Request

## 📝 Licencia

MIT License - ver LICENSE file para detalles

## 📞 Contacto

**Autor**: Diego Letelier
**Proyecto**: Churn Prediction - Telecomunicaciones
**Versión**: 1.0.0
**Fecha**: Enero 2024

---
*Proyecto desarrollado como parte del portafolio de Data Science y Machine Learning en Producción*

## 🎉 ¡El Proyecto Está Listo para Producción!

Este sistema incluye:
- ✅ **Modelo entrenado** y validado
- ✅ **Pipeline ejecutable** desde línea de comandos
- ✅ **Scripts de prueba** para verificación
- ✅ **Documentación completa** de uso
- ✅ **Configuración centralizada**
- ✅ **Logging y manejo de errores**

**¡Puedes empezar a predecir churn de clientes inmediatamente!**
