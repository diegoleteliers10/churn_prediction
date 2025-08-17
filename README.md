# 📊 Predicción de Abandono de Clientes (Churn) - Telecomunicaciones

## 📋 Descripción del Proyecto

Sistema de machine learning para predecir qué clientes tienen mayor probabilidad de abandonar un servicio de telecomunicaciones, utilizando el dataset "Telco Customer Churn".

## 🎯 Objetivos

- **Principal**: Identificar clientes con alta probabilidad de churn
- **Métrica Clave**: Maximizar Recall (≥85%)
- **Impacto**: Reducir tasa de abandono mediante acciones preventivas

## 📁 Estructura del Proyecto

```
churn_prediction/
├── data/
│   ├── raw/                    # Datos originales
│   ├── processed/              # Datos procesados
│   └── external/               # Datos externos
├── notebooks/
│   ├── exploratory/            # Análisis exploratorio
│   └── modeling/               # Desarrollo de modelos
├── src/
│   ├── data/                   # Scripts de datos
│   ├── features/               # Feature engineering
│   ├── models/                 # Modelos ML
│   └── visualization/          # Visualizaciones
├── models/
│   ├── trained/                # Modelos entrenados
│   └── pipelines/              # Pipelines ML
├── reports/
│   ├── figures/                # Gráficos y visualizaciones
│   └── tables/                 # Tablas de resultados
├── config/                     # Archivos de configuración
├── requirements.txt            # Dependencias Python
└── README.md                   # Este archivo
```

## 🚀 Instalación y Setup

1. **Clonar/Descargar el proyecto**
2. **Crear entorno virtual**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```
3. **Instalar dependencias**:
   ```bash
   pip install -r requirements.txt
   ```

## 📊 Dataset

**Fuente**: Telco Customer Churn Dataset (Kaggle)
- **Registros**: ~7,000 clientes
- **Variables**: 21 features
- **Target**: Churn (Yes/No)

## 🔄 Pipeline del Proyecto

1. **Exploración de Datos** → notebooks/exploratory/
2. **Limpieza y Preprocessing** → src/data/
3. **Feature Engineering** → src/features/
4. **Modelado ML** → src/models/
5. **Evaluación** → notebooks/modeling/
6. **Deployment** → models/

## 📈 Modelos Implementados

- **Regresión Logística** (Baseline interpretable)
- **Random Forest** (Robusto y explicable)
- **XGBoost** (Alta performance)
- **Ensemble Methods** (Combinación optimizada)

## 🎯 Métricas de Evaluación

- **Recall**: ≥85% (Prioridad máxima)
- **Precision**: ≥40%
- **AUC-ROC**: ≥0.80
- **F1-Score**: Balanceado

## 📊 Resultados Esperados

- Identificación temprana de clientes en riesgo
- Insights sobre factores de churn
- Recomendaciones estratégicas de retención
- ROI medible en acciones preventivas

## 🔧 Uso del Sistema

```python
# Cargar modelo entrenado
from src.models import ChurnPredictor

predictor = ChurnPredictor.load('models/trained/best_model.pkl')

# Predecir churn para nuevo cliente
prediction = predictor.predict(customer_data)
churn_probability = predictor.predict_proba(customer_data)
```

## 👥 Contribución

1. Fork el proyecto
2. Crear branch de feature
3. Commit cambios
4. Push al branch
5. Abrir Pull Request

## 📝 Licencia

MIT License - ver LICENSE file para detalles

## 📞 Contacto

**Autor**: [Diego Letelier]
**Email**: [dleteliersr@gmail.com]

---
*Proyecto desarrollado como parte del portafolio de Data Science*
