#!/usr/bin/env python3
"""
Utilidades para la aplicación Streamlit de Churn Prediction
"""

import pandas as pd
import numpy as np
import joblib
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
import streamlit as st

def setup_logging():
    """Configurar logging para la aplicación"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Cargar configuración del proyecto"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        st.error(f"❌ Archivo de configuración no encontrado: {config_path}")
        return {}
    except Exception as e:
        st.error(f"❌ Error al cargar configuración: {e}")
        return {}

class ChurnConfig:
    """Clase para manejar configuración del proyecto"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.config = load_config(config_path)
        
    @property
    def target_column(self):
        return self.config.get('model', {}).get('target_column', 'Churn')
    
    @property
    def test_size(self):
        return self.config.get('model', {}).get('test_size', 0.15)
    
    @property
    def random_state(self):
        return self.config.get('model', {}).get('random_state', 42)
    
    @property
    def target_metrics(self):
        return {
            'recall': self.config.get('model', {}).get('target_recall', 0.85),
            'precision': self.config.get('model', {}).get('target_precision', 0.60), 
            'auc': self.config.get('model', {}).get('target_auc', 0.80)
        }

def setup_project_paths():
    """Configurar paths del proyecto"""
    current_path = Path.cwd()
    project_root = current_path
    
    # Buscar la raíz del proyecto
    while not (project_root / "src").exists() and project_root != project_root.parent:
        project_root = project_root.parent
    
    return project_root

def load_model_and_encoders():
    """Cargar modelo y encoders"""
    try:
        project_root = setup_project_paths()
        
        model_path = project_root / "models" / "trained" / "churn_model_xgb.pkl"
        encoders_path = project_root / "models" / "encoders" / "label_encoders.pkl"
        
        if not model_path.exists():
            st.error(f"❌ Modelo no encontrado en: {model_path}")
            return None, None
            
        if not encoders_path.exists():
            st.error(f"❌ Encoders no encontrados en: {encoders_path}")
            return None, None
        
        model = joblib.load(model_path)
        encoders = joblib.load(encoders_path)
        
        return model, encoders
        
    except Exception as e:
        st.error(f"❌ Error al cargar modelo: {e}")
        return None, None

def assign_tenure_group(tenure: int) -> str:
    """Asignar grupo de antigüedad"""
    if tenure < 12:
        return 'Nuevos (<1 año)'
    elif tenure < 24:
        return 'Establecidos (1-2 años)'
    else:
        return 'Veteranos (2+ años)'

def count_services(row: pd.Series) -> int:
    """Contar servicios activos del cliente"""
    count = 0
    if row['PhoneService'] == 'Yes':
        count += 1
    if row['MultipleLines'] == 'Yes':
        count += 1
    if row['InternetService'] in ['DSL', 'Fiber optic']:
        count += 1
    if row['OnlineSecurity'] == 'Yes':
        count += 1
    if row['OnlineBackup'] == 'Yes':
        count += 1
    if row['DeviceProtection'] == 'Yes':
        count += 1
    if row['TechSupport'] == 'Yes':
        count += 1
    if row['StreamingTV'] == 'Yes':
        count += 1
    if row['StreamingMovies'] == 'Yes':
        count += 1
    return count

def preprocess_customer_data(customer_data: Dict[str, Any], encoders: Dict) -> pd.DataFrame:
    """Preprocesar datos del cliente para predicción"""
    # Convertir a DataFrame
    df = pd.DataFrame([customer_data])
    
    # Feature engineering
    df['tenure_group'] = df['tenure'].apply(assign_tenure_group)
    df['total_services'] = df.apply(count_services, axis=1)
    
    # Aplicar label encoders
    for column, encoder in encoders.items():
        if column in df.columns:
            # Manejar valores no vistos
            df[column] = df[column].astype(str)
            df[column] = df[column].map(lambda x: x if x in encoder.classes_ else encoder.classes_[0])
            df[column] = encoder.transform(df[column])
    
    # Features en el orden correcto
    feature_columns = [
        'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
        'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
        'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
        'MonthlyCharges', 'TotalCharges', 'tenure_group', 'total_services'
    ]
    
    return df[feature_columns]

def get_risk_level(probability: float) -> str:
    """Determinar nivel de riesgo"""
    if probability >= 0.7:
        return "🔴 High Risk"
    elif probability >= 0.4:
        return "🟡 Medium Risk"
    else:
        return "�� Low Risk"

def get_confidence(probability: float) -> str:
    """Determinar nivel de confianza"""
    if abs(probability - 0.5) < 0.1:
        return "�� Low"
    elif abs(probability - 0.5) < 0.2:
        return "�� Medium"
    else:
        return "�� High"