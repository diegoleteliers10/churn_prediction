"""
Utilidades generales para el proyecto de Churn Prediction
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
import yaml

def setup_logging(level='INFO'):
    """Configurar logging del proyecto"""
    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def load_config(config_path='config/config.yaml'):
    """Cargar configuración desde archivo YAML"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_dataframe(df, filename, path='data/processed/'):
    """Guardar DataFrame con logging"""
    logger = logging.getLogger(__name__)
    filepath = Path(path) / filename
    df.to_csv(filepath, index=False)
    logger.info(f"DataFrame guardado: {filepath} - Shape: {df.shape}")

def load_dataframe(filename, path='data/processed/'):
    """Cargar DataFrame con logging"""
    logger = logging.getLogger(__name__)
    filepath = Path(path) / filename
    df = pd.read_csv(filepath)
    logger.info(f"DataFrame cargado: {filepath} - Shape: {df.shape}")
    return df

class ChurnConfig:
    """Clase para manejar configuración del proyecto"""
    
    def __init__(self, config_path='config/config.yaml'):
        self.config = load_config(config_path)
        
    @property
    def target_column(self):
        return self.config['model']['target_column']
    
    @property
    def test_size(self):
        return self.config['model']['test_size']
    
    @property
    def random_state(self):
        return self.config['model']['random_state']
    
    @property
    def target_metrics(self):
        return {
            'recall': self.config['model']['target_recall'],
            'precision': self.config['model']['target_precision'], 
            'auc': self.config['model']['target_auc']
        }
