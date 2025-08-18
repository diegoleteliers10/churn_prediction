#!/usr/bin/env python3
"""
Pipeline de Predicción de Churn - Telecomunicaciones
====================================================

Script ejecutable para predecir la probabilidad de churn de nuevos clientes
basado en el modelo XGBoost entrenado.

Uso:
    python pipeline_script.py --customer_data customer_data.csv
    python pipeline_script.py --customer_data customer_data.csv --threshold 0.5
    python pipeline_script.py --customer_data customer_data.csv --output results.json

Autor: Diego Letelier
Versión: 1.0.0
"""

import pandas as pd
import numpy as np
import joblib
import json
import argparse
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Union, Any
from datetime import datetime

# Suprimir warnings
warnings.filterwarnings('ignore')

# Configurar paths del proyecto
def setup_project_paths():
    """Configurar paths del proyecto para encontrar modelos y encoders"""
    current_path = Path.cwd()
    
    # Buscar la raíz del proyecto (donde está src/)
    project_root = current_path
    while not (project_root / "src").exists() and project_root != project_root.parent:
        project_root = project_root.parent
    
    return project_root

class ChurnPredictionPipeline:
    """Pipeline completo para predicción de churn"""
    
    def __init__(self, model_path: str = None, encoders_path: str = None):
        """
        Inicializar el pipeline de predicción
        
        Args:
            model_path: Ruta al modelo entrenado
            encoders_path: Ruta a los encoders
        """
        self.project_root = setup_project_paths()
        
        # Rutas por defecto
        if model_path is None:
            model_path = self.project_root / "models" / "trained" / "churn_model_xgb.pkl"
        if encoders_path is None:
            encoders_path = self.project_root / "models" / "encoders" / "label_encoders.pkl"
        
        # Verificar que existan los archivos
        if not Path(model_path).exists():
            raise FileNotFoundError(f"Modelo no encontrado en: {model_path}")
        if not Path(encoders_path).exists():
            raise FileNotFoundError(f"Encoders no encontrados en: {encoders_path}")
        
        # Cargar modelo y encoders
        print("🔄 Cargando modelo y encoders...")
        self.model = joblib.load(model_path)
        self.label_encoders = joblib.load(encoders_path)
        
        print(f"✅ Modelo cargado: {type(self.model).__name__}")
        print(f"✅ Encoders disponibles: {list(self.label_encoders.keys())}")
        
        # Configuración por defecto
        self.default_threshold = 0.4
        
    def assign_tenure_group(self, tenure: int) -> str:
        """Asignar grupo de antigüedad basado en tenure"""
        if tenure < 12:
            return 'Nuevos (<1 año)'
        elif tenure < 24:
            return 'Establecidos (1-2 años)'
        else:
            return 'Veteranos (2+ años)'
    
    def count_services(self, row: pd.Series) -> int:
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
    
    def preprocess_new_customer(self, customer_data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocesar datos de un nuevo cliente para predicción
        
        Args:
            customer_data: Datos del cliente (dict o DataFrame)
            
        Returns:
            DataFrame preprocesado y codificado
        """
        # Convertir a DataFrame si es dict
        if isinstance(customer_data, dict):
            df_new = pd.DataFrame([customer_data])
        else:
            df_new = customer_data.copy()
        
        # 1. Crear tenure_group
        df_new['tenure_group'] = df_new['tenure'].apply(self.assign_tenure_group)
        
        # 2. Crear total_services
        df_new['total_services'] = df_new.apply(self.count_services, axis=1)
        
        # 3. Quitar customerID si existe
        if 'customerID' in df_new.columns:
            X = df_new.drop(['customerID'], axis=1)
        else:
            X = df_new.copy()
        
        # Identificar características categóricas
        categorical_features = X.select_dtypes(include=['object']).columns.tolist()
        
        X_encoded = X.copy()
        
        # Aplicar encoders entrenados
        for col in categorical_features:
            if col in self.label_encoders:
                try:
                    X_encoded[col] = self.label_encoders[col].transform(X[col])
                except ValueError as e:
                    print(f"⚠️ Valor desconocido en {col}: {X[col].iloc[0]}")
                    print(f"   Asignando valor por defecto: 0")
                    # Asignar valor por defecto
                    X_encoded[col] = 0
        
        return X_encoded
    
    def get_risk_level(self, probability: float) -> str:
        """Clasificar nivel de riesgo basado en probabilidad"""
        if probability >= 0.7:
            return 'High Risk'
        elif probability >= 0.4:
            return 'Medium Risk'
        else:
            return 'Low Risk'
    
    def predict_churn(self, customer_data: Union[Dict, pd.DataFrame], 
                     threshold: float = None) -> Dict[str, Any]:
        """
        Predecir probabilidad de churn de un cliente
        
        Args:
            customer_data: Datos del cliente
            threshold: Umbral de decisión (default: 0.4)
            
        Returns:
            Dict con predicción, probabilidad y explicación
        """
        if threshold is None:
            threshold = self.default_threshold
        
        # Preprocesar datos
        X_processed = self.preprocess_new_customer(customer_data)
        
        # Obtener probabilidades
        churn_proba = self.model.predict_proba(X_processed)[0, 1]
        
        # Hacer predicción con threshold
        will_churn = churn_proba >= threshold
        
        # Crear resultado
        result = {
            'customer_id': customer_data.get('customerID', 'Unknown') if isinstance(customer_data, dict) else 'Unknown',
            'churn_probability': float(round(churn_proba, 4)),
            'will_churn': bool(will_churn),
            'risk_level': self.get_risk_level(churn_proba),
            'confidence': 'High' if abs(churn_proba - 0.5) > 0.3 else 'Medium' if abs(churn_proba - 0.5) > 0.15 else 'Low',
            'threshold_used': threshold,
            'prediction_timestamp': datetime.now().isoformat()
        }
        
        return result
    
    def predict_batch(self, customers_data: List[Dict], 
                     threshold: float = None) -> List[Dict[str, Any]]:
        """
        Predecir churn para múltiples clientes
        
        Args:
            customers_data: Lista de datos de clientes
            threshold: Umbral de decisión
            
        Returns:
            Lista de predicciones
        """
        results = []
        
        for i, customer_data in enumerate(customers_data):
            try:
                result = self.predict_churn(customer_data, threshold)
                results.append(result)
                print(f"✅ Cliente {i+1}/{len(customers_data)} procesado: {result['customer_id']}")
            except Exception as e:
                print(f"❌ Error procesando cliente {i+1}: {str(e)}")
                error_result = {
                    'customer_id': customer_data.get('customerID', f'Customer_{i+1}'),
                    'error': str(e),
                    'prediction_timestamp': datetime.now().isoformat()
                }
                results.append(error_result)
        
        return results
    
    def load_customer_data(self, file_path: str) -> Union[Dict, pd.DataFrame]:
        """
        Cargar datos del cliente desde archivo
        
        Args:
            file_path: Ruta al archivo de datos
            
        Returns:
            Datos del cliente
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            raise FileNotFoundError(f"Archivo no encontrado: {file_path}")
        
        # Detectar tipo de archivo
        if file_path.suffix.lower() == '.csv':
            data = pd.read_csv(file_path)
        elif file_path.suffix.lower() in ['.xlsx', '.xls']:
            data = pd.read_excel(file_path)
        elif file_path.suffix.lower() == '.json':
            with open(file_path, 'r') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Formato de archivo no soportado: {file_path.suffix}")
        
        return data
    
    def save_results(self, results: Union[Dict, List[Dict]], 
                    output_path: str = None) -> str:
        """
        Guardar resultados de predicción
        
        Args:
            results: Resultados a guardar
            output_path: Ruta de salida (opcional)
            
        Returns:
            Ruta donde se guardaron los resultados
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"churn_prediction_results_{timestamp}.json"
        
        # Crear directorio si no existe
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Guardar resultados
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Resultados guardados en: {output_path}")
        return str(output_path)

def main():
    """Función principal del script"""
    parser = argparse.ArgumentParser(
        description="Pipeline de Predicción de Churn - Telecomunicaciones",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos de uso:
  # Predicción simple
  python pipeline_script.py --customer_data customer.csv
  
  # Con threshold personalizado
  python pipeline_script.py --customer_data customer.csv --threshold 0.5
  
  # Guardar resultados en archivo específico
  python pipeline_script.py --customer_data customer.csv --output results.json
  
  # Modo batch para múltiples clientes
  python pipeline_script.py --customer_data customers_batch.csv --batch
  
  # Verificar instalación
  python pipeline_script.py --check
        """
    )
    
    parser.add_argument(
        '--customer_data', 
        type=str, 
        help='Ruta al archivo con datos del cliente (CSV, Excel, JSON)'
    )
    
    parser.add_argument(
        '--threshold', 
        type=float, 
        default=0.4,
        help='Umbral de decisión para churn (default: 0.4)'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        help='Ruta de salida para los resultados (opcional)'
    )
    
    parser.add_argument(
        '--batch', 
        action='store_true',
        help='Modo batch para múltiples clientes'
    )
    
    parser.add_argument(
        '--check', 
        action='store_true',
        help='Verificar instalación y cargar modelo'
    )
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        help='Ruta personalizada al modelo entrenado'
    )
    
    parser.add_argument(
        '--encoders_path', 
        type=str, 
        help='Ruta personalizada a los encoders'
    )
    
    args = parser.parse_args()
    
    # Verificar instalación
    if args.check:
        try:
            pipeline = ChurnPredictionPipeline(
                model_path=args.model_path,
                encoders_path=args.encoders_path
            )
            print("✅ Pipeline configurado correctamente")
            print(f"   Modelo: {type(pipeline.model).__name__}")
            print(f"   Encoders: {len(pipeline.label_encoders)} disponibles")
            return
        except Exception as e:
            print(f"❌ Error configurando pipeline: {str(e)}")
            return 1
    
    # Verificar argumentos requeridos
    if not args.customer_data:
        parser.print_help()
        return 1
    
    try:
        # Inicializar pipeline
        print("🚀 Iniciando Pipeline de Predicción de Churn...")
        pipeline = ChurnPredictionPipeline(
            model_path=args.model_path,
            encoders_path=args.encoders_path
        )
        
        # Cargar datos del cliente
        print(f"📁 Cargando datos desde: {args.customer_data}")
        customer_data = pipeline.load_customer_data(args.customer_data)
        
        # Realizar predicción
        if args.batch or (isinstance(customer_data, pd.DataFrame) and len(customer_data) > 1):
            print(f"🔄 Procesando {len(customer_data)} clientes en modo batch...")
            results = pipeline.predict_batch(customer_data.to_dict('records'), args.threshold)
        else:
            print("🔄 Procesando cliente individual...")
            if isinstance(customer_data, pd.DataFrame):
                customer_data = customer_data.iloc[0].to_dict()
            results = pipeline.predict_churn(customer_data, args.threshold)
        
        # Mostrar resultados
        print("\n" + "="*60)
        print("📊 RESULTADOS DE PREDICCIÓN")
        print("="*60)
        
        if isinstance(results, list):
            for i, result in enumerate(results):
                if 'error' not in result:
                    print(f"\nCliente {i+1}: {result['customer_id']}")
                    print(f"   Probabilidad de Churn: {result['churn_probability']:.2%}")
                    print(f"   Predicción: {'CHURN' if result['will_churn'] else 'NO CHURN'}")
                    print(f"   Nivel de Riesgo: {result['risk_level']}")
                    print(f"   Confianza: {result['confidence']}")
                else:
                    print(f"\nCliente {i+1}: {result['customer_id']} - ERROR: {result['error']}")
        else:
            print(f"Cliente: {results['customer_id']}")
            print(f"Probabilidad de Churn: {results['churn_probability']:.2%}")
            print(f"Predicción: {'CHURN' if results['will_churn'] else 'NO CHURN'}")
            print(f"Nivel de Riesgo: {results['risk_level']}")
            print(f"Confianza: {results['confidence']}")
            print(f"Threshold usado: {results['threshold_used']}")
        
        # Guardar resultados
        if args.output:
            output_path = pipeline.save_results(results, args.output)
        else:
            output_path = pipeline.save_results(results)
        
        print(f"\n✅ Predicción completada exitosamente")
        print(f"📁 Resultados guardados en: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Error en el pipeline: {str(e)}")
        print(f"   Detalles: {type(e).__name__}")
        return 1

if __name__ == "__main__":
    sys.exit(main())
