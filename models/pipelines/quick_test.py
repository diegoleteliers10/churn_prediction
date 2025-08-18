#!/usr/bin/env python3
"""
Test Rápido de Predicción de Churn
===================================

Script súper simple para probar rápidamente la predicción de churn
de un cliente y ver el resultado en consola.

Autor: Diego Letelier
Versión: 1.0.0
"""

import sys
from pathlib import Path

# Agregar el directorio actual al path
sys.path.append(str(Path(__file__).parent))

try:
    from pipeline_script import ChurnPredictionPipeline
    print("✅ Pipeline importado correctamente")
except ImportError as e:
    print(f"❌ Error importando pipeline: {e}")
    sys.exit(1)

def main():
    """Test rápido con cliente predefinido"""
    
    print("🚀 TEST RÁPIDO DE PREDICCIÓN DE CHURN")
    print("=" * 50)
    
    try:
        # Inicializar pipeline
        print("🔄 Inicializando pipeline...")
        pipeline = ChurnPredictionPipeline()
        print("✅ Pipeline listo!")
        
        # Cliente de prueba (alto riesgo de churn)
        test_customer = {
            'customerID': 'TEST001',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 2,  # Cliente nuevo
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',  # Mayor predictor de churn
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 85.0,
            'TotalCharges': 170.0
        }
        
        # Hacer predicción
        print("🔄 Prediciendo...")
        result = pipeline.predict_churn(test_customer)
        
        # Mostrar resultado
        print("\n" + "=" * 50)
        print("🎯 RESULTADO:")
        print("=" * 50)
        
        # Resultado principal
        if result['will_churn']:
            print("🔴 CHURN DETECTADO!")
        else:
            print("🟢 NO CHURN")
        
        print(f"📊 Probabilidad: {result['churn_probability']:.2%}")
        print(f"⚠️  Nivel de Riesgo: {result['risk_level']}")
        print(f"🎯 Confianza: {result['confidence']}")
        
        # Interpretación simple
        print("\n💡 INTERPRETACIÓN:")
        if result['will_churn']:
            print("❌ Este cliente probablemente abandonará el servicio")
            print("   Recomendación: Intervención urgente")
        else:
            print("✅ Este cliente probablemente se quedará")
            print("   Recomendación: Mantener satisfecho")
        
        print("\n✅ Test completado!")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")

if __name__ == "__main__":
    main()
