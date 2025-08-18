#!/usr/bin/env python3
"""
Script de Prueba para el Pipeline de Predicción de Churn
=======================================================

Este script prueba todas las funcionalidades del pipeline para asegurar
que funcione correctamente antes de usarlo en producción.

Autor: Diego Letelier
Versión: 1.0.0
"""

import sys
import json
from pathlib import Path

# Agregar el directorio actual al path para importar el pipeline
sys.path.append(str(Path(__file__).parent))

try:
    from pipeline_script import ChurnPredictionPipeline
    print("✅ Pipeline importado correctamente")
except ImportError as e:
    print(f"❌ Error importando pipeline: {e}")
    sys.exit(1)

def test_pipeline_initialization():
    """Probar la inicialización del pipeline"""
    print("\n🧪 Probando inicialización del pipeline...")
    
    try:
        pipeline = ChurnPredictionPipeline()
        print("✅ Pipeline inicializado correctamente")
        print(f"   Modelo: {type(pipeline.model).__name__}")
        print(f"   Encoders: {len(pipeline.label_encoders)} disponibles")
        return pipeline
    except Exception as e:
        print(f"❌ Error inicializando pipeline: {e}")
        return None

def test_single_customer_prediction(pipeline):
    """Probar predicción de un cliente individual"""
    print("\n🧪 Probando predicción de cliente individual...")
    
    # Cliente de prueba con alta probabilidad de churn
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
    
    try:
        result = pipeline.predict_churn(test_customer)
        print("✅ Predicción individual exitosa")
        print(f"   Cliente: {result['customer_id']}")
        print(f"   Probabilidad: {result['churn_probability']:.2%}")
        print(f"   Predicción: {'CHURN' if result['will_churn'] else 'NO CHURN'}")
        print(f"   Riesgo: {result['risk_level']}")
        print(f"   Confianza: {result['confidence']}")
        return result
    except Exception as e:
        print(f"❌ Error en predicción individual: {e}")
        return None

def test_batch_prediction(pipeline):
    """Probar predicción en lote"""
    print("\n🧪 Probando predicción en lote...")
    
    # Múltiples clientes de prueba
    test_customers = [
        {
            'customerID': 'TEST001',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 2,
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',
            'MonthlyCharges': 85.0,
            'TotalCharges': 170.0
        },
        {
            'customerID': 'TEST002',
            'gender': 'Female',
            'SeniorCitizen': 0,
            'Partner': 'Yes',
            'Dependents': 'Yes',
            'tenure': 36,
            'PhoneService': 'Yes',
            'MultipleLines': 'Yes',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'Yes',
            'OnlineBackup': 'Yes',
            'DeviceProtection': 'Yes',
            'TechSupport': 'Yes',
            'StreamingTV': 'Yes',
            'StreamingMovies': 'Yes',
            'Contract': 'Two year',
            'PaperlessBilling': 'No',
            'PaymentMethod': 'Credit card (automatic)',
            'MonthlyCharges': 95.0,
            'TotalCharges': 3420.0
        }
    ]
    
    try:
        results = pipeline.predict_batch(test_customers)
        print("✅ Predicción en lote exitosa")
        print(f"   Clientes procesados: {len(results)}")
        
        for i, result in enumerate(results):
            if 'error' not in result:
                print(f"   Cliente {i+1}: {result['customer_id']} - {result['churn_probability']:.2%}")
            else:
                print(f"   Cliente {i+1}: ERROR - {result['error']}")
        
        return results
    except Exception as e:
        print(f"❌ Error en predicción en lote: {e}")
        return None

def test_different_thresholds(pipeline):
    """Probar diferentes thresholds"""
    print("\n🧪 Probando diferentes thresholds...")
    
    test_customer = {
        'customerID': 'TEST003',
        'gender': 'Female',
        'SeniorCitizen': 0,
        'Partner': 'No',
        'Dependents': 'No',
        'tenure': 8,
        'PhoneService': 'Yes',
        'MultipleLines': 'No',
        'InternetService': 'DSL',
        'OnlineSecurity': 'No',
        'OnlineBackup': 'No',
        'DeviceProtection': 'No',
        'TechSupport': 'No',
        'StreamingTV': 'No',
        'StreamingMovies': 'No',
        'Contract': 'Month-to-month',
        'PaperlessBilling': 'Yes',
        'PaymentMethod': 'Electronic check',
        'MonthlyCharges': 65.0,
        'TotalCharges': 520.0
    }
    
    thresholds = [0.3, 0.4, 0.5, 0.6]
    
    try:
        for threshold in thresholds:
            result = pipeline.predict_churn(test_customer, threshold)
            print(f"   Threshold {threshold}: {'CHURN' if result['will_churn'] else 'NO CHURN'} "
                  f"(Prob: {result['churn_probability']:.2%})")
        
        print("✅ Prueba de thresholds exitosa")
        return True
    except Exception as e:
        print(f"❌ Error probando thresholds: {e}")
        return False

def test_file_operations(pipeline):
    """Probar operaciones de archivo"""
    print("\n🧪 Probando operaciones de archivo...")
    
    # Crear archivo CSV de prueba
    test_csv = Path("test_customer.csv")
    test_data = """customerID,gender,SeniorCitizen,Partner,Dependents,tenure,PhoneService,MultipleLines,InternetService,OnlineSecurity,OnlineBackup,DeviceProtection,TechSupport,StreamingTV,StreamingMovies,Contract,PaperlessBilling,PaymentMethod,MonthlyCharges,TotalCharges
TEST004,Male,0,No,No,15,Yes,Yes,DSL,Yes,No,No,No,Yes,No,One year,Yes,Bank transfer (automatic),55.0,825.0"""
    
    try:
        with open(test_csv, 'w') as f:
            f.write(test_data)
        
        # Cargar datos
        customer_data = pipeline.load_customer_data(test_csv)
        print("✅ Carga de archivo CSV exitosa")
        
        # Hacer predicción
        result = pipeline.predict_churn(customer_data.iloc[0].to_dict())
        print(f"   Predicción: {result['churn_probability']:.2%}")
        
        # Guardar resultados
        output_path = pipeline.save_results(result, "test_results.json")
        print(f"   Resultados guardados en: {output_path}")
        
        # Limpiar archivos de prueba
        test_csv.unlink()
        Path(output_path).unlink()
        
        print("✅ Operaciones de archivo exitosas")
        return True
        
    except Exception as e:
        print(f"❌ Error en operaciones de archivo: {e}")
        # Limpiar en caso de error
        if test_csv.exists():
            test_csv.unlink()
        return False

def main():
    """Función principal de pruebas"""
    print("🚀 INICIANDO PRUEBAS DEL PIPELINE DE CHURN PREDICTION")
    print("=" * 60)
    
    # Contador de pruebas
    total_tests = 5
    passed_tests = 0
    
    # 1. Prueba de inicialización
    pipeline = test_pipeline_initialization()
    if pipeline:
        passed_tests += 1
    
    if not pipeline:
        print("\n❌ No se puede continuar sin pipeline inicializado")
        return
    
    # 2. Prueba de predicción individual
    if test_single_customer_prediction(pipeline):
        passed_tests += 1
    
    # 3. Prueba de predicción en lote
    if test_batch_prediction(pipeline):
        passed_tests += 1
    
    # 4. Prueba de diferentes thresholds
    if test_different_thresholds(pipeline):
        passed_tests += 1
    
    # 5. Prueba de operaciones de archivo
    if test_file_operations(pipeline):
        passed_tests += 1
    
    # Resumen final
    print("\n" + "=" * 60)
    print("📊 RESUMEN DE PRUEBAS")
    print("=" * 60)
    print(f"Pruebas totales: {total_tests}")
    print(f"Pruebas exitosas: {passed_tests}")
    print(f"Pruebas fallidas: {total_tests - passed_tests}")
    
    if passed_tests == total_tests:
        print("\n🎉 ¡TODAS LAS PRUEBAS EXITOSAS!")
        print("✅ El pipeline está listo para usar en producción")
        return 0
    else:
        print(f"\n⚠️ {total_tests - passed_tests} prueba(s) fallida(s)")
        print("❌ Revisar errores antes de usar en producción")
        return 1

if __name__ == "__main__":
    sys.exit(main())
