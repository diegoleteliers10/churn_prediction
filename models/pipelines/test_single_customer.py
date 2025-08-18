#!/usr/bin/env python3
"""
Test Simple para Predicción de Churn - Cliente Individual
========================================================

Script simple para probar la predicción de churn de un solo cliente
y ver la respuesta del modelo en la consola.

Autor: Diego Letelier
Versión: 1.0.0
"""

import sys
from pathlib import Path

# Agregar el directorio actual al path para importar el pipeline
sys.path.append(str(Path(__file__).parent))

try:
    from pipeline_script import ChurnPredictionPipeline
    print("✅ Pipeline importado correctamente")
except ImportError as e:
    print(f"❌ Error importando pipeline: {e}")
    sys.exit(1)

def test_single_customer():
    """Probar predicción de un solo cliente"""
    
    print("🚀 INICIANDO TEST DE CLIENTE INDIVIDUAL")
    print("=" * 50)
    
    try:
        # Inicializar pipeline
        print("🔄 Inicializando pipeline...")
        pipeline = ChurnPredictionPipeline()
        print("✅ Pipeline inicializado")
        
        # Cliente de prueba con alta probabilidad de churn
        test_customer = {
            'customerID': 'TEST001',
            'gender': 'Male',
            'SeniorCitizen': 0,
            'Partner': 'No',
            'Dependents': 'No',
            'tenure': 2,  # Cliente nuevo (alta probabilidad de churn)
            'PhoneService': 'Yes',
            'MultipleLines': 'No',
            'InternetService': 'Fiber optic',
            'OnlineSecurity': 'No',  # Sin servicios adicionales
            'OnlineBackup': 'No',
            'DeviceProtection': 'No',
            'TechSupport': 'No',
            'StreamingTV': 'No',
            'StreamingMovies': 'No',
            'Contract': 'Month-to-month',  # ¡El mayor predictor de churn!
            'PaperlessBilling': 'Yes',
            'PaymentMethod': 'Electronic check',  # Método con más churn
            'MonthlyCharges': 85.0,  # Alto para pocos servicios
            'TotalCharges': 170.0
        }
        
        print("\n📊 DATOS DEL CLIENTE DE PRUEBA:")
        print("-" * 30)
        for key, value in test_customer.items():
            print(f"{key:20}: {value}")
        
        # Hacer predicción
        print("\n🔄 Realizando predicción...")
        result = pipeline.predict_churn(test_customer)
        
        # Mostrar resultados
        print("\n" + "=" * 50)
        print("📊 RESULTADO DE LA PREDICCIÓN")
        print("=" * 50)
        print(f"Cliente ID: {result['customer_id']}")
        print(f"Probabilidad de Churn: {result['churn_probability']:.2%}")
        print(f"Predicción: {'🔴 CHURN' if result['will_churn'] else '🟢 NO CHURN'}")
        print(f"Nivel de Riesgo: {result['risk_level']}")
        print(f"Confianza: {result['confidence']}")
        print(f"Threshold usado: {result['threshold_used']}")
        print(f"Timestamp: {result['prediction_timestamp']}")
        
        # Interpretación
        print("\n💡 INTERPRETACIÓN:")
        print("-" * 20)
        if result['will_churn']:
            print("🔴 Este cliente tiene ALTO RIESGO de abandonar el servicio")
            if result['churn_probability'] >= 0.7:
                print("   ⚠️  Probabilidad muy alta - Acción inmediata requerida")
            elif result['churn_probability'] >= 0.5:
                print("   ⚠️  Probabilidad alta - Intervención urgente")
            else:
                print("   ⚠️  Probabilidad moderada - Monitoreo cercano")
        else:
            print("🟢 Este cliente tiene BAJO RIESGO de abandonar el servicio")
            if result['churn_probability'] <= 0.2:
                print("   ✅ Cliente muy leal - Mantener satisfecho")
            elif result['churn_probability'] <= 0.3:
                print("   ✅ Cliente estable - Continuar servicio actual")
            else:
                print("   ⚠️  Cliente en zona de atención - Monitorear")
        
        # Factores clave
        print("\n🔍 FACTORES CLAVE QUE INFLUYEN:")
        print("-" * 35)
        if test_customer['Contract'] == 'Month-to-month':
            print("❌ Contrato mes a mes (ALTO RIESGO)")
        elif test_customer['Contract'] == 'One year':
            print("⚠️  Contrato de 1 año (RIESGO MEDIO)")
        else:
            print("✅ Contrato de 2 años (BAJO RIESGO)")
        
        if test_customer['tenure'] < 12:
            print("❌ Cliente nuevo (<1 año) - ALTO RIESGO")
        elif test_customer['tenure'] < 24:
            print("⚠️  Cliente establecido (1-2 años) - RIESGO MEDIO")
        else:
            print("✅ Cliente veterano (2+ años) - BAJO RIESGO")
        
        if test_customer['PaymentMethod'] == 'Electronic check':
            print("❌ Pago por cheque electrónico - ALTO RIESGO")
        else:
            print("✅ Método de pago automático - BAJO RIESGO")
        
        print("\n✅ Test completado exitosamente!")
        return result
        
    except Exception as e:
        print(f"❌ Error en el test: {str(e)}")
        print(f"   Detalles: {type(e).__name__}")
        return None

def test_custom_customer():
    """Permitir al usuario ingresar datos personalizados"""
    
    print("\n" + "=" * 50)
    print("🎯 TEST CON DATOS PERSONALIZADOS")
    print("=" * 50)
    
    try:
        # Inicializar pipeline
        pipeline = ChurnPredictionPipeline()
        
        # Solicitar datos del cliente
        print("Ingresa los datos del cliente (presiona Enter para usar valores por defecto):")
        
        customer_data = {}
        
        # Campos con valores por defecto
        defaults = {
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
        }
        
        # Solicitar cada campo
        for field, default_value in defaults.items():
            if field == 'customerID':
                customer_data[field] = 'CUSTOM001'
                continue
                
            user_input = input(f"{field} (default: {default_value}): ").strip()
            if user_input:
                # Convertir tipos
                if field in ['SeniorCitizen', 'tenure']:
                    customer_data[field] = int(user_input)
                elif field in ['MonthlyCharges', 'TotalCharges']:
                    customer_data[field] = float(user_input)
                else:
                    customer_data[field] = user_input
            else:
                customer_data[field] = default_value
        
        # Hacer predicción
        print("\n🔄 Realizando predicción con datos personalizados...")
        result = pipeline.predict_churn(customer_data)
        
        # Mostrar resultados
        print("\n" + "=" * 50)
        print("📊 RESULTADO DE LA PREDICCIÓN PERSONALIZADA")
        print("=" * 50)
        print(f"Cliente ID: {result['customer_id']}")
        print(f"Probabilidad de Churn: {result['churn_probability']:.2%}")
        print(f"Predicción: {'🔴 CHURN' if result['will_churn'] else '🟢 NO CHURN'}")
        print(f"Nivel de Riesgo: {result['risk_level']}")
        print(f"Confianza: {result['confidence']}")
        
        return result
        
    except Exception as e:
        print(f"❌ Error en test personalizado: {str(e)}")
        return None

def main():
    """Función principal"""
    print("🧪 TEST SIMPLE DE PREDICCIÓN DE CHURN")
    print("=" * 50)
    
    while True:
        print("\nSelecciona una opción:")
        print("1. Test con cliente predefinido (alto riesgo)")
        print("2. Test con datos personalizados")
        print("3. Salir")
        
        choice = input("\nOpción (1-3): ").strip()
        
        if choice == '1':
            test_single_customer()
        elif choice == '2':
            test_custom_customer()
        elif choice == '3':
            print("👋 ¡Hasta luego!")
            break
        else:
            print("❌ Opción inválida. Por favor selecciona 1, 2 o 3.")

if __name__ == "__main__":
    main()
