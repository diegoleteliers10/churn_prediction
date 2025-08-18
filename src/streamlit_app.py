#!/usr/bin/env python3
"""
Aplicación Streamlit para Predicción de Churn - Telecomunicaciones
================================================================

Interfaz web completa para predecir la probabilidad de churn de clientes
utilizando el modelo XGBoost entrenado.

Autor: Diego Letelier
Versión: 1.0.0
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path

# Importar utilidades
from utils import (
    setup_logging, ChurnConfig, load_model_and_encoders,
    preprocess_customer_data, get_risk_level, get_confidence
)

# ============================================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================================

st.set_page_config(
    page_title="Churn Prediction - Telecomunicaciones",
    page_icon="��",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# INICIALIZACIÓN
# ============================================================================

@st.cache_resource
def initialize_app():
    """Inicializar la aplicación y cargar recursos"""
    logger = setup_logging()
    config = ChurnConfig()
    model, encoders = load_model_and_encoders()
    
    return logger, config, model, encoders

# Inicializar
logger, config, model, encoders = initialize_app()

# ============================================================================
# HEADER Y SIDEBAR
# ============================================================================

# Header principal
st.title("🚀 Predicción de Churn - Telecomunicaciones")
st.markdown("---")

# Sidebar con información del proyecto
with st.sidebar:
    st.header("📋 Información del Proyecto")
    
    if model is not None:
        st.success("✅ Modelo cargado correctamente")
        st.info(f"**Tipo:** {type(model).__name__}")
    else:
        st.error("❌ Modelo no disponible")
    
    if encoders is not None:
        st.success("✅ Encoders cargados")
        st.info(f"**Variables:** {len(encoders)}")
    else:
        st.error("❌ Encoders no disponibles")
    
    st.markdown("---")
    
    st.header("🎯 Métricas Objetivo")
    target_metrics = config.target_metrics
    st.metric("Recall Objetivo", f"{target_metrics['recall']:.1%}")
    st.metric("Precision Objetivo", f"{target_metrics['precision']:.1%}")
    st.metric("AUC Objetivo", f"{target_metrics['auc']:.2f}")
    
    st.markdown("---")
    
    st.header("�� Métricas del Modelo")
    st.metric("AUC Actual", "0.752")
    st.metric("Recall Actual", "76.74%")
    st.metric("Precision Actual", "51.34%")
    
    st.markdown("---")
    
    st.header("🔧 Configuración")
    threshold = st.slider(
        "Threshold de Clasificación",
        min_value=0.0,
        max_value=1.0,
        value=0.4,
        step=0.05,
        help="Probabilidad mínima para clasificar como churn"
    )

# ============================================================================
# PESTAÑAS PRINCIPALES
# ============================================================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎯 Predicción Individual", 
    "�� Predicción en Lote", 
    "�� Análisis del Modelo",
    "�� Cargar Datos"
])

# ============================================================================
# PESTAÑA 1: PREDICCIÓN INDIVIDUAL
# ============================================================================

with tab1:
    st.header("🎯 Predicción Individual de Churn")
    
    if model is None or encoders is None:
        st.error("❌ No se puede realizar predicciones. Verifica que el modelo esté cargado.")
    else:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("📝 Datos del Cliente")
            
            # Formulario de entrada
            with st.form("customer_form"):
                # Información básica
                st.markdown("**Información Básica**")
                customer_id = st.text_input("ID del Cliente", value="CLIENT001")
                gender = st.selectbox("Género", ["Male", "Female"])
                senior_citizen = st.selectbox("Ciudadano Mayor", [0, 1])
                partner = st.selectbox("Pareja", ["Yes", "No"])
                dependents = st.selectbox("Dependientes", ["Yes", "No"])
                
                # Servicios
                st.markdown("**Servicios**")
                tenure = st.number_input("Antigüedad (meses)", min_value=0, max_value=120, value=5)
                phone_service = st.selectbox("Servicio Telefónico", ["Yes", "No"])
                multiple_lines = st.selectbox("Múltiples Líneas", ["Yes", "No", "No phone service"])
                internet_service = st.selectbox("Servicio Internet", ["DSL", "Fiber optic", "No"])
                
                # Seguridad y soporte
                st.markdown("**Seguridad y Soporte**")
                online_security = st.selectbox("Seguridad Online", ["Yes", "No", "No internet service"])
                online_backup = st.selectbox("Backup Online", ["Yes", "No", "No internet service"])
                device_protection = st.selectbox("Protección de Dispositivo", ["Yes", "No", "No internet service"])
                tech_support = st.selectbox("Soporte Técnico", ["Yes", "No", "No internet service"])
                
                # Streaming
                st.markdown("**Streaming**")
                streaming_tv = st.selectbox("TV Streaming", ["Yes", "No", "No internet service"])
                streaming_movies = st.selectbox("Películas Streaming", ["Yes", "No", "No internet service"])
                
                # Contrato y facturación
                st.markdown("**Contrato y Facturación**")
                contract = st.selectbox("Contrato", ["Month-to-month", "One year", "Two year"])
                paperless_billing = st.selectbox("Facturación Sin Papel", ["Yes", "No"])
                payment_method = st.selectbox("Método de Pago", [
                    "Electronic check", "Mailed check", 
                    "Bank transfer (automatic)", "Credit card (automatic)"
                ])
                
                # Cargos
                st.markdown("**Cargos**")
                monthly_charges = st.number_input("Cargos Mensuales ($)", min_value=0.0, max_value=200.0, value=55.0)
                total_charges = st.number_input("Cargos Totales ($)", min_value=0.0, max_value=10000.0, value=275.0)
                
                submitted = st.form_submit_button("🚀 Predecir Churn")
        
        with col2:
            st.subheader("📊 Resultado de la Predicción")
            
            if submitted:
                try:
                    # Crear datos del cliente
                    customer_data = {
                        'customerID': customer_id,
                        'gender': gender,
                        'SeniorCitizen': senior_citizen,
                        'Partner': partner,
                        'Dependents': dependents,
                        'tenure': tenure,
                        'PhoneService': phone_service,
                        'MultipleLines': multiple_lines,
                        'InternetService': internet_service,
                        'OnlineSecurity': online_security,
                        'OnlineBackup': online_backup,
                        'DeviceProtection': device_protection,
                        'TechSupport': tech_support,
                        'StreamingTV': streaming_tv,
                        'StreamingMovies': streaming_movies,
                        'Contract': contract,
                        'PaperlessBilling': paperless_billing,
                        'PaymentMethod': payment_method,
                        'MonthlyCharges': monthly_charges,
                        'TotalCharges': total_charges
                    }
                    
                    # Preprocesar datos
                    processed_data = preprocess_customer_data(customer_data, encoders)
                    
                    # Realizar predicción
                    probability = model.predict_proba(processed_data)[0][1]
                    will_churn = probability >= threshold
                    
                    # Mostrar resultado
                    st.markdown("---")
                    
                    # Probabilidad
                    st.metric(
                        "Probabilidad de Churn",
                        f"{probability:.1%}",
                        delta=f"{'ALTO' if will_churn else 'BAJO'} riesgo"
                    )
                    
                    # Clasificación
                    risk_level = get_risk_level(probability)
                    confidence = get_confidence(probability)
                    
                    st.info(f"**Nivel de Riesgo:** {risk_level}")
                    st.info(f"**Confianza:** {confidence}")
                    
                    # Recomendación
                    if will_churn:
                        st.warning("⚠️ **RECOMENDACIÓN:** Cliente en riesgo de churn. Intervención urgente requerida.")
                    else:
                        st.success("✅ **RECOMENDACIÓN:** Cliente estable. Monitoreo rutinario.")
                    
                    # Gráfico de probabilidad
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=probability * 100,
                        domain={'x': [0, 1], 'y': [0, 1]},
                        title={'text': "Probabilidad de Churn (%)"},
                        delta={'reference': threshold * 100},
                        gauge={
                            'axis': {'range': [None, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 40], 'color': "lightgreen"},
                                {'range': [40, 70], 'color': "yellow"},
                                {'range': [70, 100], 'color': "red"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': threshold * 100
                            }
                        }
                    ))
                    
                    fig.update_layout(height=300)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Información adicional
                    with st.expander("📋 Detalles de la Predicción"):
                        st.json({
                            "customer_id": customer_id,
                            "churn_probability": round(probability, 4),
                            "will_churn": will_churn,
                            "threshold_used": threshold,
                            "prediction_timestamp": datetime.now().isoformat(),
                            "features_processed": len(processed_data.columns)
                        })
                
                except Exception as e:
                    st.error(f"❌ Error en la predicción: {e}")
                    logger.error(f"Error en predicción individual: {e}")

# ============================================================================
# PESTAÑA 2: PREDICCIÓN EN LOTE
# ============================================================================

with tab2:
    st.header("�� Predicción en Lote")
    
    if model is None or encoders is None:
        st.error("❌ No se puede realizar predicciones. Verifica que el modelo esté cargado.")
    else:
        st.info("💡 Sube un archivo CSV con datos de múltiples clientes para predicción en lote.")
        
        uploaded_file = st.file_uploader(
            "📁 Subir archivo CSV",
            type=['csv'],
            help="El archivo debe contener las columnas requeridas sin la columna 'Churn'"
        )
        
        if uploaded_file is not None:
            try:
                # Leer archivo
                df = pd.read_csv(uploaded_file)
                st.success(f"✅ Archivo cargado: {len(df)} clientes")
                
                # Mostrar preview
                st.subheader("📋 Vista previa de los datos")
                st.dataframe(df.head())
                
                # Verificar columnas requeridas
                required_columns = [
                    'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
                    'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
                    'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
                    'MonthlyCharges', 'TotalCharges'
                ]
                
                missing_columns = [col for col in required_columns if col not in df.columns]
                
                if missing_columns:
                    st.error(f"❌ Columnas faltantes: {missing_columns}")
                else:
                    st.success("✅ Todas las columnas requeridas están presentes")
                    
                    # Botón para predicción
                    if st.button("🚀 Realizar Predicciones en Lote"):
                        with st.spinner("Procesando predicciones..."):
                            predictions = []
                            
                            for idx, row in df.iterrows():
                                try:
                                    # Convertir fila a diccionario
                                    customer_data = row.to_dict()
                                    
                                    # Preprocesar
                                    processed_data = preprocess_customer_data(customer_data, encoders)
                                    
                                    # Predicción
                                    probability = model.predict_proba(processed_data)[0][1]
                                    will_churn = probability >= threshold
                                    
                                    predictions.append({
                                        'customer_id': customer_data.get('customerID', f'CLIENT_{idx}'),
                                        'churn_probability': round(probability, 4),
                                        'will_churn': will_churn,
                                        'risk_level': get_risk_level(probability),
                                        'threshold_used': threshold
                                    })
                                    
                                except Exception as e:
                                    st.warning(f"⚠️ Error en cliente {idx}: {e}")
                            
                            if predictions:
                                # Crear DataFrame de resultados
                                results_df = pd.DataFrame(predictions)
                                
                                # Mostrar resultados
                                st.subheader("📊 Resultados de las Predicciones")
                                st.dataframe(results_df)
                                
                                # Estadísticas
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    st.metric("Total Clientes", len(results_df))
                                
                                with col2:
                                    churn_count = results_df['will_churn'].sum()
                                    st.metric("Predicción Churn", churn_count)
                                
                                with col3:
                                    churn_rate = churn_count / len(results_df)
                                    st.metric("Tasa de Churn", f"{churn_rate:.1%}")
                                
                                with col4:
                                    avg_prob = results_df['churn_probability'].mean()
                                    st.metric("Prob. Promedio", f"{avg_prob:.1%}")
                                
                                # Gráfico de distribución
                                fig = px.histogram(
                                    results_df, 
                                    x='churn_probability',
                                    nbins=20,
                                    title="Distribución de Probabilidades de Churn",
                                    labels={'churn_probability': 'Probabilidad de Churn', 'count': 'Número de Clientes'}
                                )
                                fig.add_vline(x=threshold, line_dash="dash", line_color="red", annotation_text=f"Threshold: {threshold}")
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Descargar resultados
                                csv = results_df.to_csv(index=False)
                                st.download_button(
                                    label="📥 Descargar Resultados CSV",
                                    data=csv,
                                    file_name=f"churn_predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
                
            except Exception as e:
                st.error(f"❌ Error al procesar archivo: {e}")

# ============================================================================
# PESTAÑA 3: ANÁLISIS DEL MODELO
# ============================================================================

with tab3:
    st.header("�� Análisis del Modelo")
    
    if model is None:
        st.error("❌ Modelo no disponible para análisis.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Performance del Modelo")
            
            # Métricas actuales vs objetivo
            metrics_data = {
                'Métrica': ['AUC', 'Recall', 'Precision', 'F1-Score'],
                'Actual': [0.752, 0.7674, 0.5134, 0.6152],
                'Objetivo': [0.80, 0.85, 0.60, 0.70]
            }
            
            metrics_df = pd.DataFrame(metrics_data)
            metrics_df['Diferencia'] = metrics_df['Actual'] - metrics_df['Objetivo']
            
            st.dataframe(metrics_df, use_container_width=True)
            
            # Gráfico de comparación
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                name='Actual',
                x=metrics_df['Métrica'],
                y=metrics_df['Actual'],
                marker_color='lightblue'
            ))
            
            fig.add_trace(go.Bar(
                name='Objetivo',
                x=metrics_df['Métrica'],
                y=metrics_df['Objetivo'],
                marker_color='orange'
            ))
            
            fig.update_layout(
                title="Métricas Actuales vs Objetivo",
                barmode='group',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🎯 Análisis de Threshold")
            
            # Simular diferentes thresholds
            thresholds = np.arange(0.1, 1.0, 0.05)
            recall_values = []
            precision_values = []
            f1_values = []
            
            for thresh in thresholds:
                # Aquí podrías usar métricas reales del modelo
                # Por ahora usamos valores simulados
                recall = 0.9 - 0.3 * thresh  # Simulado
                precision = 0.3 + 0.4 * thresh  # Simulado
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                recall_values.append(recall)
                precision_values.append(precision)
                f1_values.append(f1)
            
            # Gráfico de threshold
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=thresholds,
                y=recall_values,
                mode='lines+markers',
                name='Recall',
                line=dict(color='blue')
            ))
            
            fig.add_trace(go.Scatter(
                x=thresholds,
                y=precision_values,
                mode='lines+markers',
                name='Precision',
                line=dict(color='red')
            ))
            
            fig.add_trace(go.Scatter(
                x=thresholds,
                y=f1_values,
                mode='lines+markers',
                name='F1-Score',
                line=dict(color='green')
            ))
            
            fig.add_vline(x=threshold, line_dash="dash", line_color="black", annotation_text=f"Threshold actual: {threshold}")
            
            fig.update_layout(
                title="Métricas por Threshold",
                xaxis_title="Threshold",
                yaxis_title="Valor",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Información del modelo
        st.subheader("🔍 Información del Modelo")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"**Tipo de Modelo:** {type(model).__name__}")
            st.info(f"**Features Utilizadas:** {len(encoders) + 2 if encoders else 'N/A'}")  # +2 por tenure_group y total_services
            st.info(f"**Threshold Actual:** {threshold}")
        
        with col2:
            st.info(f"**Fecha de Entrenamiento:** 2024-01-15")
            st.info(f"**Dataset de Entrenamiento:** Telco Customer Churn")
            st.info(f"**Tamaño de Test:** {config.test_size:.1%}")

# ============================================================================
# PESTAÑA 4: CARGAR DATOS
# ============================================================================

with tab4:
    st.header("📁 Cargar y Analizar Datos")
    
    st.info("💡 Sube un archivo CSV para analizar la distribución de variables.")
    
    uploaded_data = st.file_uploader(
        "📁 Subir archivo CSV para análisis",
        type=['csv'],
        key="data_analysis"
    )
    
    if uploaded_data is not None:
        try:
            df_analysis = pd.read_csv(uploaded_data)
            st.success(f"✅ Datos cargados: {len(df_analysis)} registros, {len(df_analysis.columns)} columnas")
            
            # Información básica
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📋 Información del Dataset")
                st.write(f"**Forma:** {df_analysis.shape}")
                st.write(f"**Tipos de datos:**")
                st.dataframe(df_analysis.dtypes.to_frame('Tipo'))
            
            with col2:
                st.subheader("📊 Estadísticas Descriptivas")
                st.dataframe(df_analysis.describe())
            
            # Análisis de variables categóricas
            st.subheader("📈 Análisis de Variables Categóricas")
            
            categorical_cols = df_analysis.select_dtypes(include=['object']).columns
            
            if len(categorical_cols) > 0:
                for col in categorical_cols[:6]:  # Mostrar solo las primeras 6
                    if col in df_analysis.columns:
                        fig = px.bar(
                            df_analysis[col].value_counts(),
                            title=f"Distribución de {col}",
                            labels={'value': 'Frecuencia', 'index': col}
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Análisis de variables numéricas
            st.subheader("📊 Análisis de Variables Numéricas")
            
            numeric_cols = df_analysis.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) > 0:
                for col in numeric_cols[:4]:  # Mostrar solo las primeras 4
                    if col in df_analysis.columns:
                        fig = px.histogram(
                            df_analysis, 
                            x=col,
                            title=f"Distribución de {col}",
                            nbins=20
                        )
                        st.plotly_chart(fig, use_container_width=True)
            
            # Correlaciones (si hay suficientes variables numéricas)
            if len(numeric_cols) > 2:
                st.subheader("🔗 Matriz de Correlaciones")
                corr_matrix = df_analysis[numeric_cols].corr()
                
                fig = px.imshow(
                    corr_matrix,
                    title="Matriz de Correlaciones",
                    color_continuous_scale='RdBu',
                    aspect="auto"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        except Exception as e:
            st.error(f"❌ Error al analizar datos: {e}")

# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>🚀 Churn Prediction App - Telecomunicaciones | Versión 1.0.0</p>
        <p>Desarrollado por Diego Letelier | 2024</p>
    </div>
    """,
    unsafe_allow_html=True
)