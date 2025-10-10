"""
Dashboard Streamlit Premium pour le monitoring MLOps
Design professionnel avec animations et visualisations avancées
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import os
from dotenv import load_dotenv
import mlflow

load_dotenv()

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Configuration de la page
st.set_page_config(
    page_title="Churn MLOps Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS Premium
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    
    /* Global Styles */
    * {
        font-family: 'Inter', sans-serif;
    }
    
    /* Header Gradient */
    .main-header {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
        padding: 1rem 0;
        animation: fadeIn 1s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Subtitle */
    .subtitle {
        text-align: center;
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 3rem;
        font-weight: 300;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 15px 40px rgba(0,0,0,0.15);
    }
    
    /* Status Boxes */
    .success-box {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #28a745;
        box-shadow: 0 5px 15px rgba(40, 167, 69, 0.2);
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    .warning-box {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeeba 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #ffc107;
        box-shadow: 0 5px 15px rgba(255, 193, 7, 0.2);
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    .danger-box {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #dc3545;
        box-shadow: 0 5px 15px rgba(220, 53, 69, 0.2);
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    .info-box {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        border-left: 5px solid #17a2b8;
        box-shadow: 0 5px 15px rgba(23, 162, 184, 0.2);
        margin: 1rem 0;
        animation: slideIn 0.5s ease-out;
    }
    
    @keyframes slideIn {
        from { opacity: 0; transform: translateX(-20px); }
        to { opacity: 1; transform: translateX(0); }
    }
    
    /* Sidebar Styling */
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 0.75rem 2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: #2c3e50;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid #667eea;
    }
    
    /* Charts Container */
    .chart-container {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 5px 20px rgba(0,0,0,0.08);
        margin: 1rem 0;
    }
    
    /* Badge Styling */
    .badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.85rem;
        margin: 0.2rem;
    }
    
    .badge-success {
        background: #28a745;
        color: white;
    }
    
    .badge-warning {
        background: #ffc107;
        color: #212529;
    }
    
    .badge-danger {
        background: #dc3545;
        color: white;
    }
    
    .badge-info {
        background: #17a2b8;
        color: white;
    }
    
    /* Hide Streamlit Branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Dataframe Styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)


# Helper functions
@st.cache_data(ttl=60)
def load_mlflow_runs():
    """Charge les runs MLflow"""
    try:
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000'))
        experiment = mlflow.get_experiment_by_name(os.getenv('MLFLOW_EXPERIMENT_NAME', 'churn-prediction'))
        
        if experiment:
            runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
            return runs
        return pd.DataFrame()
    except:
        return pd.DataFrame()


def generate_synthetic_predictions(n=100):
    """Génère des prédictions synthétiques pour la démo"""
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=n, freq='H')
    
    data = {
        'timestamp': dates,
        'customer_id': [f'C{str(i).zfill(6)}' for i in range(n)],
        'churn_probability': np.random.beta(2, 5, n),
        'prediction': np.random.choice([0, 1], n, p=[0.73, 0.27]),
        'latency_ms': np.random.gamma(2, 20, n),
        'model_version': ['1.0'] * n
    }
    
    df = pd.DataFrame(data)
    df['risk_level'] = pd.cut(df['churn_probability'], 
                               bins=[0, 0.4, 0.7, 1.0],
                               labels=['LOW', 'MEDIUM', 'HIGH'])
    
    return df


def get_api_health():
    """Check API health"""
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        return response.json()
    except:
        return {"status": "unavailable", "model_loaded": False}


def create_gauge_chart(value, title, max_value=100):
    """Créer un gauge chart stylé"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title, 'font': {'size': 20, 'color': '#2c3e50'}},
        delta={'reference': max_value * 0.85},
        gauge={
            'axis': {'range': [None, max_value], 'tickwidth': 1},
            'bar': {'color': "#667eea"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, max_value * 0.5], 'color': '#d4edda'},
                {'range': [max_value * 0.5, max_value * 0.8], 'color': '#fff3cd'},
                {'range': [max_value * 0.8, max_value], 'color': '#f8d7da'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    fig.update_layout(
        height=250,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "#2c3e50", 'family': "Inter"}
    )
    
    return fig


# Main app
def main():
    # Header
    col_logo, col_title = st.columns([1, 8])
    with col_logo:
        st.markdown('<div style="font-size: 5rem; text-align: center;">🚀</div>', unsafe_allow_html=True)
    with col_title:
        st.markdown('<div class="main-header" style="text-align: left; margin-top: 1rem;">Churn Prediction Analytics</div>', 
                    unsafe_allow_html=True)
    
    st.markdown('<div class="subtitle">MLOps Platform - Real-time Monitoring & Insights</div>', 
                unsafe_allow_html=True)
    
    # Sidebar Premium
    with st.sidebar:
        st.markdown("### 🎯 Navigation")
        
        page = st.radio(
            "",
            ["🏠 Dashboard", "📊 Model Analytics", "🔍 Live Predictions", 
             "⚠️ Monitoring Center", "🧪 API Testing"],
            label_visibility="collapsed"
        )
        
        st.markdown("---")
        st.markdown("### ⚙️ Settings")
        
        refresh_interval = st.slider("Auto-refresh (seconds)", 10, 300, 60)
        show_advanced = st.checkbox("Advanced Mode", value=False)
        
        st.markdown("---")
        st.markdown("### 📡 System Status")
        
        # API Health avec animation
        health = get_api_health()
        if health['status'] == 'healthy':
            st.markdown('<span class="badge badge-success">✅ API Online</span>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-danger">❌ API Offline</span>', 
                       unsafe_allow_html=True)
        
        # MLflow Status
        runs = load_mlflow_runs()
        if not runs.empty:
            st.markdown('<span class="badge badge-success">✅ MLflow Connected</span>', 
                       unsafe_allow_html=True)
        else:
            st.markdown('<span class="badge badge-warning">⚠️ MLflow Loading</span>', 
                       unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("### 📅 Last Updated")
        st.info(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    # Page routing
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Model Analytics":
        show_model_performance()
    elif page == "🔍 Live Predictions":
        show_predictions()
    elif page == "⚠️ Monitoring Center":
        show_monitoring()
    elif page == "🧪 API Testing":
        show_api_test()


def show_dashboard():
    """Page dashboard premium"""
    
    # KPIs avec gradient cards
    col1, col2, col3, col4 = st.columns(4)
    
    metrics_data = [
        ("🎯", "Model Accuracy", "87.3%", "+2.1%", True),
        ("📈", "Daily Predictions", "1,247", "+156", True),
        ("⚡", "Avg Latency", "45ms", "-5ms", True),
        ("⚠️", "At-Risk Clients", "234", "+12", False)
    ]
    
    for col, (icon, label, value, delta, is_good) in zip([col1, col2, col3, col4], metrics_data):
        with col:
            delta_color = "normal" if is_good else "inverse"
            st.metric(
                label=f"{icon} {label}",
                value=value,
                delta=delta,
                delta_color=delta_color
            )
    
    st.markdown("---")
    
    # Main Charts
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown('<div class="section-header">📊 Churn Probability Distribution</div>', 
                   unsafe_allow_html=True)
        
        predictions = generate_synthetic_predictions(300)
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(
            x=predictions['churn_probability'],
            nbinsx=40,
            marker=dict(
                color=predictions['churn_probability'],
                colorscale='RdYlGn_r',
                line=dict(color='white', width=1)
            ),
            hovertemplate='Probability: %{x:.2f}<br>Count: %{y}<extra></extra>'
        ))
        
        fig.update_layout(
            xaxis_title="Churn Probability",
            yaxis_title="Number of Customers",
            height=400,
            template='plotly_white',
            hovermode='x unified',
            font=dict(family="Inter")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown('<div class="section-header">🎯 Risk Distribution</div>', 
                   unsafe_allow_html=True)
        
        risk_counts = predictions['risk_level'].value_counts()
        
        colors = ['#28a745', '#ffc107', '#dc3545']
        
        fig = go.Figure(data=[go.Pie(
            labels=risk_counts.index,
            values=risk_counts.values,
            hole=0.5,
            marker=dict(colors=colors, line=dict(color='white', width=2)),
            textinfo='label+percent',
            textfont=dict(size=14, family="Inter"),
            hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            height=400,
            showlegend=True,
            legend=dict(orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.1),
            font=dict(family="Inter")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Timeline avec area chart
    st.markdown('<div class="section-header">📉 7-Day Prediction Trends</div>', 
               unsafe_allow_html=True)
    
    daily_preds = predictions.set_index('timestamp').resample('D').agg({
        'prediction': 'sum',
        'customer_id': 'count'
    }).reset_index()
    daily_preds.columns = ['Date', 'Churned', 'Total']
    daily_preds['Retained'] = daily_preds['Total'] - daily_preds['Churned']
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=daily_preds['Date'], 
        y=daily_preds['Retained'],
        name='Retained',
        fill='tonexty',
        line=dict(color='#28a745', width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    fig.add_trace(go.Scatter(
        x=daily_preds['Date'], 
        y=daily_preds['Churned'],
        name='Churned',
        fill='tozeroy',
        line=dict(color='#dc3545', width=3),
        mode='lines+markers',
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        height=350,
        xaxis_title="Date",
        yaxis_title="Number of Customers",
        template='plotly_white',
        hovermode='x unified',
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        font=dict(family="Inter")
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance Gauges
    st.markdown('<div class="section-header">⚡ Real-time Performance</div>', 
               unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fig = create_gauge_chart(87.3, "Model Accuracy %", 100)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = create_gauge_chart(45, "API Latency (ms)", 100)
        st.plotly_chart(fig, use_container_width=True)
    
    with col3:
        fig = create_gauge_chart(98.5, "API Uptime %", 100)
        st.plotly_chart(fig, use_container_width=True)


def show_model_performance():
    """Page model analytics premium"""
    st.markdown('<div class="section-header">📊 Model Performance Analytics</div>', 
               unsafe_allow_html=True)
    
    runs = load_mlflow_runs()
    
    if not runs.empty:
        # Extract metrics
        metrics_df = runs[['tags.mlflow.runName', 
                           'metrics.accuracy', 
                           'metrics.precision',
                           'metrics.recall',
                           'metrics.f1_score',
                           'metrics.roc_auc']].copy()
        
        metrics_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        metrics_df = metrics_df.dropna()
        
        # Styled table
        st.markdown("### 🏆 Model Comparison")
        st.dataframe(
            metrics_df.style
                .background_gradient(cmap='RdYlGn', subset=['Accuracy', 'F1-Score', 'ROC-AUC'])
                .format({'Accuracy': '{:.3f}', 'Precision': '{:.3f}', 'Recall': '{:.3f}', 
                        'F1-Score': '{:.3f}', 'ROC-AUC': '{:.3f}'}),
            use_container_width=True,
            height=250
        )
        
        # Radar Chart
        st.markdown("### 🎯 Multi-dimensional Comparison")
        
        fig = go.Figure()
        
        for idx, row in metrics_df.iterrows():
            fig.add_trace(go.Scatterpolar(
                r=[row['Accuracy'], row['Precision'], row['Recall'], row['F1-Score'], row['ROC-AUC']],
                theta=['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                fill='toself',
                name=row['Model']
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 1])
            ),
            showlegend=True,
            height=500,
            font=dict(family="Inter")
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
    else:
        st.warning("⚠️ No MLflow runs found. Please train models first.")
    
    # Feature importance
    st.markdown("### 🔑 Feature Importance Analysis")
    
    features = {
        'Feature': ['tenure_months', 'monthly_charges', 'contract_type', 
                   'total_charges', 'risk_score', 'tech_support',
                   'online_security', 'payment_method', 'is_new_customer', 'senior_citizen'],
        'Importance': [0.18, 0.15, 0.12, 0.10, 0.09, 0.08, 0.07, 0.06, 0.08, 0.07]
    }
    
    feature_df = pd.DataFrame(features).sort_values('Importance', ascending=True)
    
    fig = go.Figure(go.Bar(
        x=feature_df['Importance'],
        y=feature_df['Feature'],
        orientation='h',
        marker=dict(
            color=feature_df['Importance'],
            colorscale='Viridis',
            line=dict(color='white', width=1)
        ),
        text=feature_df['Importance'],
        texttemplate='%{text:.2f}',
        textposition='outside'
    ))
    
    fig.update_layout(
        height=500,
        xaxis_title="Importance Score",
        yaxis_title="Features",
        template='plotly_white',
        font=dict(family="Inter")
    )
    
    st.plotly_chart(fig, use_container_width=True)


def show_predictions():
    """Page predictions premium"""
    st.markdown('<div class="section-header">🔍 Live Prediction Stream</div>', 
               unsafe_allow_html=True)
    
    predictions = generate_synthetic_predictions(100)
    
    # Filtres stylés
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        risk_filter = st.multiselect(
            "🎯 Risk Level",
            options=['LOW', 'MEDIUM', 'HIGH'],
            default=['MEDIUM', 'HIGH']
        )
    
    with col2:
        pred_filter = st.selectbox(
            "📊 Prediction Type",
            options=['All', 'Churn', 'No Churn']
        )
    
    with col3:
        time_range = st.selectbox(
            "⏰ Time Range",
            options=['Last Hour', 'Last 24h', 'Last 7 days']
        )
    
    with col4:
        sort_by = st.selectbox(
            "🔽 Sort By",
            options=['Timestamp', 'Probability', 'Risk Level']
        )
    
    # Apply filters
    filtered = predictions[predictions['risk_level'].isin(risk_filter)]
    
    if pred_filter == 'Churn':
        filtered = filtered[filtered['prediction'] == 1]
    elif pred_filter == 'No Churn':
        filtered = filtered[filtered['prediction'] == 0]
    
    # Stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("📋 Total Predictions", len(filtered))
    with col2:
        churn_rate = (filtered['prediction'].sum() / len(filtered) * 100)
        st.metric("📈 Churn Rate", f"{churn_rate:.1f}%")
    with col3:
        avg_prob = filtered['churn_probability'].mean()
        st.metric("🎲 Avg Probability", f"{avg_prob:.1%}")
    
    st.markdown("---")
    
    # Display table
    display_df = filtered[['timestamp', 'customer_id', 'churn_probability', 
                           'prediction', 'risk_level']].copy()
    display_df['churn_probability'] = display_df['churn_probability'].apply(lambda x: f"{x:.2%}")
    display_df['prediction'] = display_df['prediction'].map({0: '✅ No Churn', 1: '❌ Churn'})
    display_df.columns = ['⏰ Timestamp', '👤 Customer', '📊 Probability', '🎯 Prediction', '⚠️ Risk']
    
    st.dataframe(
        display_df,
        use_container_width=True,
        height=500
    )
    
    # Download avec style
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        csv = filtered.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"predictions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )


def show_monitoring():
    """Page monitoring premium"""
    st.markdown('<div class="section-header">⚠️ System Monitoring Center</div>', 
               unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Data Drift Detection")
        
        drift_score = 0.08
        threshold = 0.10
        
        if drift_score < threshold:
            st.markdown(f'''
            <div class="success-box">
                <h3>✅ System Stable</h3>
                <p>Drift Score: <b>{drift_score:.3f}</b> (Threshold: {threshold})</p>
                <p>No action required. Data distribution is within normal parameters.</p>
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="danger-box">
                <h3>⚠️ Drift Detected!</h3>
                <p>Drift Score: <b>{drift_score:.3f}</b> (Threshold: {threshold})</p>
                <p>Model retraining recommended.</p>
            </div>
            ''', unsafe_allow_html=True)
        
        # Drift chart
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        drift_data = pd.DataFrame({
            'Date': dates,
            'Drift Score': np.random.uniform(0.02, 0.12, 30)
        })
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=drift_data['Date'],
            y=drift_data['Drift Score'],
            mode='lines+markers',
            name='Drift Score',
            line=dict(color='#667eea', width=3),
            marker=dict(size=8),
            fill='tonexty'
        ))
        
        fig.add_hline(
            y=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="Alert Threshold",
            annotation_position="right"
        )
        
        fig.update_layout(
            height=300,
            template='plotly_white',
            hovermode='x unified',
            font=dict(family="Inter")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.markdown("### ⚡ API Performance Metrics")
        
        avg_latency = 45
        p95_latency = 78
        p99_latency = 120
        
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            st.metric("Avg", f"{avg_latency}ms", "-5ms")
        with col_b:
            st.metric("P95", f"{p95_latency}ms", "+3ms")
        with col_c:
            st.metric("P99", f"{p99_latency}ms", "-8ms")
        
        # Latency distribution
        predictions = generate_synthetic_predictions(200)
        
        fig = go.Figure()
        
        fig.add_trace(go.Histogram(
            x=predictions['latency_ms'],
            nbinsx=40,
            marker=dict(
                color='#667eea',
                line=dict(color='white', width=1)
            ),
            name='Latency'
        ))
        
        fig.update_layout(
            height=300,
            xaxis_title="Latency (ms)",
            yaxis_title="Frequency",
            template='plotly_white',
            showlegend=False,
            font=dict(family="Inter")
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    # Alerts timeline
    st.markdown("### 🔔 Recent System Alerts")
    
    alerts = [
        ('2025-10-08 22:30', 'SUCCESS', '✅ Model v1.3 deployed successfully'),
        ('2025-10-08 20:15', 'INFO', 'ℹ️ Batch prediction job completed (1.2K records)'),
        ('2025-10-08 18:45', 'WARNING', '⚠️ API latency spike detected (avg: 95ms)'),
        ('2025-10-08 16:20', 'INFO', 'ℹ️ Database backup completed'),
        ('2025-10-08 14:00', 'SUCCESS', '✅ Model retraining completed (F1: 0.876)')
    ]
    
    for timestamp, alert_type, message in alerts:
        if alert_type == 'SUCCESS':
            st.markdown(f'''
            <div class="success-box">
                <b>{timestamp}</b> - {message}
            </div>
            ''', unsafe_allow_html=True)
        elif alert_type == 'WARNING':
            st.markdown(f'''
            <div class="warning-box">
                <b>{timestamp}</b> - {message}
            </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
            <div class="info-box">
                <b>{timestamp}</b> - {message}
            </div>
            ''', unsafe_allow_html=True)


def show_api_test():
    """Page API testing premium"""
    st.markdown('<div class="section-header">🧪 API Testing Laboratory</div>', 
               unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <h4>💡 Quick Testing Interface</h4>
        <p>Configure customer parameters and get real-time churn predictions from the ML model.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📝 Customer Profile")
        
        # Tabs for organized input
        tab1, tab2, tab3 = st.tabs(["👤 Basic Info", "💳 Billing", "📦 Services"])
        
        with tab1:
            customer_id = st.text_input("Customer ID", "C000001", key="cust_id")
            
            col_a, col_b = st.columns(2)
            with col_a:
                tenure = st.slider("Tenure (months)", 0, 72, 12, key="tenure")
                senior = st.selectbox("Senior Citizen", [0, 1], key="senior")
            with col_b:
                contract = st.selectbox("Contract Type", 
                                       ['Month-to-month', 'One year', 'Two year'],
                                       key="contract")
        
        with tab2:
            col_a, col_b = st.columns(2)
            with col_a:
                monthly_charges = st.number_input(
                    "Monthly Charges ($)", 
                    20.0, 200.0, 75.0, 
                    step=5.0,
                    key="monthly"
                )
            with col_b:
                total_charges = st.number_input(
                    "Total Charges ($)", 
                    100.0, 10000.0, 900.0,
                    step=100.0,
                    key="total"
                )
            
            payment = st.selectbox(
                "Payment Method",
                ['Electronic check', 'Mailed check', 'Credit card', 'Bank transfer'],
                key="payment"
            )
        
        with tab3:
            col_a, col_b = st.columns(2)
            
            with col_a:
                phone = st.selectbox("Phone Service", ['Yes', 'No'], key="phone")
                internet = st.selectbox("Internet Service", 
                                       ['DSL', 'Fiber optic', 'No'],
                                       key="internet")
                online_security = st.selectbox("Online Security", 
                                              ['Yes', 'No', 'No internet'],
                                              key="security")
            
            with col_b:
                tech_support = st.selectbox("Tech Support", 
                                           ['Yes', 'No', 'No internet'],
                                           key="support")
                streaming_tv = st.selectbox("Streaming TV", 
                                           ['Yes', 'No', 'No internet'],
                                           key="tv")
                streaming_movies = st.selectbox("Streaming Movies", 
                                               ['Yes', 'No', 'No internet'],
                                               key="movies")
        
        st.markdown("---")
        
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn2:
            predict_btn = st.button("🚀 Predict Churn", type="primary", use_container_width=True)
        
        if predict_btn:
            payload = {
                "customer_id": customer_id,
                "tenure_months": tenure,
                "monthly_charges": monthly_charges,
                "total_charges": total_charges,
                "contract_type": contract,
                "payment_method": payment,
                "internet_service": internet,
                "phone_service": phone,
                "online_security": online_security,
                "tech_support": tech_support,
                "senior_citizen": senior
            }
            
            with st.spinner('🔮 Analyzing customer data...'):
                try:
                    response = requests.post(
                    f"{API_URL}/predict",  # ✅ CORRECT
                    json=payload,
                    timeout=5
                )
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.session_state['prediction_result'] = result
                        st.success("✅ Prediction completed!")
                    else:
                        st.error(f"❌ API Error: {response.status_code}")
                        st.session_state.pop('prediction_result', None)
                        
                except requests.exceptions.ConnectionError:
                    st.error("❌ Cannot connect to API. Make sure it's running on localhost:8000")
                    st.session_state.pop('prediction_result', None)
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.pop('prediction_result', None)
    
    with col2:
        st.markdown("### 📊 Prediction Results")
        
        if 'prediction_result' in st.session_state:
            result = st.session_state['prediction_result']
            
            prob = result.get('churn_probability', 0)
            risk = result.get('risk_level', 'UNKNOWN')
            prediction = result.get('churn_prediction', False)
            
            # Big result card
            if risk == 'HIGH' or prob > 0.7:
                st.markdown(f'''
                <div class="danger-box" style="text-align: center; padding: 2rem;">
                    <h1 style="font-size: 4rem; margin: 0;">❌</h1>
                    <h2 style="color: #dc3545; margin: 1rem 0;">HIGH RISK</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{prob:.1%}</h1>
                    <p style="font-size: 1.2rem;">Churn Probability</p>
                    <hr style="border-color: #dc3545;">
                    <p style="margin-top: 1rem;"><b>Recommendation:</b> Immediate retention action required</p>
                </div>
                ''', unsafe_allow_html=True)
            elif risk == 'MEDIUM' or prob > 0.4:
                st.markdown(f'''
                <div class="warning-box" style="text-align: center; padding: 2rem;">
                    <h1 style="font-size: 4rem; margin: 0;">⚠️</h1>
                    <h2 style="color: #ffc107; margin: 1rem 0;">MEDIUM RISK</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{prob:.1%}</h1>
                    <p style="font-size: 1.2rem;">Churn Probability</p>
                    <hr style="border-color: #ffc107;">
                    <p style="margin-top: 1rem;"><b>Recommendation:</b> Monitor closely and engage proactively</p>
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div class="success-box" style="text-align: center; padding: 2rem;">
                    <h1 style="font-size: 4rem; margin: 0;">✅</h1>
                    <h2 style="color: #28a745; margin: 1rem 0;">LOW RISK</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">{prob:.1%}</h1>
                    <p style="font-size: 1.2rem;">Churn Probability</p>
                    <hr style="border-color: #28a745;">
                    <p style="margin-top: 1rem;"><b>Status:</b> Customer retention healthy</p>
                </div>
                ''', unsafe_allow_html=True)
            
            # Detailed metrics
            st.markdown("### 📋 Detailed Analysis")
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                st.metric("Churn Prediction", "Yes" if prediction else "No")
                st.metric("Risk Level", risk)
            
            with col_b:
                st.metric("Confidence", f"{result.get('confidence', 0):.1%}")
                st.metric("Model Version", result.get('model_version', 'Unknown'))
            
            # JSON response
            with st.expander("🔍 View Raw API Response"):
                st.json(result)
            
            # Action recommendations
            if prob > 0.5:
                st.markdown("### 💡 Recommended Actions")
                
                actions = [
                    "📞 Schedule personalized retention call",
                    "🎁 Offer loyalty discount or upgrade incentive",
                    "📧 Send targeted retention email campaign",
                    "👤 Assign dedicated account manager",
                    "📊 Review service usage and satisfaction"
                ]
                
                for action in actions:
                    st.markdown(f"- {action}")
        
        else:
            st.markdown('''
            <div class="info-box" style="text-align: center; padding: 3rem;">
                <h1 style="font-size: 4rem; margin: 0;">🎯</h1>
                <h3 style="margin: 1rem 0;">Awaiting Prediction</h3>
                <p>Configure customer parameters and click "Predict Churn" to see results</p>
            </div>
            ''', unsafe_allow_html=True)
            
            # Sample predictions showcase
            st.markdown("### 📌 Sample Predictions")
            
            samples = pd.DataFrame({
                'Customer': ['C001234', 'C005678', 'C009012'],
                'Tenure': ['3 months', '24 months', '60 months'],
                'Contract': ['Month-to-month', 'One year', 'Two year'],
                'Probability': ['78%', '35%', '12%'],
                'Risk': ['🔴 HIGH', '🟡 MEDIUM', '🟢 LOW']
            })
            
            st.dataframe(samples, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()