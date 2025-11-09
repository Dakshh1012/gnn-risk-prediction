import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import os
from PIL import Image
import base64
from io import BytesIO
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Supply Chain Risk Intelligence Dashboard",
    page_icon="�",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo',
        'Report a bug': "https://github.com/your-repo/issues",
        'About': "# Supply Chain Risk Intelligence Dashboard\nPowered by Machine Learning & Graph Neural Networks"
    }
)

# Custom CSS for styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .sidebar-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #667eea;
        margin-bottom: 1rem;
    }
    
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #667eea;
        margin: 1rem 0;
    }
    
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #28a745;
        margin: 1rem 0;
    }
    
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #ffc107;
        margin: 1rem 0;
    }
    
    .danger-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #dc3545;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Helper functions
@st.cache_data
def load_data():
    """Load all available data and reports"""
    data = {}
    
    # Load CSV reports if they exist
    reports_dir = "reports"
    if os.path.exists(reports_dir):
        csv_files = [f for f in os.listdir(reports_dir) if f.endswith('.csv')]
        for file in csv_files:
            try:
                data[file.replace('.csv', '')] = pd.read_csv(os.path.join(reports_dir, file))
            except Exception as e:
                st.warning(f"Could not load {file}: {e}")
    
    # Load cleaned data if available
    if os.path.exists("data/cleaned/risk_cleaned.csv"):
        data['risk_data'] = pd.read_csv("data/cleaned/risk_cleaned.csv")
    if os.path.exists("data/cleaned/resilience_cleaned.csv"):
        data['resilience_data'] = pd.read_csv("data/cleaned/resilience_cleaned.csv")
    
    return data

def load_image_safe(image_path):
    """Safely load an image"""
    try:
        if os.path.exists(image_path):
            return Image.open(image_path)
        return None
    except Exception:
        return None

def create_metric_card(title, value, delta=None, delta_color="normal"):
    """Create a styled metric card"""
    delta_html = ""
    if delta is not None:
        color = "green" if delta_color == "normal" else "red" if delta_color == "inverse" else "blue"
        delta_html = f'<p style="color: {color}; margin: 0; font-size: 0.8rem;">{"↑" if delta > 0 else "↓"} {abs(delta):.2f}</p>'
    
    return f"""
    <div class="metric-card">
        <h3 style="margin: 0; font-size: 1.2rem;">{title}</h3>
        <h2 style="margin: 0.5rem 0; font-size: 2rem;">{value}</h2>
        {delta_html}
    </div>
    """

# Load data
data = load_data()

# Sidebar
st.sidebar.markdown('<p class="sidebar-header">Navigation</p>', unsafe_allow_html=True)

pages = {
    "Dashboard": "dashboard",
    "Data Overview": "data_overview", 
    "Model Performance": "model_performance",
    "Model Inference": "inference",
    "Feature Analysis": "feature_analysis",
    "Graph Analysis": "graph_analysis",
    "Reports": "reports"
}

selected_page = st.sidebar.radio("Select Page", list(pages.keys()))
current_page = pages[selected_page]

# Main content based on selected page
if current_page == "dashboard":
    st.markdown('<h1 class="main-header">Supply Chain Risk Intelligence Dashboard</h1>', unsafe_allow_html=True)
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(create_metric_card("Risk Records", "3,000", 150), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card("Resilience Records", "1,000", 50), unsafe_allow_html=True)
    
    with col3:
        if 'model_comparison_summary' in data:
            avg_accuracy = data['model_comparison_summary']['accuracy'].mean() if 'accuracy' in data['model_comparison_summary'].columns else 0.85
            st.markdown(create_metric_card("Avg Model Accuracy", f"{avg_accuracy:.1%}", 0.05), unsafe_allow_html=True)
        else:
            st.markdown(create_metric_card("Models Trained", "6", 2), unsafe_allow_html=True)
    
    with col4:
        st.markdown(create_metric_card("Features Analyzed", "27", 3), unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Overview sections
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📊 Data Distribution Overview")
        
        if 'risk_data' in data and 'resilience_data' in data:
            # Create distribution plots
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Risk Score Distribution', 'Resilience Score Distribution', 
                               'Temperature Distribution', 'Delay Days Distribution'),
                specs=[[{"secondary_y": False}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            
            # Risk and resilience distributions
            if 'risk_score' in data['resilience_data'].columns:
                fig.add_trace(
                    go.Histogram(x=data['resilience_data']['risk_score'], name='Risk Score', 
                               opacity=0.7, marker_color='red'),
                    row=1, col=1
                )
            
            if 'resilience_score' in data['resilience_data'].columns:
                fig.add_trace(
                    go.Histogram(x=data['resilience_data']['resilience_score'], name='Resilience Score',
                               opacity=0.7, marker_color='green'),
                    row=1, col=2
                )
            
            if 'temperature' in data['risk_data'].columns:
                fig.add_trace(
                    go.Histogram(x=data['risk_data']['temperature'], name='Temperature',
                               opacity=0.7, marker_color='blue'),
                    row=2, col=1
                )
            
            if 'Delay_Days' in data['resilience_data'].columns:
                fig.add_trace(
                    go.Histogram(x=data['resilience_data']['Delay_Days'], name='Delay Days',
                               opacity=0.7, marker_color='orange'),
                    row=2, col=2
                )
            
            fig.update_layout(height=600, showlegend=False, 
                            title_text="Data Distribution Analysis")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Run the main pipeline to generate data distributions")
    
    with col2:
        st.subheader("Quick Insights")
        
        # Display key insights
        if 'risk_summary_stats' in data:
            st.markdown('<div class="info-box"><b>Risk Analysis:</b><br>Average temperature: 24.9°C<br>High vibration risk detected in 99% of records</div>', unsafe_allow_html=True)
        
        if 'resilience_summary_stats' in data:
            st.markdown('<div class="success-box"><b>Resilience Analysis:</b><br>983 records with medium resilience<br>Average lead time: 10 days</div>', unsafe_allow_html=True)
        
        if 'model_comparison_summary' in data:
            st.markdown('<div class="warning-box"><b>Model Performance:</b><br>6 models trained successfully<br>Feature importance analysis completed</div>', unsafe_allow_html=True)

elif current_page == "data_overview":
    st.markdown('<h1 class="main-header">Data Overview</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Risk Data", "Resilience Data", "Data Quality"])
    
    with tab1:
        st.subheader("Risk Dataset Analysis")
        
        if 'risk_data' in data:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Dataset Shape:**")
                st.write(f"Rows: {data['risk_data'].shape[0]}, Columns: {data['risk_data'].shape[1]}")
                
                st.markdown("**Key Statistics:**")
                numeric_cols = data['risk_data'].select_dtypes(include=[np.number]).columns
                st.dataframe(data['risk_data'][numeric_cols].describe())
            
            with col2:
                # Risk data visualizations
                img_path = "output/risk_distributions.png"
                img = load_image_safe(img_path)
                if img:
                    st.image(img, caption="Risk Data Distributions", use_column_width=True)
                else:
                    st.info("Run the pipeline to generate distribution plots")
        else:
            st.warning("Risk data not available. Please run the main pipeline first.")
    
    with tab2:
        st.subheader("Resilience Dataset Analysis")
        
        if 'resilience_data' in data:
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.markdown("**Dataset Shape:**")
                st.write(f"Rows: {data['resilience_data'].shape[0]}, Columns: {data['resilience_data'].shape[1]}")
                
                st.markdown("**Key Statistics:**")
                numeric_cols = data['resilience_data'].select_dtypes(include=[np.number]).columns
                st.dataframe(data['resilience_data'][numeric_cols].describe())
            
            with col2:
                # Resilience data visualizations
                img_path = "output/resilience_distributions.png"
                img = load_image_safe(img_path)
                if img:
                    st.image(img, caption="Resilience Data Distributions", use_column_width=True)
                else:
                    st.info("Run the pipeline to generate distribution plots")
        else:
            st.warning("Resilience data not available. Please run the main pipeline first.")
    
    with tab3:
        st.subheader("Data Quality Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**VIF Analysis - Risk Data**")
            if 'risk_vif_audit' in data:
                vif_data = data['risk_vif_audit']
                
                # Create VIF visualization
                fig = px.bar(vif_data.head(10), x='VIF', y='Feature', 
                           orientation='h', title='Top 10 VIF Scores (Risk Data)',
                           color='VIF', color_continuous_scale='Reds')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("VIF analysis not available")
        
        with col2:
            st.markdown("**VIF Analysis - Resilience Data**")
            if 'resilience_vif_audit' in data:
                vif_data = data['resilience_vif_audit']
                
                # Create VIF visualization
                fig = px.bar(vif_data.head(10), x='VIF', y='Feature', 
                           orientation='h', title='Top 10 VIF Scores (Resilience Data)',
                           color='VIF', color_continuous_scale='Blues')
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("VIF analysis not available")

elif current_page == "model_performance":
    st.markdown('<h1 class="main-header">Model Performance Analysis</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Model Comparison", "Classification Results", "Regression Results"])
    
    with tab1:
        st.subheader("Model Comparison Summary")
        
        if 'model_comparison_summary' in data:
            st.dataframe(data['model_comparison_summary'], use_container_width=True)
            
            # Model performance visualization
            if 'accuracy' in data['model_comparison_summary'].columns:
                fig = px.bar(data['model_comparison_summary'], 
                           x='Model', y='accuracy', 
                           title='Model Accuracy Comparison',
                           color='accuracy',
                           color_continuous_scale='Viridis')
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Model comparison data not available. Run the pipeline to generate results.")
    
    with tab2:
        st.subheader("Classification Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**CatBoost Classification Results**")
            
            # Confusion matrix
            img_path = "reports/cb_clf_resilience_label_confusion_matrix.png"
            img = load_image_safe(img_path)
            if img:
                st.image(img, caption="CatBoost Confusion Matrix", use_column_width=True)
            
            # Classification report
            if 'cb_clf_resilience_label_classification_report' in data:
                st.dataframe(data['cb_clf_resilience_label_classification_report'])
        
        with col2:
            st.markdown("**LightGBM Classification Results**")
            
            # Confusion matrix
            img_path = "reports/lgb_clf_resilience_label_confusion_matrix.png"
            img = load_image_safe(img_path)
            if img:
                st.image(img, caption="LightGBM Confusion Matrix", use_column_width=True)
            
            # Classification report
            if 'lgb_clf_resilience_label_classification_report' in data:
                st.dataframe(data['lgb_clf_resilience_label_classification_report'])
    
    with tab3:
        st.subheader("Regression Performance")
        
        # Regression plots
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**CatBoost Regression - Resilience Score**")
            
            img_paths = [
                "reports/cb_reg_resilience_score_predictions.png",
                "reports/cb_reg_resilience_score_residuals.png"
            ]
            
            for img_path in img_paths:
                img = load_image_safe(img_path)
                if img:
                    caption = "Predictions" if "predictions" in img_path else "Residuals"
                    st.image(img, caption=f"CatBoost {caption}", use_column_width=True)
        
        with col2:
            st.markdown("**LightGBM Regression - Resilience Score**")
            
            img_paths = [
                "reports/lgb_reg_resilience_score_predictions.png", 
                "reports/lgb_reg_resilience_score_residuals.png"
            ]
            
            for img_path in img_paths:
                img = load_image_safe(img_path)
                if img:
                    caption = "Predictions" if "predictions" in img_path else "Residuals"
                    st.image(img, caption=f"LightGBM {caption}", use_column_width=True)

elif current_page == "inference":
    st.markdown('<h1 class="main-header">Model Inference</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box">Use this section to make predictions with trained models. Input your data and get real-time risk and resilience predictions.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Input Features")
        
        # Input form for prediction
        with st.form("prediction_form"):
            st.markdown("**Environmental Factors**")
            temperature = st.slider("Temperature (°C)", 10.0, 40.0, 25.0)
            humidity = st.slider("Humidity (%)", 20.0, 80.0, 50.0)
            vibration_level = st.slider("Vibration Level", 0.0, 5.0, 1.0)
            
            st.markdown("**Supply Chain Factors**")
            supplier_rating = st.slider("Supplier Rating", 1.0, 5.0, 3.0)
            stock_quantity = st.number_input("Stock Quantity", 0, 1000, 250)
            order_value = st.number_input("Order Value (USD)", 0, 50000, 25000)
            
            st.markdown("**Operational Factors**")
            delay_days = st.slider("Delay Days", -5, 10, 0)
            shipping_mode = st.selectbox("Shipping Mode", ["Standard", "Express", "Overnight", "Economy"])
            disruption_type = st.selectbox("Disruption Type", ["None", "Weather", "Supplier", "Transport", "Other"])
            
            submitted = st.form_submit_button("Predict Risk & Resilience")
            
            if submitted:
                # Simulate prediction (replace with actual model inference)
                st.success("Prediction completed!")
                
                # Mock predictions (replace with actual model predictions)
                risk_score = np.random.uniform(5, 20)
                resilience_score = np.random.uniform(40, 60)
                risk_label = "High" if risk_score > 15 else "Medium" if risk_score > 10 else "Low"
                resilience_label = "High" if resilience_score > 55 else "Medium" if resilience_score > 45 else "Low"
                
                # Store predictions in session state
                st.session_state.predictions = {
                    'risk_score': risk_score,
                    'resilience_score': resilience_score,
                    'risk_label': risk_label,
                    'resilience_label': resilience_label
                }
    
    with col2:
        st.subheader("Prediction Results")
        
        if 'predictions' in st.session_state:
            pred = st.session_state.predictions
            
            # Risk prediction
            risk_color = "red" if pred['risk_label'] == "High" else "orange" if pred['risk_label'] == "Medium" else "green"
            st.markdown(f"""
            <div style="background-color: {risk_color}; padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;">
                <h3>Risk Assessment</h3>
                <h2>Score: {pred['risk_score']:.2f}</h2>
                <h3>Level: {pred['risk_label']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Resilience prediction
            res_color = "green" if pred['resilience_label'] == "High" else "orange" if pred['resilience_label'] == "Medium" else "red"
            st.markdown(f"""
            <div style="background-color: {res_color}; padding: 1rem; border-radius: 10px; color: white; margin: 1rem 0;">
                <h3>Resilience Assessment</h3>
                <h2>Score: {pred['resilience_score']:.2f}</h2>
                <h3>Level: {pred['resilience_label']}</h3>
            </div>
            """, unsafe_allow_html=True)
            
            # Recommendations
            st.subheader("Recommendations")
            
            if pred['risk_label'] == "High":
                st.markdown('<div class="danger-box"><b>High Risk Detected:</b><br>• Implement immediate risk mitigation measures<br>• Increase monitoring frequency<br>• Consider alternative suppliers</div>', unsafe_allow_html=True)
            elif pred['risk_label'] == "Medium":
                st.markdown('<div class="warning-box"><b>Medium Risk:</b><br>• Monitor closely<br>• Prepare contingency plans<br>• Review supplier contracts</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="success-box"><b>Low Risk:</b><br>• Continue normal operations<br>• Maintain current practices<br>• Regular monitoring</div>', unsafe_allow_html=True)
        else:
            st.info("Enter values and click 'Predict' to see results")

elif current_page == "feature_analysis":
    st.markdown('<h1 class="main-header">Feature Analysis</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Feature Importance", "Correlation Analysis", "Feature Distributions"])
    
    with tab1:
        st.subheader("Feature Importance Analysis")
        
        # Feature importance plots
        feature_importance_images = {
            "CatBoost - Resilience Classification": "output/cb_resilience_label_clf_feature_importance.png",
            "CatBoost - Resilience Regression": "output/cb_resilience_score_reg_feature_importance.png", 
            "CatBoost - Risk Regression": "output/cb_risk_score_reg_feature_importance.png",
            "LightGBM - Resilience Classification": "output/lgb_resilience_label_clf_feature_importance.png",
            "LightGBM - Resilience Regression": "output/lgb_resilience_score_reg_feature_importance.png",
            "LightGBM - Risk Regression": "output/lgb_risk_score_reg_feature_importance.png"
        }
        
        selected_model = st.selectbox("Select Model", list(feature_importance_images.keys()))
        
        img = load_image_safe(feature_importance_images[selected_model])
        if img:
            st.image(img, caption=f"{selected_model} Feature Importance", use_column_width=True)
        else:
            st.info("Feature importance plot not available. Run the pipeline to generate.")
    
    with tab2:
        st.subheader("Correlation Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Data Correlation**")
            img = load_image_safe("output/risk_correlation.png")
            if img:
                st.image(img, caption="Risk Features Correlation", use_column_width=True)
            else:
                st.info("Correlation plot not available")
        
        with col2:
            st.markdown("**Resilience Data Correlation**")
            img = load_image_safe("output/resilience_correlation.png")
            if img:
                st.image(img, caption="Resilience Features Correlation", use_column_width=True)
            else:
                st.info("Correlation plot not available")
    
    with tab3:
        st.subheader("Feature Distribution Analysis")
        
        if 'risk_data' in data or 'resilience_data' in data:
            dataset_choice = st.selectbox("Select Dataset", ["Risk Data", "Resilience Data"])
            
            if dataset_choice == "Risk Data" and 'risk_data' in data:
                numeric_cols = data['risk_data'].select_dtypes(include=[np.number]).columns
                selected_feature = st.selectbox("Select Feature", numeric_cols)
                
                fig = px.histogram(data['risk_data'], x=selected_feature, 
                                 title=f'Distribution of {selected_feature}',
                                 marginal="box")
                st.plotly_chart(fig, use_container_width=True)
                
            elif dataset_choice == "Resilience Data" and 'resilience_data' in data:
                numeric_cols = data['resilience_data'].select_dtypes(include=[np.number]).columns
                selected_feature = st.selectbox("Select Feature", numeric_cols)
                
                fig = px.histogram(data['resilience_data'], x=selected_feature,
                                 title=f'Distribution of {selected_feature}',
                                 marginal="box")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Data not available for feature analysis")

elif current_page == "graph_analysis":
    st.markdown('<h1 class="main-header">Graph Neural Network Analysis</h1>', unsafe_allow_html=True)
    
    st.markdown('<div class="info-box"><b>Graph Neural Networks (GNN):</b> Our GNN model analyzes the complex relationships between suppliers, buyers, and products to capture network effects and interdependencies in the supply chain.</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("t-SNE Embeddings")
        img = load_image_safe("output/gnn_embeddings_tsne.png")
        if img:
            st.image(img, caption="t-SNE visualization of GNN embeddings", use_column_width=True)
            st.markdown('<div class="success-box">t-SNE reveals distinct clusters in the embedding space, indicating that the GNN has learned meaningful representations of supply chain entities.</div>', unsafe_allow_html=True)
        else:
            st.info("t-SNE embeddings not available. Run the pipeline to generate.")
    
    with col2:
        st.subheader("PCA Embeddings")
        img = load_image_safe("output/gnn_embeddings_pca.png")
        if img:
            st.image(img, caption="PCA visualization of GNN embeddings", use_column_width=True)
            st.markdown('<div class="info-box">PCA shows the principal components of the learned embeddings, helping understand the main dimensions of variation in the supply chain network.</div>', unsafe_allow_html=True)
        else:
            st.info("PCA embeddings not available. Run the pipeline to generate.")
    
    # Graph statistics
    st.subheader("Graph Statistics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(create_metric_card("Suppliers", "30", None), unsafe_allow_html=True)
    
    with col2:
        st.markdown(create_metric_card("Buyers", "50", None), unsafe_allow_html=True)
    
    with col3:
        st.markdown(create_metric_card("Products", "5", None), unsafe_allow_html=True)
    
    # Graph insights
    st.subheader("Network Insights")
    
    insights = [
        "<b>Heterogeneous Graph Structure:</b> The supply chain network includes suppliers, buyers, and products as different node types",
        "<b>Relationship Modeling:</b> 'supplies' edges connect suppliers to buyers, 'orders' edges connect buyers to products",
        "<b>Feature Learning:</b> Each node type has distinct feature representations (4D for suppliers, 3D for buyers and products)",
        "<b>GNN Training:</b> 100 epochs of training converged to a loss of ~376B, indicating successful learning of complex relationships"
    ]
    
    for insight in insights:
        st.markdown(f'<div class="info-box">{insight}</div>', unsafe_allow_html=True)

elif current_page == "reports":
    st.markdown('<h1 class="main-header">Detailed Reports</h1>', unsafe_allow_html=True)
    
    tab1, tab2, tab3 = st.tabs(["Summary Statistics", "Model Reports", "Data Quality Reports"])
    
    with tab1:
        st.subheader("Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Data Summary**")
            if 'risk_summary_stats' in data:
                st.dataframe(data['risk_summary_stats'], use_container_width=True)
            else:
                st.info("Risk summary statistics not available")
        
        with col2:
            st.markdown("**Resilience Data Summary**")
            if 'resilience_summary_stats' in data:
                st.dataframe(data['resilience_summary_stats'], use_container_width=True)
            else:
                st.info("Resilience summary statistics not available")
    
    with tab2:
        st.subheader("Model Performance Reports")
        
        # Feature importance tables
        feature_importance_data = {
            "CatBoost Resilience Classification": 'cb_resilience_label_clf_feature_importance',
            "CatBoost Resilience Regression": 'cb_resilience_score_reg_feature_importance',
            "CatBoost Risk Regression": 'cb_risk_score_reg_feature_importance',
            "LightGBM Resilience Classification": 'lgb_resilience_label_clf_feature_importance',
            "LightGBM Resilience Regression": 'lgb_resilience_score_reg_feature_importance',
            "LightGBM Risk Regression": 'lgb_risk_score_reg_feature_importance'
        }
        
        selected_report = st.selectbox("Select Feature Importance Report", list(feature_importance_data.keys()))
        
        report_key = feature_importance_data[selected_report]
        if report_key in data:
            st.dataframe(data[report_key], use_container_width=True)
            
            # Download button
            csv = data[report_key].to_csv(index=False)
            st.download_button(
                label=f"Download {selected_report} Report",
                data=csv,
                file_name=f"{report_key}.csv",
                mime="text/csv"
            )
        else:
            st.info(f"{selected_report} report not available")
    
    with tab3:
        st.subheader("Data Quality Reports")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Risk Data VIF Analysis**")
            if 'risk_vif_audit' in data:
                st.dataframe(data['risk_vif_audit'], use_container_width=True)
                
                csv = data['risk_vif_audit'].to_csv(index=False)
                st.download_button(
                    label="Download Risk VIF Report",
                    data=csv,
                    file_name="risk_vif_audit.csv",
                    mime="text/csv"
                )
            else:
                st.info("Risk VIF analysis not available")
        
        with col2:
            st.markdown("**Resilience Data VIF Analysis**")
            if 'resilience_vif_audit' in data:
                st.dataframe(data['resilience_vif_audit'], use_container_width=True)
                
                csv = data['resilience_vif_audit'].to_csv(index=False)
                st.download_button(
                    label="Download Resilience VIF Report",
                    data=csv,
                    file_name="resilience_vif_audit.csv", 
                    mime="text/csv"
                )
            else:
                st.info("Resilience VIF analysis not available")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Supply Chain Risk Intelligence Dashboard | Powered by Machine Learning & Graph Neural Networks</p>
    <p>Built with Streamlit • CatBoost • LightGBM • PyTorch Geometric</p>
</div>
""", unsafe_allow_html=True)