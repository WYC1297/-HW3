"""
Enhanced SMS Spam Classifier - Advanced Streamlit Web Application
Comprehensive interactive interface with advanced analytics and multi-phase model analysis
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from pathlib import Path
import sys
import os
from datetime import datetime
import json
import random
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from streamlit_option_menu import option_menu
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

# Add src directory to path
current_dir = Path(__file__).parent if '__file__' in locals() else Path.cwd()
sys.path.append(str(current_dir / 'src'))

# Import custom modules
from src.data_loader import SMSDataLoader
from src.advanced_preprocessor import AdvancedTextPreprocessor

# Page configuration
st.set_page_config(
    page_title="Enhanced SMS Spam Classifier",
    page_icon="🎨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #FF6B6B, #4ECDC4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .phase-header {
        font-size: 1.8rem;
        font-weight: bold;
        color: #2E86AB;
        margin: 1rem 0;
        padding: 0.5rem;
        border-left: 4px solid #4ECDC4;
        background-color: #F8F9FA;
    }
    
    .metric-container {
        background-color: #FFFFFF;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
    
    .model-performance {
        border: 2px solid #4ECDC4;
        border-radius: 15px;
        padding: 1rem;
        margin: 1rem 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    .visualization-section {
        background-color: #F8F9FA;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .success-metric {
        color: #28a745;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .warning-metric {
        color: #ffc107;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .error-metric {
        color: #dc3545;
        font-weight: bold;
        font-size: 1.2rem;
    }
    
    .feature-insight {
        background-color: #E8F4FD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #45B7D1;
        margin: 0.5rem 0;
    }
    
    .dataset-stats {
        background-color: #FFF3CD;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #FFEAA7;
        margin: 0.5rem 0;
    }
    
    .spam-result {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #f5c6cb;
    }
    
    .ham-result {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedSpamClassifierApp:
    """Enhanced Streamlit application class for comprehensive SMS spam classification analysis."""
    
    def __init__(self):
        self.colors = {
            'spam': '#FF6B6B',
            'ham': '#4ECDC4',
            'primary': '#45B7D1',
            'secondary': '#96CEB4',
            'accent': '#FFEAA7'
        }
        self.data_loader = SMSDataLoader()
        self.preprocessor = AdvancedTextPreprocessor()
        self.models = {}
        self.vectorizers = {}
        self.feature_selectors = {}
        self.model_configs = [
            {
                'name': 'Phase 1 (Baseline)',
                'model_path': 'models/svm_spam_classifier.joblib',
                'vectorizer_path': 'models/tfidf_vectorizer.joblib',
                'feature_selector_path': None,
                'description': 'Baseline SVM with basic TF-IDF features'
            },
            {
                'name': 'Phase 2 (Optimized)',
                'model_path': 'models/optimized_svm_classifier.joblib',
                'vectorizer_path': 'models/optimized_tfidf_vectorizer.joblib',
                'feature_selector_path': 'models/optimized_feature_selector.joblib',
                'description': 'Optimized SVM with feature selection and tuned parameters'
            },
            {
                'name': 'Phase 3 (Advanced)',
                'model_path': 'models/phase3_final_svm_classifier.joblib',
                'vectorizer_path': 'models/phase3_final_tfidf_vectorizer.joblib',
                'feature_selector_path': 'models/phase3_final_feature_selector.joblib',
                'description': 'Advanced SVM with SMOTE balancing and grid search optimization'
            }
        ]
    
    @st.cache_data
    def load_models(_self):
        """Load all available models with enhanced error handling."""
        models = {}
        vectorizers = {}
        feature_selectors = {}
        
        for config in _self.model_configs:
            try:
                model_path = Path(config['model_path'])
                vectorizer_path = Path(config['vectorizer_path'])
                
                if model_path.exists() and vectorizer_path.exists():
                    models[config['name']] = joblib.load(model_path)
                    vectorizers[config['name']] = joblib.load(vectorizer_path)
                    
                    # Load feature selector if available
                    if config['feature_selector_path']:
                        selector_path = Path(config['feature_selector_path'])
                        if selector_path.exists():
                            feature_selectors[config['name']] = joblib.load(selector_path)
                    
                    st.success(f"✅ Successfully loaded {config['name']}")
                else:
                    st.warning(f"⚠️ Model files not found for {config['name']}")
                    
            except Exception as e:
                st.error(f"❌ Error loading {config['name']}: {str(e)}")
        
        return models, vectorizers, feature_selectors
    
    @st.cache_data
    def load_data(_self):
        """Load and prepare dataset."""
        try:
            df = _self.data_loader.load_data()
            if df is not None:
                df = _self.data_loader.basic_clean()
                return df
            else:
                st.error("❌ Failed to load dataset")
                return None
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return None
    
    def predict_message(self, message: str, model_name: str):
        """Predict spam/ham for a given message using specified model."""
        try:
            if model_name not in self.models:
                return None, None, "Model not available"
            
            # Get model components
            model = self.models[model_name]
            vectorizer = self.vectorizers[model_name]
            feature_selector = self.feature_selectors.get(model_name)
            
            # Transform message
            X_transformed = vectorizer.transform([message])
            
            # Apply feature selection if available
            if feature_selector is not None:
                X_transformed = feature_selector.transform(X_transformed)
            
            # Convert to dense if needed
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
            
            # Make prediction
            prediction = model.predict(X_transformed)[0]
            confidence = model.decision_function(X_transformed)[0]
            
            # Normalize confidence score
            scaler = MinMaxScaler()
            confidence_norm = scaler.fit_transform([[confidence]])[0][0]
            
            return prediction, confidence_norm, None
            
        except Exception as e:
            return None, None, str(e)
    
    def create_dataset_overview(self, df: pd.DataFrame):
        """Create enhanced dataset overview with advanced statistics."""
        st.markdown('<div class="phase-header">📊 Dataset Overview</div>', unsafe_allow_html=True)
        
        # Basic statistics
        total_messages = len(df)
        spam_count = len(df[df['label'] == 'spam'])
        ham_count = len(df[df['label'] == 'ham'])
        spam_percentage = (spam_count / total_messages) * 100
        
        # Enhanced statistics
        df = df.copy()
        df['message_length'] = df['message'].str.len()
        df['word_count'] = df['message'].str.split().str.len()
        df['exclamation_count'] = df['message'].apply(lambda x: x.count('!'))
        df['uppercase_ratio'] = df['message'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📧 Total Messages", f"{total_messages:,}")
            st.metric("📈 Avg Spam Length", f"{df[df['label'] == 'spam']['message_length'].mean():.0f}")
        
        with col2:
            st.metric("🚫 Spam Messages", f"{spam_count:,}")
            st.metric("📊 Avg Ham Length", f"{df[df['label'] == 'ham']['message_length'].mean():.0f}")
        
        with col3:
            st.metric("✅ Ham Messages", f"{ham_count:,}")
            st.metric("🔤 Avg Spam Words", f"{df[df['label'] == 'spam']['word_count'].mean():.1f}")
        
        with col4:
            st.metric("⚖️ Spam Percentage", f"{spam_percentage:.1f}%")
            st.metric("💬 Avg Ham Words", f"{df[df['label'] == 'ham']['word_count'].mean():.1f}")
        
        # Enhanced visualization
        st.markdown('<div class="visualization-section">', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Class distribution pie chart
            fig_pie = px.pie(
                values=[spam_count, ham_count],
                names=['Spam', 'Ham'],
                title="📊 Message Distribution",
                color_discrete_sequence=[self.colors['spam'], self.colors['ham']]
            )
            fig_pie.update_layout(height=400)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Message length comparison
            fig_box = go.Figure()
            
            fig_box.add_trace(go.Box(
                y=df[df['label'] == 'spam']['message_length'],
                name='Spam',
                marker_color=self.colors['spam']
            ))
            
            fig_box.add_trace(go.Box(
                y=df[df['label'] == 'ham']['message_length'],
                name='Ham',
                marker_color=self.colors['ham']
            ))
            
            fig_box.update_layout(
                title="📏 Message Length Distribution",
                yaxis_title="Character Count",
                height=400
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return df
    
    def create_advanced_wordclouds(self, df: pd.DataFrame):
        """Generate advanced word clouds."""
        st.markdown('<div class="phase-header">☁️ Word Cloud Analysis</div>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🚫 Spam Messages")
            spam_text = ' '.join(df[df['label'] == 'spam']['message'])
            
            if spam_text:
                spam_wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='Reds',
                    max_words=50,
                    random_state=42
                ).generate(spam_text)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(spam_wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
        
        with col2:
            st.subheader("✅ Ham Messages")
            ham_text = ' '.join(df[df['label'] == 'ham']['message'])
            
            if ham_text:
                ham_wordcloud = WordCloud(
                    width=400, height=300,
                    background_color='white',
                    colormap='Blues',
                    max_words=50,
                    random_state=42
                ).generate(ham_text)
                
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.imshow(ham_wordcloud, interpolation='bilinear')
                ax.axis('off')
                st.pyplot(fig)
                plt.close()
    
    def evaluate_all_models(self, df: pd.DataFrame):
        """Comprehensive evaluation of all models."""
        st.markdown('<div class="phase-header">🔍 Model Performance Analysis</div>', unsafe_allow_html=True)
        
        # Prepare test data
        X = df['message']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model_results = {}
        
        for model_name in self.models.keys():
            try:
                # Get model components
                model = self.models[model_name]
                vectorizer = self.vectorizers[model_name]
                feature_selector = self.feature_selectors.get(model_name)
                
                # Transform test data
                X_test_transformed = vectorizer.transform(X_test)
                
                if feature_selector is not None:
                    X_test_transformed = feature_selector.transform(X_test_transformed)
                
                if hasattr(X_test_transformed, 'toarray'):
                    X_test_transformed = X_test_transformed.toarray()
                
                # Make predictions
                y_pred = model.predict(X_test_transformed)
                y_scores = model.decision_function(X_test_transformed)
                
                # Convert to binary format
                y_test_binary = (y_test == 'spam').astype(int)
                y_pred_binary = (y_pred == 'spam').astype(int) if isinstance(y_pred[0], str) else y_pred
                
                # Calculate metrics
                metrics = {
                    'Accuracy': accuracy_score(y_test_binary, y_pred_binary),
                    'Precision': precision_score(y_test_binary, y_pred_binary),
                    'Recall': recall_score(y_test_binary, y_pred_binary),
                    'F1-Score': f1_score(y_test_binary, y_pred_binary)
                }
                
                # ROC and PR curves
                scaler = MinMaxScaler()
                y_scores_norm = scaler.fit_transform(y_scores.reshape(-1, 1)).flatten()
                
                fpr, tpr, _ = roc_curve(y_test_binary, y_scores_norm)
                roc_auc = auc(fpr, tpr)
                
                precision_curve, recall_curve, _ = precision_recall_curve(y_test_binary, y_scores_norm)
                pr_auc = auc(recall_curve, precision_curve)
                
                metrics['ROC AUC'] = roc_auc
                metrics['PR AUC'] = pr_auc
                
                model_results[model_name] = {
                    'metrics': metrics,
                    'y_test': y_test_binary,
                    'y_pred': y_pred_binary,
                    'y_scores': y_scores_norm,
                    'roc_data': (fpr, tpr, roc_auc),
                    'pr_data': (precision_curve, recall_curve, pr_auc)
                }
                
            except Exception as e:
                st.error(f"❌ Error evaluating {model_name}: {str(e)}")
        
        return model_results
    
    def display_model_performance(self, model_results: dict):
        """Display comprehensive model performance analysis."""
        if not model_results:
            st.warning("⚠️ No model results available")
            return
        
        # Performance metrics comparison
        st.subheader("📈 Performance Metrics Comparison")
        
        metrics_df = pd.DataFrame({
            model: results['metrics'] 
            for model, results in model_results.items()
        }).T
        
        # Display metrics table
        st.dataframe(metrics_df.round(4), use_container_width=True)
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Metrics comparison chart
            fig_metrics = go.Figure()
            
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            for metric in metrics:
                values = [model_results[model]['metrics'][metric] for model in model_results.keys()]
                fig_metrics.add_trace(go.Bar(
                    x=list(model_results.keys()),
                    y=values,
                    name=metric
                ))
            
            fig_metrics.update_layout(
                title="📊 Performance Metrics Comparison",
                yaxis_title="Score",
                height=400,
                barmode='group'
            )
            st.plotly_chart(fig_metrics, use_container_width=True)
        
        with col2:
            # ROC curves comparison
            fig_roc = go.Figure()
            
            for model_name, results in model_results.items():
                fpr, tpr, roc_auc = results['roc_data']
                fig_roc.add_trace(go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {roc_auc:.3f})',
                    line=dict(width=3)
                ))
            
            # Add diagonal line
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name='Random Classifier',
                showlegend=False
            ))
            
            fig_roc.update_layout(
                title="📈 ROC Curves Comparison",
                xaxis_title="False Positive Rate",
                yaxis_title="True Positive Rate",
                height=400
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        
        # Individual model details
        st.subheader("🔍 Individual Model Analysis")
        
        selected_model = st.selectbox(
            "Select model for detailed analysis:",
            options=list(model_results.keys())
        )
        
        if selected_model in model_results:
            results = model_results[selected_model]
            
            # Confusion matrix
            col1, col2 = st.columns(2)
            
            with col1:
                cm = confusion_matrix(results['y_test'], results['y_pred'])
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=['Ham', 'Spam'],
                    y=['Ham', 'Spam'],
                    color_continuous_scale='Blues',
                    title=f"🎯 Confusion Matrix - {selected_model}"
                )
                fig_cm.update_layout(height=400)
                st.plotly_chart(fig_cm, use_container_width=True)
            
            with col2:
                # Precision-Recall curve
                precision_curve, recall_curve, pr_auc = results['pr_data']
                
                fig_pr = go.Figure()
                fig_pr.add_trace(go.Scatter(
                    x=recall_curve, y=precision_curve,
                    mode='lines',
                    name=f'PR Curve (AUC = {pr_auc:.3f})',
                    line=dict(width=3, color=self.colors['primary'])
                ))
                
                fig_pr.update_layout(
                    title=f"📊 Precision-Recall Curve - {selected_model}",
                    xaxis_title="Recall",
                    yaxis_title="Precision",
                    height=400
                )
                st.plotly_chart(fig_pr, use_container_width=True)
            
            # Model description
            model_config = next(
                (config for config in self.model_configs if config['name'] == selected_model),
                None
            )
            if model_config:
                st.markdown(
                    f'<div class="feature-insight">'
                    f'<strong>Model Description:</strong> {model_config["description"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )
    
    def create_interactive_prediction(self):
        """Create interactive prediction interface."""
        st.markdown('<div class="phase-header">🔮 Interactive Prediction</div>', unsafe_allow_html=True)
        
        # Model selection
        if not self.models:
            st.error("❌ No models loaded. Please check model files.")
            return
        
        selected_model = st.selectbox(
            "Choose prediction model:",
            options=list(self.models.keys()),
            help="Select which model to use for prediction"
        )
        
        # Text input
        message_input = st.text_area(
            "📝 Enter SMS message to classify:",
            placeholder="Type your message here...",
            height=100,
            help="Enter the SMS message you want to classify as spam or ham"
        )
        
        if st.button("🚀 Classify Message", type="primary"):
            if message_input.strip():
                prediction, confidence, error = self.predict_message(message_input, selected_model)
                
                if error:
                    st.error(f"❌ Prediction error: {error}")
                else:
                    # Display prediction result
                    if prediction == 'spam' or (isinstance(prediction, (int, float)) and prediction == 1):
                        st.markdown(
                            f'<div class="spam-result">'
                            f'🚫 <strong>SPAM DETECTED</strong><br>'
                            f'Confidence: {confidence:.1%}<br>'
                            f'Model: {selected_model}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    else:
                        st.markdown(
                            f'<div class="ham-result">'
                            f'✅ <strong>LEGITIMATE MESSAGE</strong><br>'
                            f'Confidence: {confidence:.1%}<br>'
                            f'Model: {selected_model}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                    
                    # Additional analysis
                    st.subheader("📊 Message Analysis")
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("📏 Character Count", len(message_input))
                    
                    with col2:
                        word_count = len(message_input.split())
                        st.metric("🔤 Word Count", word_count)
                    
                    with col3:
                        exclamation_count = message_input.count('!')
                        st.metric("❗ Exclamation Marks", exclamation_count)
            else:
                st.warning("⚠️ Please enter a message to classify")
    
    def display_visualization_gallery(self):
        """Display visualization gallery from Phase 4 analysis."""
        st.markdown('<div class="phase-header">🎨 Visualization Gallery</div>', unsafe_allow_html=True)
        
        viz_dir = Path("visualizations")
        if not viz_dir.exists():
            st.warning("⚠️ Visualization directory not found. Run Phase 4 analysis first.")
            return
        
        # Interactive dashboard
        dashboard_path = viz_dir / "interactive_dashboard.html"
        if dashboard_path.exists():
            st.subheader("🌐 Interactive Dashboard")
            st.info("📊 View the comprehensive interactive dashboard with all analysis results")
            
            if st.button("🚀 Open Interactive Dashboard"):
                with open(dashboard_path, 'r', encoding='utf-8') as f:
                    dashboard_html = f.read()
                st.components.v1.html(dashboard_html, height=800, scrolling=True)
        
        # Static visualizations
        st.subheader("📊 Static Visualizations")
        
        viz_files = {
            "Dataset Distribution": "dataset_distribution_analysis.png",
            "Advanced Word Clouds": "advanced_wordclouds.png",
            "N-gram Analysis": "ngram_analysis.png",
            "Phase Comparison": "comprehensive_phase_comparison.png"
        }
        
        cols = st.columns(2)
        
        for i, (title, filename) in enumerate(viz_files.items()):
            file_path = viz_dir / filename
            if file_path.exists():
                with cols[i % 2]:
                    st.subheader(title)
                    st.image(str(file_path), use_column_width=True)
            else:
                with cols[i % 2]:
                    st.warning(f"⚠️ {title} not found")
        
        # Analysis reports
        st.subheader("📋 Analysis Reports")
        
        report_files = {
            "Comprehensive Analysis": "comprehensive_analysis_report.md",
            "Phase 4 Visualization": "phase4_visualization_report.md"
        }
        
        for title, filename in report_files.items():
            file_path = viz_dir / filename
            if file_path.exists():
                with st.expander(f"📄 {title}"):
                    with open(file_path, 'r', encoding='utf-8') as f:
                        report_content = f.read()
                    st.markdown(report_content)
    
    def run(self):
        """Main application runner."""
        # Header
        st.markdown('<h1 class="main-header">🎨 Enhanced SMS Spam Classifier</h1>', unsafe_allow_html=True)
        st.markdown("### Advanced Analytics & Multi-Phase Model Analysis")
        
        # Load models and data
        self.models, self.vectorizers, self.feature_selectors = self.load_models()
        df = self.load_data()
        
        if df is None:
            st.error("❌ Cannot proceed without dataset")
            return
        
        # Navigation menu
        selected = option_menu(
            menu_title=None,
            options=["📊 Dashboard", "🔮 Prediction", "📈 Model Analysis", "🎨 Visualizations", "ℹ️ About"],
            icons=["graph-up", "magic", "bar-chart", "palette", "info-circle"],
            menu_icon="cast",
            default_index=0,
            orientation="horizontal",
            styles={
                "container": {"padding": "0!important", "background-color": "#fafafa"},
                "icon": {"color": "#45B7D1", "font-size": "25px"},
                "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#eee"},
                "nav-link-selected": {"background-color": "#4ECDC4"},
            }
        )
        
        # Page content based on selection
        if selected == "📊 Dashboard":
            self.create_dataset_overview(df)
            
            if len(self.models) > 0:
                model_results = self.evaluate_all_models(df)
                
                st.markdown('<div class="phase-header">🏆 Quick Performance Summary</div>', unsafe_allow_html=True)
                
                if model_results:
                    # Find best model
                    best_model = max(model_results.keys(), 
                                   key=lambda x: model_results[x]['metrics']['F1-Score'])
                    best_f1 = model_results[best_model]['metrics']['F1-Score']
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("🥇 Best Model", best_model)
                    
                    with col2:
                        st.metric("📈 Best F1-Score", f"{best_f1:.4f}")
                    
                    with col3:
                        st.metric("🔧 Models Loaded", len(self.models))
        
        elif selected == "🔮 Prediction":
            self.create_interactive_prediction()
        
        elif selected == "📈 Model Analysis":
            if len(self.models) > 0:
                model_results = self.evaluate_all_models(df)
                self.display_model_performance(model_results)
            else:
                st.error("❌ No models loaded for analysis")
        
        elif selected == "🎨 Visualizations":
            self.display_visualization_gallery()
            self.create_advanced_wordclouds(df)
        
        elif selected == "ℹ️ About":
            st.markdown('<div class="phase-header">ℹ️ About This Application</div>', unsafe_allow_html=True)
            
            st.markdown("""
            ### 🎯 Enhanced SMS Spam Classifier
            
            This advanced application provides comprehensive analysis and classification of SMS messages using 
            multiple machine learning models with different optimization strategies.
            
            #### 🚀 Features:
            - **Multi-Phase Model Analysis**: Compare performance across 3 different optimization phases
            - **Interactive Prediction**: Real-time spam classification with confidence scores
            - **Advanced Visualizations**: Comprehensive charts, word clouds, and interactive dashboards
            - **Performance Analytics**: Detailed metrics, ROC curves, and confusion matrices
            - **Professional Interface**: Enhanced UI with modern styling and intuitive navigation
            
            #### 📊 Model Phases:
            1. **Phase 1 (Baseline)**: Basic SVM with TF-IDF features
            2. **Phase 2 (Optimized)**: Enhanced with feature selection and parameter tuning
            3. **Phase 3 (Advanced)**: SMOTE balancing and grid search optimization
            
            #### 🛠️ Technical Stack:
            - **Machine Learning**: scikit-learn, imbalanced-learn
            - **Visualization**: Plotly, Matplotlib, Seaborn, WordCloud
            - **Web Interface**: Streamlit
            - **Data Processing**: Pandas, NumPy
            
            #### 📈 Performance Metrics:
            - Accuracy, Precision, Recall, F1-Score
            - ROC AUC and Precision-Recall AUC
            - Confusion matrices and performance curves
            
            ---
            
            **🔬 Research & Development**: This application demonstrates the evolution of machine learning 
            models through systematic optimization and provides insights into spam detection patterns.
            """)
        
        # Footer
        st.markdown("---")
        st.markdown("🎨 Enhanced SMS Spam Classifier | Advanced Analytics Dashboard")

# Run the application
if __name__ == "__main__":
    app = EnhancedSpamClassifierApp()
    app.run()