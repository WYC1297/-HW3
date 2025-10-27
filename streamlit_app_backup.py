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
</style>
""", unsafe_allow_html=True)
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .prediction-result {
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
        text-align: center;
        font-weight: bold;
        font-size: 1.2rem;
    }
    .spam-result {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #e57373;
    }
    .ham-result {
        background-color: #e8f5e8;
        color: #2e7d32;
        border: 2px solid #81c784;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

class SMSSpamClassifierApp:
    """Main Streamlit application class for SMS spam classification."""
    
    def __init__(self):
        """Initialize the application with models and data."""
        self.setup_session_state()
        self.load_models()
        self.load_data()
    
    def setup_session_state(self):
        """Initialize session state variables."""
        if 'prediction_history' not in st.session_state:
            st.session_state.prediction_history = []
        if 'current_model' not in st.session_state:
            st.session_state.current_model = 'Phase 3 (Advanced)'
    
    @st.cache_resource
    def load_models(_self):
        """Load trained models and vectorizers."""
        models = {}
        
        # Model configurations - updated to match actual file names
        model_configs = {
            'Phase 1 (Baseline)': {
                'model_path': 'models/svm_spam_classifier.joblib',
                'vectorizer_path': 'models/tfidf_vectorizer.joblib'
            },
            'Phase 2 (Optimized)': {
                'model_path': 'models/optimized_svm_classifier.joblib',
                'vectorizer_path': 'models/optimized_tfidf_vectorizer.joblib'
            }
        }
        
        # Check for Phase 3 models with different possible names
        phase3_model_options = [
            ('models/phase3_final_svm_classifier.joblib', 'models/phase3_final_tfidf_vectorizer.joblib'),
            ('models/optimized_svm_classifier.joblib', 'models/optimized_tfidf_vectorizer.joblib')
        ]
        
        for model_path, vectorizer_path in phase3_model_options:
            if Path(model_path).exists() and Path(vectorizer_path).exists():
                # Use Phase 2 models as Phase 3 if Phase 3 specific models don't exist
                if 'optimized' in model_path:
                    model_configs['Phase 3 (Advanced)'] = {
                        'model_path': model_path,
                        'vectorizer_path': vectorizer_path,
                        'note': 'Using Phase 2 optimized models'
                    }
                else:
                    model_configs['Phase 3 (Advanced)'] = {
                        'model_path': model_path,
                        'vectorizer_path': vectorizer_path
                    }
                break
        
        # Load models
        for phase, config in model_configs.items():
            if Path(config['model_path']).exists() and Path(config['vectorizer_path']).exists():
                try:
                    models[phase] = {
                        'model': joblib.load(config['model_path']),
                        'vectorizer': joblib.load(config['vectorizer_path']),
                        'note': config.get('note', '')
                    }
                    print(f"✅ Successfully loaded {phase}")
                except Exception as e:
                    print(f"❌ Could not load {phase} model: {e}")
            else:
                print(f"⚠️  Model files not found for {phase}")
                print(f"   Model: {config['model_path']} - exists: {Path(config['model_path']).exists()}")
                print(f"   Vectorizer: {config['vectorizer_path']} - exists: {Path(config['vectorizer_path']).exists()}")
        
        return models
    
    @st.cache_data
    def load_data(_self):
        """Load and cache the SMS dataset."""
        try:
            data_loader = SMSDataLoader()
            df = data_loader.load_data()
            if df is not None:
                df = data_loader.basic_clean()
                return df
        except Exception as e:
            st.error(f"Error loading data: {e}")
        return None
    
    def predict_text(self, text, model_name):
        """Predict whether a text message is spam or ham."""
        if model_name not in self.models:
            return None, None, "Model not available"
        
        try:
            model = self.models[model_name]['model']
            vectorizer = self.models[model_name]['vectorizer']
            
            # Vectorize the text
            text_vectorized = vectorizer.transform([text])
            
            # Apply feature selection if available
            if 'feature_selector' in self.models[model_name]:
                feature_selector = self.models[model_name]['feature_selector']
                text_vectorized = feature_selector.transform(text_vectorized)
            
            # Handle sparse matrix compatibility
            if hasattr(text_vectorized, 'toarray'):
                text_vectorized = text_vectorized.toarray()
            
            # Make prediction
            prediction = model.predict(text_vectorized)[0]
            
            # Get confidence score (decision function)
            confidence_score = model.decision_function(text_vectorized)[0]
            
            # Convert to probability-like score
            probability = 1 / (1 + np.exp(-confidence_score))  # Sigmoid transformation
            
            # Ensure prediction is in the right format (string)
            if hasattr(prediction, 'dtype') and prediction.dtype in ['int32', 'int64']:
                prediction = 'ham' if prediction == 0 else 'spam'
            
            return prediction, probability, None
            
        except Exception as e:
            return None, None, str(e)
    
    def render_home_page(self):
        """Render the main home page."""
        st.markdown('<h1 class="main-header">📱 SMS Spam Classifier</h1>', unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
        <h3>🎯 Project Overview</h3>
        <p>This is an advanced SMS spam classification system that uses machine learning to automatically 
        detect spam messages. The system has been developed through multiple optimization phases, 
        achieving high precision and recall rates for accurate spam detection.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Model performance metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
            <h4>📊 Phase 3 Performance</h4>
            <ul>
            <li><strong>Accuracy:</strong> 98.84%</li>
            <li><strong>Precision:</strong> 93.08%</li>
            <li><strong>Recall:</strong> 94.53%</li>
            <li><strong>F1-Score:</strong> 93.80%</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
            <h4>🔧 Technical Features</h4>
            <ul>
            <li>Advanced TF-IDF vectorization</li>
            <li>SMOTE balancing</li>
            <li>SVM with RBF kernel</li>
            <li>Optimized thresholds</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
            <h4>📈 Dataset Statistics</h4>
            <ul>
            <li><strong>Total Messages:</strong> 5,160</li>
            <li><strong>Ham Messages:</strong> 4,507 (87.3%)</li>
            <li><strong>Spam Messages:</strong> 653 (12.7%)</li>
            <li><strong>Avg Spam Length:</strong> 159.6 chars</li>
            </ul>
            </div>
            """, unsafe_allow_html=True)
        
        # Available Models Section
        st.markdown("### 🤖 Available Models")
        
        # Display loaded models
        if hasattr(self, 'models') and self.models:
            available_models = []
            for phase_name, model_info in self.models.items():
                status = "✅ Ready"
                note = model_info.get('note', '')
                if note:
                    status += f" ({note})"
                available_models.append(f"**{phase_name}**: {status}")
            
            st.markdown("\n".join(available_models))
        else:
            st.warning("⚠️ No models loaded. Please check model files in the models/ directory.")
        
        # Model comparison table
        if hasattr(self, 'models') and len(self.models) > 1:
            st.markdown("#### 📊 Quick Performance Comparison")
            
            performance_data = {
                'Phase 1 (Baseline)': {'Accuracy': '98.16%', 'Precision': '100.00%', 'Recall': '85.16%', 'F1-Score': '91.98%'},
                'Phase 2 (Optimized)': {'Accuracy': '98.45%', 'Precision': '99.12%', 'Recall': '88.28%', 'F1-Score': '93.39%'},
                'Phase 3 (Advanced)': {'Accuracy': '98.84%', 'Precision': '93.08%', 'Recall': '94.53%', 'F1-Score': '93.80%'}
            }
            
            # Filter to only show available models
            available_performance = {k: v for k, v in performance_data.items() if k in self.models}
            
            if available_performance:
                df_performance = pd.DataFrame(available_performance).T
                st.dataframe(df_performance, use_container_width=True)
        
        # Quick prediction section
        st.markdown("### 🚀 Quick Prediction Test")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            sample_text = st.text_area(
                "Enter a text message to classify:",
                placeholder="Type your message here...",
                height=100
            )
        
        with col2:
            model_choice = st.selectbox(
                "Select Model:",
                list(self.models.keys()) if hasattr(self, 'models') else ['Phase 3 (Advanced)'],
                index=2 if len(self.models) >= 3 else 0
            )
        
        if st.button("🔍 Classify Message", type="primary"):
            if sample_text.strip():
                self.perform_quick_prediction(sample_text, model_choice)
            else:
                st.warning("Please enter a message to classify.")
    
    def perform_quick_prediction(self, text, model_name):
        """Perform and display quick prediction."""
        prediction, probability, error = self.predict_text(text, model_name)
        
        if error:
            st.error(f"Prediction error: {error}")
            return
        
        # Display result
        if prediction == 'spam':
            st.markdown(f"""
            <div class="prediction-result spam-result">
            🚨 SPAM DETECTED<br>
            Confidence: {probability:.2%}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-result ham-result">
            ✅ LEGITIMATE MESSAGE<br>
            Confidence: {(1-probability):.2%}
            </div>
            """, unsafe_allow_html=True)
        
        # Show model info
        st.info(f"Prediction made using: **{model_name}**")
    
    def render_classifier_page(self):
        """Render the main classification interface."""
        st.header("📝 SMS Spam Classifier")
        
        # Model selection
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Select Classification Model")
            
        with col2:
            selected_model = st.selectbox(
                "Model Version:",
                list(self.models.keys()) if hasattr(self, 'models') else ['Phase 3 (Advanced)'],
                key="model_selector"
            )
        
        # Display model information
        model_info = {
            'Phase 1 (Baseline)': "SVM with basic TF-IDF vectorization",
            'Phase 2 (Optimized)': "Enhanced preprocessing and feature engineering",
            'Phase 3 (Advanced)': "SMOTE balancing, optimized TF-IDF, threshold tuning"
        }
        
        if selected_model in model_info:
            st.info(f"**{selected_model}**: {model_info[selected_model]}")
        
        # Text input section
        st.subheader("📱 Enter Message to Classify")
        
        # Text input methods
        input_method = st.radio(
            "Input Method:",
            ["Type Message", "Use Sample Messages"],
            horizontal=True
        )
        
        if input_method == "Type Message":
            user_text = st.text_area(
                "Message Content:",
                placeholder="Enter the SMS message you want to classify...",
                height=150,
                key="user_input_text"
            )
        else:
            # Sample messages
            sample_messages = {
                "Ham Sample 1": "Hey, are we still meeting for lunch tomorrow?",
                "Ham Sample 2": "Thanks for the birthday wishes! Had a great time at the party.",
                "Ham Sample 3": "Can you pick up milk on your way home? Thanks!",
                "Spam Sample 1": "CONGRATULATIONS! You've won a £1000 cash prize! Call now to claim your reward!",
                "Spam Sample 2": "URGENT! Your account will be suspended. Click here to verify: www.fakebank.com",
                "Spam Sample 3": "FREE ringtones! Send STOP to 85555 to opt out. Standard rates apply."
            }
            
            selected_sample = st.selectbox("Choose a sample message:", list(sample_messages.keys()))
            user_text = sample_messages[selected_sample]
            st.text_area("Selected Message:", value=user_text, height=100, disabled=True)
        
        # Classification section
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            classify_button = st.button(
                "🔍 Classify Message",
                type="primary",
                use_container_width=True
            )
        
        if classify_button and user_text.strip():
            self.process_classification(user_text, selected_model)
        elif classify_button:
            st.warning("⚠️ Please enter a message to classify.")
        
        # Prediction history
        self.display_prediction_history()
    
    def process_classification(self, text, model_name):
        """Process text classification and display results."""
        with st.spinner("🤖 Analyzing message..."):
            prediction, probability, error = self.predict_text(text, model_name)
        
        if error:
            st.error(f"❌ Classification Error: {error}")
            return
        
        # Display main result
        st.subheader("🎯 Classification Result")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if prediction == 'spam':
                st.markdown(f"""
                <div class="prediction-result spam-result">
                🚨 SPAM DETECTED
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-result ham-result">
                ✅ LEGITIMATE MESSAGE (HAM)
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            # Confidence visualization
            confidence = probability if prediction == 'spam' else (1 - probability)
            
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = confidence * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Confidence Score"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "#dc3545" if prediction == 'spam' else "#28a745"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "lightgreen"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed analysis
        st.subheader("📊 Detailed Analysis")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Prediction", prediction.upper())
        with col2:
            st.metric("Confidence", f"{confidence:.2%}")
        with col3:
            st.metric("Message Length", f"{len(text)} chars")
        with col4:
            st.metric("Word Count", f"{len(text.split())} words")
        
        # Add to history
        self.add_to_history(text, prediction, confidence, model_name)
        
        # Text analysis
        self.display_text_analysis(text)
    
    def add_to_history(self, text, prediction, confidence, model_name):
        """Add prediction to session history."""
        history_entry = {
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'text': text[:100] + "..." if len(text) > 100 else text,
            'prediction': prediction,
            'confidence': confidence,
            'model': model_name
        }
        
        st.session_state.prediction_history.insert(0, history_entry)
        
        # Keep only last 10 predictions
        if len(st.session_state.prediction_history) > 10:
            st.session_state.prediction_history = st.session_state.prediction_history[:10]
    
    def display_prediction_history(self):
        """Display prediction history."""
        if st.session_state.prediction_history:
            st.subheader("📚 Recent Predictions")
            
            for i, entry in enumerate(st.session_state.prediction_history[:5]):
                with st.expander(f"{entry['timestamp']} - {entry['prediction'].upper()} ({entry['confidence']:.1%})"):
                    st.write(f"**Text:** {entry['text']}")
                    st.write(f"**Model:** {entry['model']}")
                    st.write(f"**Confidence:** {entry['confidence']:.2%}")
    
    def display_text_analysis(self, text):
        """Display detailed text analysis."""
        st.subheader("🔍 Text Analysis")
        
        # Basic statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Character Analysis:**")
            st.write(f"- Total characters: {len(text)}")
            st.write(f"- Alphabetic characters: {sum(c.isalpha() for c in text)}")
            st.write(f"- Numeric characters: {sum(c.isdigit() for c in text)}")
            st.write(f"- Special characters: {sum(not c.isalnum() and not c.isspace() for c in text)}")
        
        with col2:
            st.write("**Word Analysis:**")
            words = text.split()
            st.write(f"- Total words: {len(words)}")
            st.write(f"- Average word length: {np.mean([len(w) for w in words]):.1f}")
            st.write(f"- Uppercase words: {sum(w.isupper() for w in words)}")
            st.write(f"- Contains URLs: {'Yes' if re.search(r'http[s]?://|www\.', text) else 'No'}")
        
        # Suspicious patterns
        suspicious_patterns = {
            'Multiple exclamations': len(re.findall(r'!{2,}', text)),
            'All caps words': len(re.findall(r'\b[A-Z]{3,}\b', text)),
            'Numbers in text': len(re.findall(r'\d+', text)),
            'Money symbols': len(re.findall(r'[$£€¥]', text)),
            'Phone numbers': len(re.findall(r'\b\d{3,4}[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b', text))
        }
        
        if any(suspicious_patterns.values()):
            st.write("**Potential Spam Indicators:**")
            for pattern, count in suspicious_patterns.items():
                if count > 0:
                    st.write(f"- {pattern}: {count}")
    
    def render_dashboard_page(self):
        """Render the model performance dashboard."""
        st.header("📊 Model Performance Dashboard")
        
        # Dashboard navigation tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📈 Performance Overview", 
            "🎯 Model Analysis", 
            "🔍 Token Analysis",
            "📋 Confusion Matrix",
            "⚙️ Threshold Analysis"
        ])
        
        with tab1:
            self.render_performance_overview()
        
        with tab2:
            self.render_model_analysis()
            
        with tab3:
            self.render_token_frequency_analysis()
            
        with tab4:
            self.render_confusion_matrix_analysis()
            
        with tab5:
            self.render_threshold_analysis()
    
    def render_performance_overview(self):
        """Render the performance overview tab."""
        st.subheader("🏆 Model Performance Comparison")
        
        # Load historical results
        historical_results = {
            'Phase 1 (Baseline)': {
                'Accuracy': 0.9816, 'Precision': 1.0000, 'Recall': 0.8516, 'F1-Score': 0.9198
            },
            'Phase 2 (Optimized)': {
                'Accuracy': 0.9845, 'Precision': 0.9912, 'Recall': 0.8828, 'F1-Score': 0.9339
            },
            'Phase 3 (Advanced)': {
                'Accuracy': 0.9884, 'Precision': 0.9308, 'Recall': 0.9453, 'F1-Score': 0.9380
            }
        }
        
        # Performance comparison
        df_metrics = pd.DataFrame(historical_results).T
        
        # Interactive bar chart
        fig = px.bar(
            df_metrics.reset_index(), 
            x='index', 
            y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            title="Performance Metrics by Phase",
            labels={'index': 'Model Phase', 'value': 'Score'},
            barmode='group'
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance evolution line chart
        st.subheader("📈 Performance Evolution")
        
        fig_line = px.line(
            df_metrics.reset_index(), 
            x='index', 
            y=['Accuracy', 'Precision', 'Recall', 'F1-Score'],
            title="Performance Trends Across Development Phases",
            markers=True
        )
        fig_line.update_layout(height=400)
        st.plotly_chart(fig_line, use_container_width=True)
        
        # Detailed metrics table
        st.subheader("📋 Detailed Metrics")
        
        styled_df = df_metrics.style.format("{:.4f}").background_gradient(
            cmap='RdYlGn', vmin=0.85, vmax=1.0
        )
        st.dataframe(styled_df, use_container_width=True)
    
    def render_model_analysis(self):
        """Render detailed model analysis with ROC and PR curves."""
        st.subheader("🎯 Detailed Model Analysis")
        
        # Model selection for analysis
        selected_model = st.selectbox(
            "Select model for detailed analysis:",
            list(self.models.keys()) if hasattr(self, 'models') else ['Phase 3 (Advanced)'],
            key="analysis_model_selector"
        )
        
        if selected_model not in self.models:
            st.warning(f"Model {selected_model} not available for analysis.")
            return
        
        # Get test data
        if self.data is None:
            st.error("Data not available for analysis.")
            return
        
        # Prepare test data
        X = self.data['message']
        y = self.data['label']
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Load model and vectorizer
        model = self.models[selected_model]['model']
        vectorizer = self.models[selected_model]['vectorizer']
        
        try:
            # Transform test data
            X_test_transformed = vectorizer.transform(X_test)
            
            # Apply feature selection if available
            if 'feature_selector' in self.models[selected_model]:
                feature_selector = self.models[selected_model]['feature_selector']
                X_test_transformed = feature_selector.transform(X_test_transformed)
            
            if hasattr(X_test_transformed, 'toarray'):
                X_test_transformed = X_test_transformed.toarray()
            
            # Make predictions
            y_pred = model.predict(X_test_transformed)
            y_scores = model.decision_function(X_test_transformed)
            
            # Convert labels to binary for metrics calculation
            y_test_binary = (y_test == 'spam').astype(int)
            
            # Handle prediction format - convert to binary if needed
            if hasattr(y_pred, 'dtype') and y_pred.dtype in ['int32', 'int64']:
                y_pred_binary = y_pred
            else:
                y_pred_binary = (y_pred == 'spam').astype(int)
            
            # Create ROC and PR curves
            col1, col2 = st.columns(2)
            
            with col1:
                self.plot_roc_curve(y_test_binary, y_scores, selected_model)
            
            with col2:
                self.plot_precision_recall_curve(y_test_binary, y_scores, selected_model)
            
            # Performance metrics
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(y_test_binary, y_pred_binary)
            precision = precision_score(y_test_binary, y_pred_binary)
            recall = recall_score(y_test_binary, y_pred_binary)
            f1 = f1_score(y_test_binary, y_pred_binary)
            
            st.subheader(f"📊 {selected_model} Test Performance")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Accuracy", f"{accuracy:.4f}")
            with col2:
                st.metric("Precision", f"{precision:.4f}")
            with col3:
                st.metric("Recall", f"{recall:.4f}")
            with col4:
                st.metric("F1-Score", f"{f1:.4f}")
                
        except Exception as e:
            st.error(f"Error in model analysis: {e}")
    
    def plot_roc_curve(self, y_true, y_scores, model_name):
        """Plot ROC curve."""
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        fig = go.Figure()
        
        # ROC curve
        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode='lines',
            name=f'ROC Curve (AUC = {roc_auc:.3f})',
            line=dict(color='blue', width=2)
        ))
        
        # Random classifier line
        fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            name='Random Classifier',
            line=dict(color='red', dash='dash', width=1)
        ))
        
        fig.update_layout(
            title=f'ROC Curve - {model_name}',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            width=400,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def plot_precision_recall_curve(self, y_true, y_scores, model_name):
        """Plot Precision-Recall curve."""
        from sklearn.metrics import precision_recall_curve, auc
        
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        fig = go.Figure()
        
        # PR curve
        fig.add_trace(go.Scatter(
            x=recall, y=precision,
            mode='lines',
            name=f'PR Curve (AUC = {pr_auc:.3f})',
            line=dict(color='green', width=2)
        ))
        
        # Baseline (random classifier for imbalanced data)
        baseline = y_true.mean()
        fig.add_hline(
            y=baseline, 
            line_dash="dash", 
            line_color="red",
            annotation_text=f"Baseline ({baseline:.3f})"
        )
        
        fig.update_layout(
            title=f'Precision-Recall Curve - {model_name}',
            xaxis_title='Recall',
            yaxis_title='Precision',
            width=400,
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_token_frequency_analysis(self):
        """Render token frequency analysis for spam vs ham."""
        st.subheader("🔍 Top Token Frequency Analysis")
        
        if self.data is None:
            st.error("Data not available for token analysis.")
            return
        
        # Token analysis parameters
        col1, col2 = st.columns(2)
        
        with col1:
            top_n = st.slider("Number of top tokens to display:", 10, 50, 20)
        
        with col2:
            analysis_type = st.selectbox(
                "Analysis type:",
                ["Unigrams", "Bigrams", "Trigrams"]
            )
        
        # Prepare data
        ham_text = ' '.join(self.data[self.data['label'] == 'ham']['message'].astype(str))
        spam_text = ' '.join(self.data[self.data['label'] == 'spam']['message'].astype(str))
        
        # Extract tokens based on type
        if analysis_type == "Unigrams":
            ham_tokens = self.extract_unigrams(ham_text)
            spam_tokens = self.extract_unigrams(spam_text)
        elif analysis_type == "Bigrams":
            ham_tokens = self.extract_bigrams(ham_text)
            spam_tokens = self.extract_bigrams(spam_text)
        else:  # Trigrams
            ham_tokens = self.extract_trigrams(ham_text)
            spam_tokens = self.extract_trigrams(spam_text)
        
        # Create comparison charts
        col1, col2 = st.columns(2)
        
        with col1:
            self.plot_token_frequency(ham_tokens, f"Top {top_n} {analysis_type} in Ham Messages", 
                                    top_n, color='#28a745')
        
        with col2:
            self.plot_token_frequency(spam_tokens, f"Top {top_n} {analysis_type} in Spam Messages", 
                                    top_n, color='#dc3545')
        
        # Comparative analysis
        st.subheader("📊 Comparative Token Analysis")
        
        # Find tokens unique to each class
        ham_top_set = set(dict(ham_tokens.most_common(top_n)).keys())
        spam_top_set = set(dict(spam_tokens.most_common(top_n)).keys())
        
        unique_ham = ham_top_set - spam_top_set
        unique_spam = spam_top_set - ham_top_set
        common_tokens = ham_top_set & spam_top_set
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**🟢 Unique to Ham:**")
            for token in list(unique_ham)[:10]:
                st.write(f"• {token}")
        
        with col2:
            st.markdown("**🔴 Unique to Spam:**")
            for token in list(unique_spam)[:10]:
                st.write(f"• {token}")
        
        with col3:
            st.markdown("**🟡 Common Tokens:**")
            for token in list(common_tokens)[:10]:
                st.write(f"• {token}")
    
    def extract_unigrams(self, text):
        """Extract unigrams from text."""
        import re
        from collections import Counter
        
        words = re.findall(r'\b\w+\b', text.lower())
        return Counter(words)
    
    def extract_bigrams(self, text):
        """Extract bigrams from text."""
        import re
        from collections import Counter
        
        words = re.findall(r'\b\w+\b', text.lower())
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        return Counter(bigrams)
    
    def extract_trigrams(self, text):
        """Extract trigrams from text."""
        import re
        from collections import Counter
        
        words = re.findall(r'\b\w+\b', text.lower())
        trigrams = [f"{words[i]} {words[i+1]} {words[i+2]}" for i in range(len(words)-2)]
        return Counter(trigrams)
    
    def plot_token_frequency(self, token_counter, title, top_n, color):
        """Plot token frequency chart."""
        top_tokens = dict(token_counter.most_common(top_n))
        
        fig = px.bar(
            x=list(top_tokens.values()),
            y=list(top_tokens.keys()),
            orientation='h',
            title=title,
            labels={'x': 'Frequency', 'y': 'Tokens'},
            color_discrete_sequence=[color]
        )
        
        fig.update_layout(
            height=600,
            yaxis={'categoryorder': 'total ascending'}
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_confusion_matrix_analysis(self):
        """Render confusion matrix analysis."""
        st.subheader("📋 Confusion Matrix Analysis")
        
        # Model selection
        selected_model = st.selectbox(
            "Select model for confusion matrix:",
            list(self.models.keys()) if hasattr(self, 'models') else ['Phase 3 (Advanced)'],
            key="confusion_model_selector"
        )
        
        if selected_model not in self.models:
            st.warning(f"Model {selected_model} not available.")
            return
        
        if self.data is None:
            st.error("Data not available for analysis.")
            return
        
        # Prepare test data
        X = self.data['message']
        y = self.data['label']
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Load model and make predictions
        model = self.models[selected_model]['model']
        vectorizer = self.models[selected_model]['vectorizer']
        
        try:
            # Transform and predict
            X_test_transformed = vectorizer.transform(X_test)
            
            # Apply feature selection if available
            if 'feature_selector' in self.models[selected_model]:
                feature_selector = self.models[selected_model]['feature_selector']
                X_test_transformed = feature_selector.transform(X_test_transformed)
            
            if hasattr(X_test_transformed, 'toarray'):
                X_test_transformed = X_test_transformed.toarray()
            
            y_pred = model.predict(X_test_transformed)
            
            # Ensure consistent label format
            # Convert predictions to string labels if they are numeric
            if hasattr(y_pred, 'dtype') and y_pred.dtype in ['int32', 'int64']:
                y_pred_labels = ['ham' if pred == 0 else 'spam' for pred in y_pred]
            else:
                y_pred_labels = y_pred
            
            # Create confusion matrix with consistent string labels
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_test, y_pred_labels, labels=['ham', 'spam'])
            
            # Plot confusion matrix
            fig = px.imshow(
                cm,
                text_auto=True,
                aspect="auto",
                title=f"Confusion Matrix - {selected_model}",
                labels=dict(x="Predicted", y="Actual"),
                x=['Ham', 'Spam'],
                y=['Ham', 'Spam'],
                color_continuous_scale='Blues'
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Confusion matrix statistics
            tn, fp, fn, tp = cm.ravel()
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("True Negatives (TN)", tn)
                st.caption("Correctly classified Ham")
            
            with col2:
                st.metric("False Positives (FP)", fp)
                st.caption("Ham classified as Spam")
            
            with col3:
                st.metric("False Negatives (FN)", fn)
                st.caption("Spam classified as Ham")
            
            with col4:
                st.metric("True Positives (TP)", tp)
                st.caption("Correctly classified Spam")
            
            # Additional metrics
            st.subheader("📊 Derived Metrics")
            
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Specificity", f"{specificity:.4f}")
                st.caption("True Negative Rate")
            
            with col2:
                st.metric("Sensitivity (Recall)", f"{sensitivity:.4f}")
                st.caption("True Positive Rate")
            
            with col3:
                st.metric("Precision", f"{precision:.4f}")
                st.caption("Positive Predictive Value")
            
            with col4:
                st.metric("NPV", f"{npv:.4f}")
                st.caption("Negative Predictive Value")
                
        except Exception as e:
            st.error(f"Error in confusion matrix analysis: {e}")
    
    def render_threshold_analysis(self):
        """Render threshold sweep analysis."""
        st.subheader("⚙️ Decision Threshold Analysis")
        
        # Model selection
        selected_model = st.selectbox(
            "Select model for threshold analysis:",
            list(self.models.keys()) if hasattr(self, 'models') else ['Phase 3 (Advanced)'],
            key="threshold_model_selector"
        )
        
        if selected_model not in self.models:
            st.warning(f"Model {selected_model} not available.")
            return
        
        if self.data is None:
            st.error("Data not available for analysis.")
            return
        
        # Threshold range
        col1, col2, col3 = st.columns(3)
        
        with col1:
            min_threshold = st.number_input("Min Threshold:", -3.0, 3.0, -2.0, 0.1)
        with col2:
            max_threshold = st.number_input("Max Threshold:", -3.0, 3.0, 2.0, 0.1)
        with col3:
            num_points = st.number_input("Number of Points:", 10, 100, 50)
        
        # Prepare test data
        X = self.data['message']
        y = self.data['label']
        
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Load model and get scores
        model = self.models[selected_model]['model']
        vectorizer = self.models[selected_model]['vectorizer']
        
        try:
            # Transform test data
            X_test_transformed = vectorizer.transform(X_test)
            
            # Apply feature selection if available
            if 'feature_selector' in self.models[selected_model]:
                feature_selector = self.models[selected_model]['feature_selector']
                X_test_transformed = feature_selector.transform(X_test_transformed)
            
            if hasattr(X_test_transformed, 'toarray'):
                X_test_transformed = X_test_transformed.toarray()
            
            # Get decision scores
            y_scores = model.decision_function(X_test_transformed)
            y_test_binary = (y_test == 'spam').astype(int)
            
            # Threshold sweep
            thresholds = np.linspace(min_threshold, max_threshold, num_points)
            results = []
            
            progress_bar = st.progress(0)
            
            for i, threshold in enumerate(thresholds):
                y_pred_threshold = (y_scores >= threshold).astype(int)
                
                from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
                
                precision = precision_score(y_test_binary, y_pred_threshold, zero_division=0)
                recall = recall_score(y_test_binary, y_pred_threshold, zero_division=0)
                f1 = f1_score(y_test_binary, y_pred_threshold, zero_division=0)
                accuracy = accuracy_score(y_test_binary, y_pred_threshold)
                
                results.append({
                    'Threshold': threshold,
                    'Precision': precision,
                    'Recall': recall,
                    'F1-Score': f1,
                    'Accuracy': accuracy
                })
                
                progress_bar.progress((i + 1) / len(thresholds))
            
            progress_bar.empty()
            
            # Convert to DataFrame
            df_results = pd.DataFrame(results)
            
            # Plot threshold sweep
            fig = px.line(
                df_results,
                x='Threshold',
                y=['Precision', 'Recall', 'F1-Score', 'Accuracy'],
                title=f"Threshold Sweep Analysis - {selected_model}",
                labels={'value': 'Score', 'variable': 'Metric'}
            )
            
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Optimal thresholds
            st.subheader("🎯 Optimal Thresholds")
            
            col1, col2, col3, col4 = st.columns(4)
            
            # Find optimal thresholds
            max_f1_idx = df_results['F1-Score'].idxmax()
            max_precision_idx = df_results['Precision'].idxmax()
            max_recall_idx = df_results['Recall'].idxmax()
            max_accuracy_idx = df_results['Accuracy'].idxmax()
            
            with col1:
                st.metric(
                    "Best F1-Score",
                    f"{df_results.loc[max_f1_idx, 'F1-Score']:.4f}",
                    f"Threshold: {df_results.loc[max_f1_idx, 'Threshold']:.3f}"
                )
            
            with col2:
                st.metric(
                    "Best Precision",
                    f"{df_results.loc[max_precision_idx, 'Precision']:.4f}",
                    f"Threshold: {df_results.loc[max_precision_idx, 'Threshold']:.3f}"
                )
            
            with col3:
                st.metric(
                    "Best Recall",
                    f"{df_results.loc[max_recall_idx, 'Recall']:.4f}",
                    f"Threshold: {df_results.loc[max_recall_idx, 'Threshold']:.3f}"
                )
            
            with col4:
                st.metric(
                    "Best Accuracy",
                    f"{df_results.loc[max_accuracy_idx, 'Accuracy']:.4f}",
                    f"Threshold: {df_results.loc[max_accuracy_idx, 'Threshold']:.3f}"
                )
            
            # Threshold table
            st.subheader("📊 Threshold Sweep Table")
            
            # Format and display table
            display_df = df_results.copy()
            display_df = display_df.round(4)
            
            # Highlight best values
            styled_df = display_df.style.highlight_max(subset=['Precision', 'Recall', 'F1-Score', 'Accuracy'])
            
            st.dataframe(styled_df, use_container_width=True)
            
            # Download button for results
            csv = df_results.to_csv(index=False)
            st.download_button(
                label="Download threshold analysis results",
                data=csv,
                file_name=f"threshold_analysis_{selected_model.replace(' ', '_')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"Error in threshold analysis: {e}")
    
    def render_data_explorer_page(self):
        """Render the data exploration page."""
        st.header("🔍 Data Explorer")
        
        if self.data is None:
            st.error("Data not available. Please check data loading.")
            return
        
        # Dataset overview
        st.subheader("📊 Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Messages", len(self.data))
        with col2:
            ham_count = len(self.data[self.data['label'] == 'ham'])
            st.metric("Ham Messages", ham_count)
        with col3:
            spam_count = len(self.data[self.data['label'] == 'spam'])
            st.metric("Spam Messages", spam_count)
        with col4:
            spam_percentage = (spam_count / len(self.data)) * 100
            st.metric("Spam Percentage", f"{spam_percentage:.1f}%")
        
        # Class distribution pie chart
        st.subheader("🥧 Class Distribution")
        
        labels = self.data['label'].value_counts()
        fig_pie = px.pie(
            values=labels.values, 
            names=labels.index,
            title="Ham vs Spam Distribution",
            color_discrete_map={'ham': '#28a745', 'spam': '#dc3545'}
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Message length analysis
        st.subheader("📏 Message Length Analysis")
        
        self.data['message_length'] = self.data['message'].str.len()
        self.data['word_count'] = self.data['message'].str.split().str.len()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                self.data, 
                x='message_length', 
                color='label',
                title="Message Length Distribution",
                nbins=50,
                color_discrete_map={'ham': '#28a745', 'spam': '#dc3545'}
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_box = px.box(
                self.data, 
                x='label', 
                y='word_count',
                title="Word Count by Message Type",
                color_discrete_map={'ham': '#28a745', 'spam': '#dc3545'}
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Sample messages
        st.subheader("📱 Sample Messages")
        
        sample_type = st.selectbox("Select message type:", ["ham", "spam"])
        sample_count = st.slider("Number of samples:", 1, 10, 5)
        
        sample_messages = self.data[self.data['label'] == sample_type].sample(n=sample_count)
        
        for i, (_, row) in enumerate(sample_messages.iterrows(), 1):
            with st.expander(f"Sample {i} - {len(row['message'])} characters"):
                st.write(row['message'])
        
        # Word frequency analysis
        st.subheader("📝 Word Frequency Analysis")
        
        analysis_type = st.selectbox("Analysis type:", ["Most common words", "Spam indicators", "Ham indicators"])
        
        if analysis_type == "Most common words":
            from collections import Counter
            
            all_text = ' '.join(self.data['message']).lower()
            words = re.findall(r'\b\w+\b', all_text)
            word_freq = Counter(words).most_common(20)
            
            df_words = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            
            fig_words = px.bar(
                df_words, 
                x='Frequency', 
                y='Word',
                orientation='h',
                title="Top 20 Most Common Words"
            )
            fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_words, use_container_width=True)
        
        elif analysis_type in ["Spam indicators", "Ham indicators"]:
            label_type = 'spam' if 'Spam' in analysis_type else 'ham'
            
            # Simple word frequency for selected type
            text_subset = ' '.join(self.data[self.data['label'] == label_type]['message']).lower()
            words = re.findall(r'\b\w+\b', text_subset)
            word_freq = Counter(words).most_common(15)
            
            df_words = pd.DataFrame(word_freq, columns=['Word', 'Frequency'])
            color = '#dc3545' if label_type == 'spam' else '#28a745'
            
            fig_words = px.bar(
                df_words, 
                x='Frequency', 
                y='Word',
                orientation='h',
                title=f"Top Words in {label_type.title()} Messages",
                color_discrete_sequence=[color]
            )
            fig_words.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_words, use_container_width=True)
    
    def run(self):
        """Run the main Streamlit application."""
        # Load models and data
        self.models = self.load_models()
        self.data = self.load_data()
        
        # Sidebar navigation
        with st.sidebar:
            st.image("https://via.placeholder.com/200x100/1f77b4/white?text=SMS+Classifier", width=200)
            
            selected = option_menu(
                menu_title="Navigation",
                options=["Home", "Classifier", "Dashboard", "Data Explorer", "About"],
                icons=["house", "search", "bar-chart", "database", "info-circle"],
                menu_icon="cast",
                default_index=0,
                styles={
                    "container": {"padding": "0!important", "background-color": "#fafafa"},
                    "icon": {"color": "#1f77b4", "font-size": "18px"},
                    "nav-link": {
                        "font-size": "16px",
                        "text-align": "left",
                        "margin": "0px",
                        "--hover-color": "#eee",
                    },
                    "nav-link-selected": {"background-color": "#1f77b4"},
                },
            )
        
        # Main content routing
        if selected == "Home":
            self.render_home_page()
        elif selected == "Classifier":
            self.render_classifier_page()
        elif selected == "Dashboard":
            self.render_dashboard_page()
        elif selected == "Data Explorer":
            self.render_data_explorer_page()
        elif selected == "About":
            self.render_about_page()
        
        # Footer
        st.markdown("---")
        st.markdown(
            "<div style='text-align: center; color: #666;'>"
            "SMS Spam Classifier v1.0 | Built with Streamlit | "
            f"Last updated: {datetime.now().strftime('%Y-%m-%d')}"
            "</div>",
            unsafe_allow_html=True
        )
    
    def render_about_page(self):
        """Render the about page with project information."""
        st.header("ℹ️ About SMS Spam Classifier")
        
        st.markdown("""
        ### 🎯 Project Overview
        
        This SMS Spam Classifier is an advanced machine learning application designed to automatically
        detect spam messages with high accuracy. The project demonstrates end-to-end ML development
        from data preprocessing to web deployment.
        
        ### 🚀 Development Phases
        
        **Phase 0: Environment Setup**
        - Python virtual environment configuration
        - Dependency management and documentation
        
        **Phase 1: SVM Baseline**
        - Basic TF-IDF vectorization
        - Support Vector Machine implementation
        - Initial performance benchmarking
        
        **Phase 2: Text Preprocessing Optimization**
        - Advanced text cleaning and preprocessing
        - Feature engineering and selection
        - Hyperparameter optimization
        
        **Phase 3: Advanced Model Optimization**
        - SMOTE for class imbalance handling
        - Advanced TF-IDF parameter tuning
        - Decision threshold optimization
        - Target: Precision & Recall > 93% ✅
        
        **Phase 4: Data Visualization**
        - Comprehensive performance analysis
        - Data pattern visualization
        - Professional reporting
        
        **Phase 5: Streamlit Web Interface** *(Current)*
        - Interactive web application
        - Real-time classification
        - Performance dashboards
        
        ### 🛠️ Technical Stack
        
        - **Machine Learning**: scikit-learn, pandas, numpy
        - **Text Processing**: NLTK, TF-IDF vectorization
        - **Visualization**: matplotlib, seaborn, plotly
        - **Web Framework**: Streamlit
        - **Data Handling**: pandas, joblib
        
        ### 📊 Key Achievements
        
        - **High Performance**: 98.84% accuracy, 93.08% precision, 94.53% recall
        - **Balanced Metrics**: Successfully balanced precision-recall trade-off
        - **Production Ready**: Complete web interface for real-time classification
        - **Comprehensive Analysis**: Full visualization and reporting suite
        
        ### 📈 Model Performance
        
        | Phase | Accuracy | Precision | Recall | F1-Score |
        |-------|----------|-----------|--------|----------|
        | Phase 1 | 98.16% | 100.00% | 85.16% | 91.98% |
        | Phase 2 | 98.45% | 99.12% | 88.28% | 93.39% |
        | Phase 3 | 98.84% | 93.08% | 94.53% | 93.80% |
        
        ### 🔧 Features
        
        - **Real-time Classification**: Instant spam detection
        - **Multiple Models**: Compare different optimization phases
        - **Interactive Dashboard**: Performance metrics and visualizations
        - **Data Explorer**: Comprehensive dataset analysis
        - **Prediction History**: Track classification results
        - **Text Analysis**: Detailed message pattern analysis
        
        ### 📝 Usage Instructions
        
        1. **Classifier**: Enter any text message to get instant spam/ham classification
        2. **Dashboard**: View model performance metrics and comparisons
        3. **Data Explorer**: Explore the training dataset and patterns
        4. **Sample Messages**: Test with pre-defined spam and ham examples
        
        ### 🎓 Educational Value
        
        This project demonstrates:
        - Complete ML pipeline development
        - Text preprocessing and feature engineering
        - Model optimization and evaluation
        - Web application deployment
        - Professional documentation and visualization
        
        ### 📧 Contact
        
        For questions or suggestions about this project, please refer to the project documentation
        or contact the development team.
        
        ---
        
        **Built with ❤️ using Python and Streamlit**
        """)

# Main application entry point
def main():
    """Main function to run the Streamlit app."""
    app = SMSSpamClassifierApp()
    app.run()

if __name__ == "__main__":
    main()