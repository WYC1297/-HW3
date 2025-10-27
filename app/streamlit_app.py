"""
Enhanced SMS Spam Classifier - Interactive Streamlit Dashboard
Comprehensive visualization explorer with live inference and threshold controls
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
from pathlib import Path
import sys
from collections import Counter
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve
)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)

# Add src directory to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir / 'src'))

# Import custom modules
try:
    from data_loader import SMSDataLoader
    from advanced_preprocessor import AdvancedTextPreprocessor
except ImportError:
    st.error("❌ Required modules not found. Please ensure src/ directory exists with proper modules.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier Dashboard",
    page_icon="📱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced styling
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
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .prediction-result {
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        text-align: center;
        font-size: 1.2rem;
        font-weight: bold;
    }
    
    .spam-result {
        background: linear-gradient(135deg, #ff6b6b, #ff8e8e);
        color: white;
        border: 2px solid #ff4757;
    }
    
    .ham-result {
        background: linear-gradient(135deg, #4ecdc4, #44a08d);
        color: white;
        border: 2px solid #00d2d3;
    }
    
    .info-box {
        background-color: #e3f2fd;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2196f3;
        margin: 1rem 0;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #4ecdc4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

class EnhancedSpamDashboard:
    """Enhanced interactive dashboard for SMS spam classification"""
    
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
        
        # Initialize session state
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_loaded' not in st.session_state:
            st.session_state.models_loaded = False
    
    @st.cache_data
    def load_data(_self):
        """Load and prepare dataset"""
        try:
            df = _self.data_loader.load_data()
            if df is not None:
                df = _self.data_loader.basic_clean()
                st.session_state.data_loaded = True
                return df
            else:
                st.error("❌ Failed to load dataset")
                return None
        except Exception as e:
            st.error(f"❌ Error loading data: {str(e)}")
            return None
    
    @st.cache_data
    def load_models(_self):
        """Load all available models"""
        models = {}
        vectorizers = {}
        feature_selectors = {}
        
        model_configs = [
            {
                'name': 'Advanced Model (Phase 3)',
                'model_path': 'models/phase3_final_svm_classifier.joblib',
                'vectorizer_path': 'models/phase3_final_tfidf_vectorizer.joblib',
                'feature_selector_path': 'models/phase3_final_feature_selector.joblib'
            },
            {
                'name': 'Optimized Model (Phase 2)',
                'model_path': 'models/optimized_svm_classifier.joblib',
                'vectorizer_path': 'models/optimized_tfidf_vectorizer.joblib',
                'feature_selector_path': 'models/optimized_feature_selector.joblib'
            },
            {
                'name': 'Baseline Model (Phase 1)',
                'model_path': 'models/svm_spam_classifier.joblib',
                'vectorizer_path': 'models/tfidf_vectorizer.joblib',
                'feature_selector_path': None
            }
        ]
        
        for config in model_configs:
            try:
                model_path = Path(config['model_path'])
                vectorizer_path = Path(config['vectorizer_path'])
                
                if model_path.exists() and vectorizer_path.exists():
                    models[config['name']] = joblib.load(model_path)
                    vectorizers[config['name']] = joblib.load(vectorizer_path)
                    
                    if config['feature_selector_path']:
                        selector_path = Path(config['feature_selector_path'])
                        if selector_path.exists():
                            feature_selectors[config['name']] = joblib.load(selector_path)
                    
                    st.sidebar.success(f"✅ {config['name']}")
                else:
                    st.sidebar.warning(f"⚠️ {config['name']} not found")
                    
            except Exception as e:
                st.sidebar.error(f"❌ {config['name']}: {str(e)}")
        
        if models:
            st.session_state.models_loaded = True
        
        return models, vectorizers, feature_selectors
    
    def predict_with_threshold(self, message: str, model_name: str, models, vectorizers, feature_selectors, threshold: float = 0.5):
        """Predict spam/ham with custom threshold"""
        try:
            if model_name not in models:
                return None, None, None, "Model not available"
            
            model = models[model_name]
            vectorizer = vectorizers[model_name]
            feature_selector = feature_selectors.get(model_name)
            
            # Transform message
            X_transformed = vectorizer.transform([message])
            if feature_selector is not None:
                X_transformed = feature_selector.transform(X_transformed)
            
            # Convert to dense if sparse
            if hasattr(X_transformed, 'toarray'):
                X_transformed = X_transformed.toarray()
            
            # Get prediction scores
            if hasattr(model, 'decision_function'):
                raw_score = model.decision_function(X_transformed)[0]
                # Convert to probability using sigmoid
                probability = 1 / (1 + np.exp(-raw_score))
            else:
                proba = model.predict_proba(X_transformed)[0]
                probability = proba[1]  # Probability of spam class
                raw_score = probability
            
            # Apply custom threshold
            prediction = 'spam' if probability >= threshold else 'ham'
            confidence = probability if prediction == 'spam' else (1 - probability)
            
            return prediction, probability, confidence, None
            
        except Exception as e:
            return None, None, None, str(e)
    
    def generate_random_example(self, df, message_type):
        """Generate a random example message of specified type"""
        import random
        
        if message_type == 'spam':
            # Get random spam message
            spam_messages = df[df['label'] == 'spam']['message'].tolist()
            return random.choice(spam_messages) if spam_messages else "WINNER! You have won $1000! Call now to claim your prize!"
        else:  # ham
            # Get random ham message  
            ham_messages = df[df['label'] == 'ham']['message'].tolist()
            return random.choice(ham_messages) if ham_messages else "Hey, are we still meeting for lunch tomorrow at 12?"
    
    def create_class_distribution_chart(self, df):
        """Create interactive class distribution visualization"""
        class_counts = df['label'].value_counts()
        total = len(df)
        
        # Create subplot figure
        fig = make_subplots(
            rows=1, cols=2,
            specs=[[{"type": "bar"}, {"type": "pie"}]],
            subplot_titles=("Message Counts", "Distribution Percentage")
        )
        
        # Bar chart
        fig.add_trace(
            go.Bar(
                x=class_counts.index,
                y=class_counts.values,
                marker_color=[self.colors['ham'], self.colors['spam']],
                text=[f'{count:,}<br>({count/total*100:.1f}%)' for count in class_counts.values],
                textposition='outside',
                name='Count'
            ),
            row=1, col=1
        )
        
        # Pie chart
        fig.add_trace(
            go.Pie(
                labels=class_counts.index,
                values=class_counts.values,
                marker_colors=[self.colors['ham'], self.colors['spam']],
                textinfo='label+percent',
                textfont_size=14,
                name='Distribution'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="📊 SMS Message Class Distribution",
            height=500,
            showlegend=False
        )
        
        return fig
    
    def create_token_frequency_chart(self, df):
        """Create token frequency analysis"""
        # Prepare data
        spam_messages = df[df['label'] == 'spam']['message']
        ham_messages = df[df['label'] == 'ham']['message']
        
        # Create TF-IDF vectorizer
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        all_messages = pd.concat([spam_messages, ham_messages])
        vectorizer.fit(all_messages)
        
        # Transform and analyze
        spam_tfidf = vectorizer.transform(spam_messages)
        ham_tfidf = vectorizer.transform(ham_messages)
        feature_names = vectorizer.get_feature_names_out()
        
        spam_scores = np.array(spam_tfidf.mean(axis=0)).flatten()
        ham_scores = np.array(ham_tfidf.mean(axis=0)).flatten()
        
        # Get top tokens
        top_n = 15
        spam_top_indices = spam_scores.argsort()[-top_n:][::-1]
        ham_top_indices = ham_scores.argsort()[-top_n:][::-1]
        
        spam_tokens = [feature_names[i] for i in spam_top_indices]
        spam_values = [spam_scores[i] for i in spam_top_indices]
        ham_tokens = [feature_names[i] for i in ham_top_indices]
        ham_values = [ham_scores[i] for i in ham_top_indices]
        
        # Create subplot
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("Top Tokens in SPAM Messages", "Top Tokens in HAM Messages"),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Spam tokens
        fig.add_trace(
            go.Bar(
                y=spam_tokens,
                x=spam_values,
                orientation='h',
                marker_color=self.colors['spam'],
                name='SPAM',
                text=[f'{v:.3f}' for v in spam_values],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Ham tokens
        fig.add_trace(
            go.Bar(
                y=ham_tokens,
                x=ham_values,
                orientation='h',
                marker_color=self.colors['ham'],
                name='HAM',
                text=[f'{v:.3f}' for v in ham_values],
                textposition='outside'
            ),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="🔤 Token Frequency Analysis - SPAM vs HAM",
            height=600,
            showlegend=False
        )
        
        return fig
    
    def create_model_performance_charts(self, df, models, vectorizers, feature_selectors):
        """Create comprehensive model performance visualizations"""
        # Prepare test data
        X = df['message']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        performance_data = {}
        
        for model_name in models.keys():
            try:
                model = models[model_name]
                vectorizer = vectorizers[model_name]
                feature_selector = feature_selectors.get(model_name)
                
                # Transform test data
                X_test_transformed = vectorizer.transform(X_test)
                if feature_selector is not None:
                    X_test_transformed = feature_selector.transform(X_test_transformed)
                
                if hasattr(X_test_transformed, 'toarray'):
                    X_test_transformed = X_test_transformed.toarray()
                
                # Predictions
                y_pred = model.predict(X_test_transformed)
                
                # Convert to binary
                y_test_binary = (y_test == 'spam').astype(int)
                y_pred_binary = (y_pred == 'spam').astype(int) if isinstance(y_pred[0], str) else y_pred
                
                # Get scores
                if hasattr(model, 'decision_function'):
                    y_scores = model.decision_function(X_test_transformed)
                else:
                    y_proba = model.predict_proba(X_test_transformed)
                    y_scores = y_proba[:, 1]
                
                # Calculate metrics
                accuracy = accuracy_score(y_test_binary, y_pred_binary)
                precision = precision_score(y_test_binary, y_pred_binary)
                recall = recall_score(y_test_binary, y_pred_binary)
                f1 = f1_score(y_test_binary, y_pred_binary)
                
                # ROC and PR curves
                fpr, tpr, _ = roc_curve(y_test_binary, y_scores)
                roc_auc = auc(fpr, tpr)
                
                prec_curve, rec_curve, _ = precision_recall_curve(y_test_binary, y_scores)
                pr_auc = auc(rec_curve, prec_curve)
                
                performance_data[model_name] = {
                    'metrics': {
                        'Accuracy': accuracy,
                        'Precision': precision,
                        'Recall': recall,
                        'F1-Score': f1,
                        'ROC_AUC': roc_auc,
                        'PR_AUC': pr_auc
                    },
                    'curves': {
                        'roc': (fpr, tpr, roc_auc),
                        'pr': (prec_curve, rec_curve, pr_auc)
                    },
                    'confusion_matrix': confusion_matrix(y_test_binary, y_pred_binary),
                    'test_data': (y_test_binary, y_pred_binary, y_scores)
                }
                
            except Exception as e:
                st.error(f"❌ Error evaluating {model_name}: {str(e)}")
        
        return performance_data
    
    def create_roc_pr_curves(self, performance_data):
        """Create ROC and PR curves visualization"""
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=("ROC Curves", "Precision-Recall Curves")
        )
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (model_name, data) in enumerate(performance_data.items()):
            fpr, tpr, roc_auc = data['curves']['roc']
            prec_curve, rec_curve, pr_auc = data['curves']['pr']
            
            # ROC Curve
            fig.add_trace(
                go.Scatter(
                    x=fpr, y=tpr,
                    mode='lines',
                    name=f'{model_name} (AUC = {roc_auc:.3f})',
                    line=dict(width=3, color=colors[i % len(colors)]),
                    hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'
                ),
                row=1, col=1
            )
            
            # PR Curve
            fig.add_trace(
                go.Scatter(
                    x=rec_curve, y=prec_curve,
                    mode='lines',
                    name=f'{model_name} (AUC = {pr_auc:.3f})',
                    line=dict(width=3, color=colors[i % len(colors)]),
                    hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>',
                    showlegend=False
                ),
                row=1, col=2
            )
        
        # Add baseline lines
        fig.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'),
                      name='Random', showlegend=False),
            row=1, col=1
        )
        
        fig.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig.update_xaxes(title_text="Recall", row=1, col=2)
        fig.update_yaxes(title_text="Precision", row=1, col=2)
        
        fig.update_layout(
            title_text="📈 Model Performance Curves",
            height=500,
            hovermode='closest'
        )
        
        return fig
    
    def create_threshold_analysis(self, performance_data, selected_model):
        """Create threshold sweep analysis for selected model"""
        if selected_model not in performance_data:
            return None
        
        y_true, y_pred, y_scores = performance_data[selected_model]['test_data']
        
        thresholds = np.linspace(0, 1, 101)
        precisions, recalls, f1_scores = [], [], []
        
        for threshold in thresholds:
            y_pred_thresh = (y_scores >= threshold).astype(int)
            
            if np.sum(y_pred_thresh) == 0:
                precisions.append(0)
                recalls.append(0)
                f1_scores.append(0)
            else:
                precision = precision_score(y_true, y_pred_thresh, zero_division=0)
                recall = recall_score(y_true, y_pred_thresh, zero_division=0)
                f1 = f1_score(y_true, y_pred_thresh, zero_division=0)
                
                precisions.append(precision)
                recalls.append(recall)
                f1_scores.append(f1)
        
        # Find optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        
        # Create plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode='lines', name='Precision',
                                line=dict(width=3, color=self.colors['primary']),
                                hovertemplate='Threshold: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode='lines', name='Recall',
                                line=dict(width=3, color=self.colors['secondary']),
                                hovertemplate='Threshold: %{x:.3f}<br>Recall: %{y:.3f}<extra></extra>'))
        fig.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode='lines', name='F1-Score',
                                line=dict(width=3, color=self.colors['spam']),
                                hovertemplate='Threshold: %{x:.3f}<br>F1-Score: %{y:.3f}<extra></extra>'))
        
        # Add optimal threshold line
        fig.add_vline(x=optimal_threshold, line_dash="dash", line_color="red",
                     annotation_text=f"Optimal F1: {optimal_threshold:.3f}")
        
        fig.update_layout(
            title=f"🎯 Threshold Analysis - {selected_model}",
            xaxis_title="Classification Threshold",
            yaxis_title="Score",
            height=500,
            hovermode='x unified'
        )
        
        return fig, optimal_threshold
    
    def run_dashboard(self):
        """Main dashboard runner"""
        st.markdown('<h1 class="main-header">📱 SMS Spam Classifier Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar
        st.sidebar.title("🎛️ Dashboard Controls")
        
        # Load data and models
        with st.sidebar:
            st.subheader("📊 Data & Models")
            
            if st.button("🔄 Reload Data & Models"):
                st.cache_data.clear()
        
        # Load data
        df = self.load_data()
        if df is None:
            st.error("❌ Cannot proceed without dataset")
            return
        
        # Load models
        models, vectorizers, feature_selectors = self.load_models()
        if not models:
            st.error("❌ No models available")
            return
        
        # Display basic stats in sidebar
        st.sidebar.markdown("### 📈 Dataset Stats")
        total_messages = len(df)
        spam_count = len(df[df['label'] == 'spam'])
        ham_count = len(df[df['label'] == 'ham'])
        
        st.sidebar.metric("Total Messages", f"{total_messages:,}")
        st.sidebar.metric("Spam Messages", f"{spam_count:,}")
        st.sidebar.metric("Ham Messages", f"{ham_count:,}")
        st.sidebar.metric("Models Loaded", len(models))
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏠 Overview", "📊 Visualizations", "🎯 Model Analysis", 
            "🔮 Live Inference", "📋 Reports"
        ])
        
        with tab1:
            st.header("📊 Dataset Overview")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{total_messages:,}</h3>
                    <p>Total Messages</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{spam_count:,}</h3>
                    <p>Spam Messages</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{ham_count:,}</h3>
                    <p>Ham Messages</p>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                spam_percentage = (spam_count / total_messages) * 100
                st.markdown(f"""
                <div class="metric-card">
                    <h3>{spam_percentage:.1f}%</h3>
                    <p>Spam Rate</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Class distribution chart
            st.plotly_chart(self.create_class_distribution_chart(df), width="stretch")
            
            # Sample messages
            st.subheader("📧 Sample Messages")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🚫 Spam Examples:**")
                spam_samples = df[df['label'] == 'spam']['message'].sample(3, random_state=42)
                for i, msg in enumerate(spam_samples, 1):
                    st.write(f"{i}. {msg[:100]}..." if len(msg) > 100 else f"{i}. {msg}")
            
            with col2:
                st.write("**✅ Ham Examples:**")
                ham_samples = df[df['label'] == 'ham']['message'].sample(3, random_state=42)
                for i, msg in enumerate(ham_samples, 1):
                    st.write(f"{i}. {msg[:100]}..." if len(msg) > 100 else f"{i}. {msg}")
        
        with tab2:
            st.header("📊 Interactive Visualizations")
            
            # Token frequency analysis
            st.plotly_chart(self.create_token_frequency_chart(df), width="stretch")
            
            # Word clouds
            st.subheader("☁️ Word Clouds")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🚫 Spam Word Cloud**")
                spam_text = ' '.join(df[df['label'] == 'spam']['message'].sample(100, random_state=42))
                if spam_text:
                    wordcloud_spam = WordCloud(
                        width=400, height=300, background_color='white',
                        colormap='Reds', max_words=100, random_state=42
                    ).generate(spam_text)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud_spam, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
            
            with col2:
                st.write("**✅ Ham Word Cloud**")
                ham_text = ' '.join(df[df['label'] == 'ham']['message'].sample(100, random_state=42))
                if ham_text:
                    wordcloud_ham = WordCloud(
                        width=400, height=300, background_color='white',
                        colormap='Blues', max_words=100, random_state=42
                    ).generate(ham_text)
                    
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.imshow(wordcloud_ham, interpolation='bilinear')
                    ax.axis('off')
                    st.pyplot(fig)
                    plt.close()
        
        with tab3:
            st.header("🎯 Model Performance Analysis")
            
            # Calculate performance data
            with st.spinner("📊 Evaluating model performance..."):
                performance_data = self.create_model_performance_charts(df, models, vectorizers, feature_selectors)
            
            if performance_data:
                # Performance metrics table
                st.subheader("📈 Performance Metrics")
                metrics_df = pd.DataFrame({
                    model: data['metrics'] for model, data in performance_data.items()
                }).T
                st.dataframe(metrics_df.round(4), width="stretch")
                
                # ROC and PR curves
                st.plotly_chart(self.create_roc_pr_curves(performance_data), width="stretch")
                
                # Confusion matrices
                st.subheader("🎯 Confusion Matrices")
                cols = st.columns(len(performance_data))
                
                for i, (model_name, data) in enumerate(performance_data.items()):
                    with cols[i]:
                        cm = data['confusion_matrix']
                        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
                        
                        fig = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=['Ham', 'Spam'],
                            y=['Ham', 'Spam'],
                            colorscale='Blues',
                            text=[[f'{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)' 
                                  for j in range(len(cm[0]))] for i in range(len(cm))],
                            texttemplate="%{text}",
                            textfont={"size": 12},
                            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
                        ))
                        
                        fig.update_layout(
                            title=f'{model_name}',
                            height=400
                        )
                        st.plotly_chart(fig, width="stretch")
                
                # Threshold analysis
                st.subheader("🎯 Threshold Analysis")
                selected_model = st.selectbox(
                    "Select model for threshold analysis:",
                    options=list(performance_data.keys())
                )
                
                if selected_model:
                    threshold_fig, optimal_threshold = self.create_threshold_analysis(performance_data, selected_model)
                    if threshold_fig:
                        st.plotly_chart(threshold_fig, width="stretch")
                        st.info(f"💡 Optimal F1 threshold for {selected_model}: **{optimal_threshold:.3f}**")
        
        with tab4:
            st.header("🔮 Live SMS Classification")
            
            # Model selection
            col1, col2 = st.columns([2, 1])
            
            with col1:
                selected_model = st.selectbox(
                    "Choose classification model:",
                    options=list(models.keys()),
                    help="Select which model to use for prediction"
                )
            
            with col2:
                threshold = st.slider(
                    "Classification Threshold:",
                    min_value=0.0, max_value=1.0, value=0.5, step=0.01,
                    help="Adjust the threshold for spam classification"
                )
            
            # Text input with example message support
            default_value = ""
            if hasattr(st.session_state, 'example_message'):
                default_value = st.session_state.example_message
                # Clear the session state after using it
                delattr(st.session_state, 'example_message')
            
            message_input = st.text_area(
                "📝 Enter SMS message to classify:",
                value=default_value,
                placeholder="Type your message here or use the example generator below...",
                height=150,
                help="Enter the SMS message you want to classify as spam or ham"
            )
            
            # Prediction
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                if st.button("🚀 Classify Message", type="primary", width="stretch"):
                    if message_input.strip():
                        prediction, probability, confidence, error = self.predict_with_threshold(
                            message_input, selected_model, models, vectorizers, feature_selectors, threshold
                        )
                        
                        if error:
                            st.error(f"❌ Error: {error}")
                        else:
                            # Display result
                            if prediction == 'spam':
                                st.markdown(f"""
                                <div class="prediction-result spam-result">
                                    🚫 SPAM DETECTED<br>
                                    Spam Probability: {probability:.1%}<br>
                                    Confidence: {confidence:.1%}<br>
                                    Model: {selected_model}<br>
                                    Threshold: {threshold:.2f}
                                </div>
                                """, unsafe_allow_html=True)
                            else:
                                st.markdown(f"""
                                <div class="prediction-result ham-result">
                                    ✅ LEGITIMATE MESSAGE<br>
                                    Spam Probability: {probability:.1%}<br>
                                    Confidence: {confidence:.1%}<br>
                                    Model: {selected_model}<br>
                                    Threshold: {threshold:.2f}
                                </div>
                                """, unsafe_allow_html=True)
                            
                            # Message analysis
                            st.subheader("📊 Message Analysis")
                            col1, col2, col3, col4 = st.columns(4)
                            
                            with col1:
                                st.metric("📏 Characters", len(message_input))
                            with col2:
                                st.metric("🔤 Words", len(message_input.split()))
                            with col3:
                                st.metric("❗ Exclamations", message_input.count('!'))
                            with col4:
                                st.metric("🔢 Numbers", sum(c.isdigit() for c in message_input))
                            
                            # Probability visualization
                            st.subheader("📈 Prediction Confidence")
                            prob_fig = go.Figure(go.Indicator(
                                mode = "gauge+number+delta",
                                value = probability * 100,
                                domain = {'x': [0, 1], 'y': [0, 1]},
                                title = {'text': "Spam Probability (%)"},
                                delta = {'reference': threshold * 100},
                                gauge = {
                                    'axis': {'range': [None, 100]},
                                    'bar': {'color': self.colors['spam'] if prediction == 'spam' else self.colors['ham']},
                                    'steps': [
                                        {'range': [0, threshold * 100], 'color': "lightgray"},
                                        {'range': [threshold * 100, 100], 'color': "lightcoral"}
                                    ],
                                    'threshold': {
                                        'line': {'color': "red", 'width': 4},
                                        'thickness': 0.75,
                                        'value': threshold * 100
                                    }
                                }
                            ))
                            prob_fig.update_layout(height=400)
                            st.plotly_chart(prob_fig, width="stretch")
                    else:
                        st.warning("⚠️ Please enter a message to classify")
            
            # Example messages
            st.subheader("💡 智能範例生成器")
            
            # Example generator
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                example_type = st.selectbox(
                    "選擇範例類型:",
                    ["🚫 Spam", "✅ Ham"],
                    help="選擇要生成的範例類型"
                )
            
            with col2:
                if st.button("🎲 生成隨機範例", type="primary", help="基於真實數據生成隨機範例"):
                    message_type = 'spam' if example_type == "🚫 Spam" else 'ham'
                    generated_example = self.generate_random_example(df, message_type)
                    st.session_state.generated_example = generated_example
                    st.success(f"已生成 {example_type} 範例！")
            
            with col3:
                if st.button("📋 使用範例", help="將生成的範例填入文本框"):
                    if hasattr(st.session_state, 'generated_example'):
                        st.session_state.example_message = st.session_state.generated_example
                        st.success("範例已填入！")
                    else:
                        st.warning("請先生成範例")
            
            # Display generated example
            if hasattr(st.session_state, 'generated_example'):
                st.markdown("**📝 生成的範例:**")
                example_text = st.session_state.generated_example
                # Truncate if too long for display
                display_text = example_text[:200] + "..." if len(example_text) > 200 else example_text
                
                # Color code based on type
                if example_type == "🚫 Spam":
                    st.markdown(f"""
                    <div style="background-color: #ffe6e6; padding: 10px; border-radius: 5px; border-left: 4px solid #ff4757;">
                        <strong>🚫 Spam 範例:</strong><br>
                        {display_text}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style="background-color: #e6f7f1; padding: 10px; border-radius: 5px; border-left: 4px solid #00d2d3;">
                        <strong>✅ Ham 範例:</strong><br>
                        {display_text}
                    </div>
                    """, unsafe_allow_html=True)
            
            # Predefined examples section
            st.markdown("---")
            st.subheader("📚 預設範例")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**🚫 Spam 範例:**")
                spam_examples = [
                    "WINNER! You have won $1000! Call now to claim your prize!",
                    "FREE! Get your iPhone now by clicking this link. Limited time offer!",
                    "Congratulations! You've been selected for a cash prize. Text YES to claim."
                ]
                for i, example in enumerate(spam_examples, 1):
                    if st.button(f"Spam 範例 {i}", key=f"spam_{i}", help=f"使用預設 Spam 範例 {i}"):
                        st.session_state.example_message = example
                        st.success(f"已選擇 Spam 範例 {i}！")
            
            with col2:
                st.write("**✅ Ham 範例:**")
                ham_examples = [
                    "Hey, are we still meeting for lunch tomorrow at 12?",
                    "Thanks for the great meeting today. Let's follow up next week.",
                    "Mom, I'll be home late tonight. Don't wait up for dinner."
                ]
                for i, example in enumerate(ham_examples, 1):
                    if st.button(f"Ham 範例 {i}", key=f"ham_{i}", help=f"使用預設 Ham 範例 {i}"):
                        st.session_state.example_message = example
                        st.success(f"已選擇 Ham 範例 {i}！")
        
        with tab5:
            st.header("📋 Analysis Reports")
            
            # Check for generated reports
            reports_dir = Path("reports/visualizations")
            if reports_dir.exists():
                st.subheader("📊 Generated Visualizations")
                
                # List available files
                viz_files = list(reports_dir.glob("*.png"))
                html_files = list(reports_dir.glob("*.html"))
                csv_files = list(reports_dir.glob("*.csv"))
                md_files = list(reports_dir.glob("*.md"))
                
                if viz_files or html_files or csv_files or md_files:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**📊 Static Images:**")
                        for file in sorted(viz_files):
                            st.write(f"• {file.name}")
                        
                        st.write("**📄 Data Files:**")
                        for file in sorted(csv_files):
                            st.write(f"• {file.name}")
                    
                    with col2:
                        st.write("**🌐 Interactive Charts:**")
                        for file in sorted(html_files):
                            st.write(f"• {file.name}")
                        
                        st.write("**📋 Reports:**")
                        for file in sorted(md_files):
                            st.write(f"• {file.name}")
                            
                            # Display markdown content
                            if st.button(f"View {file.name}", key=f"view_{file.name}"):
                                try:
                                    with open(file, 'r', encoding='utf-8') as f:
                                        content = f.read()
                                    st.markdown(content)
                                except Exception as e:
                                    st.error(f"Error reading {file.name}: {e}")
                else:
                    st.info("📝 No reports found. Run the CLI visualization script to generate reports.")
            else:
                st.info("📝 Reports directory not found. Run the CLI visualization script first.")
            
            # Generate reports button
            st.subheader("🔧 Generate New Reports")
            if st.button("🚀 Run Visualization Analysis", type="primary"):
                st.info("💡 To generate comprehensive reports, run the CLI script:")
                st.code("python scripts/visualize_spam.py", language="bash")
        
        # Footer
        st.markdown("---")
        st.markdown("🎨 **Enhanced SMS Spam Classifier Dashboard** | Interactive Analytics & Live Inference")

# Run the dashboard
if __name__ == "__main__":
    dashboard = EnhancedSpamDashboard()
    dashboard.run_dashboard()