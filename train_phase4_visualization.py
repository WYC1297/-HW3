"""
Phase 4: Advanced Data Visualization and Performance Analysis
Enhanced visual analysis of SMS spam classification with comprehensive reporting.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
import sys
import os
import random
import warnings
from collections import Counter
from wordcloud import WordCloud
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)
from sklearn.preprocessing import MinMaxScaler

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)
warnings.filterwarnings('ignore')

# Configure plotting
plt.style.use('default')
sns.set_palette("husl")

# Add src directory to path
current_dir = os.getcwd()
sys.path.append(os.path.join(current_dir, 'src'))

from src.data_loader import SMSDataLoader

class AdvancedVisualizationSuite:
    """
    Advanced visualization suite for comprehensive SMS spam classification analysis.
    """
    
    def __init__(self):
        self.colors = {
            'spam': '#FF6B6B',
            'ham': '#4ECDC4',
            'primary': '#45B7D1',
            'secondary': '#96CEB4',
            'accent': '#FFEAA7'
        }
        self.figures = {}
        
    def setup_matplotlib_style(self):
        """Configure matplotlib for publication-quality plots."""
        plt.rcParams.update({
            'figure.figsize': (12, 8),
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 11,
            'figure.titlesize': 16
        })
    
    def analyze_dataset_distribution(self, df: pd.DataFrame, save_dir: Path):
        """Create comprehensive dataset distribution analysis."""
        print("📊 Creating dataset distribution analysis...")
        
        # Calculate statistics
        total_messages = len(df)
        spam_count = len(df[df['label'] == 'spam'])
        ham_count = len(df[df['label'] == 'ham'])
        
        # Add derived features
        df = df.copy()
        df['message_length'] = df['message'].str.len()
        df['word_count'] = df['message'].str.split().str.len()
        df['exclamation_count'] = df['message'].apply(lambda x: x.count('!'))
        df['question_count'] = df['message'].apply(lambda x: x.count('?'))
        df['uppercase_ratio'] = df['message'].apply(
            lambda x: sum(1 for c in x if c.isupper()) / len(x) if len(x) > 0 else 0
        )
        
        # Create subplot figure
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('SMS Dataset Distribution Analysis', fontsize=20, fontweight='bold')
        
        # 1. Class distribution
        class_counts = df['label'].value_counts()
        colors = [self.colors['spam'] if label == 'spam' else self.colors['ham'] 
                 for label in class_counts.index]
        
        axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                      colors=colors, startangle=90)
        axes[0, 0].set_title('Class Distribution')
        
        # 2. Message length distribution
        sns.boxplot(data=df, x='label', y='message_length', ax=axes[0, 1],
                   palette={'spam': self.colors['spam'], 'ham': self.colors['ham']})
        axes[0, 1].set_title('Message Length Distribution')
        axes[0, 1].set_ylabel('Character Count')
        
        # 3. Word count distribution
        sns.violinplot(data=df, x='label', y='word_count', ax=axes[0, 2],
                      palette={'spam': self.colors['spam'], 'ham': self.colors['ham']})
        axes[0, 2].set_title('Word Count Distribution')
        axes[0, 2].set_ylabel('Word Count')
        
        # 4. Exclamation marks
        sns.barplot(data=df, x='label', y='exclamation_count', ax=axes[1, 0],
                   palette={'spam': self.colors['spam'], 'ham': self.colors['ham']})
        axes[1, 0].set_title('Average Exclamation Marks')
        axes[1, 0].set_ylabel('Count')
        
        # 5. Uppercase ratio
        sns.boxplot(data=df, x='label', y='uppercase_ratio', ax=axes[1, 1],
                   palette={'spam': self.colors['spam'], 'ham': self.colors['ham']})
        axes[1, 1].set_title('Uppercase Character Ratio')
        axes[1, 1].set_ylabel('Ratio')
        
        # 6. Message length histogram
        df[df['label'] == 'spam']['message_length'].hist(
            alpha=0.7, bins=50, color=self.colors['spam'], 
            label='Spam', ax=axes[1, 2]
        )
        df[df['label'] == 'ham']['message_length'].hist(
            alpha=0.7, bins=50, color=self.colors['ham'], 
            label='Ham', ax=axes[1, 2]
        )
        axes[1, 2].set_title('Message Length Histogram')
        axes[1, 2].set_xlabel('Character Count')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].legend()
        
        plt.tight_layout()
        plt.savefig(save_dir / 'dataset_distribution_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return {
            'total_messages': total_messages,
            'spam_count': spam_count,
            'ham_count': ham_count,
            'spam_percentage': (spam_count / total_messages) * 100,
            'avg_spam_length': df[df['label'] == 'spam']['message_length'].mean(),
            'avg_ham_length': df[df['label'] == 'ham']['message_length'].mean(),
            'avg_spam_words': df[df['label'] == 'spam']['word_count'].mean(),
            'avg_ham_words': df[df['label'] == 'ham']['word_count'].mean()
        }
    
    def create_advanced_wordclouds(self, df: pd.DataFrame, save_dir: Path):
        """Generate enhanced word clouds with styling."""
        print("☁️ Creating advanced word clouds...")
        
        # Prepare text data
        spam_text = ' '.join(df[df['label'] == 'spam']['message'])
        ham_text = ' '.join(df[df['label'] == 'ham']['message'])
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Word Cloud Analysis: Spam vs Ham Messages', fontsize=20, fontweight='bold')
        
        # Spam word cloud
        spam_wordcloud = WordCloud(
            width=800, height=400, 
            background_color='white',
            colormap='Reds',
            max_words=100,
            random_state=42
        ).generate(spam_text)
        
        axes[0].imshow(spam_wordcloud, interpolation='bilinear')
        axes[0].set_title('Spam Messages', fontsize=16, color=self.colors['spam'], fontweight='bold')
        axes[0].axis('off')
        
        # Ham word cloud
        ham_wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Blues',
            max_words=100,
            random_state=42
        ).generate(ham_text)
        
        axes[1].imshow(ham_wordcloud, interpolation='bilinear')
        axes[1].set_title('Ham Messages', fontsize=16, color=self.colors['ham'], fontweight='bold')
        axes[1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'advanced_wordclouds.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def analyze_ngrams_patterns(self, df: pd.DataFrame, save_dir: Path, n_grams=[1, 2]):
        """Analyze n-gram patterns with enhanced visualization."""
        print("🔤 Analyzing n-gram patterns...")
        
        from sklearn.feature_extraction.text import CountVectorizer
        
        fig, axes = plt.subplots(len(n_grams), 2, figsize=(16, 6 * len(n_grams)))
        if len(n_grams) == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('N-gram Frequency Analysis', fontsize=20, fontweight='bold')
        
        for i, n in enumerate(n_grams):
            # Spam n-grams
            spam_messages = df[df['label'] == 'spam']['message']
            vectorizer_spam = CountVectorizer(ngram_range=(n, n), max_features=15, stop_words='english')
            spam_counts = vectorizer_spam.fit_transform(spam_messages)
            spam_freq = spam_counts.sum(axis=0).A1
            spam_features = vectorizer_spam.get_feature_names_out()
            spam_top = sorted(zip(spam_features, spam_freq), key=lambda x: x[1], reverse=True)
            
            # Plot spam n-grams
            features, frequencies = zip(*spam_top)
            axes[i, 0].barh(range(len(features)), frequencies, color=self.colors['spam'], alpha=0.8)
            axes[i, 0].set_yticks(range(len(features)))
            axes[i, 0].set_yticklabels(features)
            axes[i, 0].set_title(f'Top {n}-grams in Spam Messages', fontweight='bold')
            axes[i, 0].set_xlabel('Frequency')
            
            # Ham n-grams
            ham_messages = df[df['label'] == 'ham']['message']
            vectorizer_ham = CountVectorizer(ngram_range=(n, n), max_features=15, stop_words='english')
            ham_counts = vectorizer_ham.fit_transform(ham_messages)
            ham_freq = ham_counts.sum(axis=0).A1
            ham_features = vectorizer_ham.get_feature_names_out()
            ham_top = sorted(zip(ham_features, ham_freq), key=lambda x: x[1], reverse=True)
            
            # Plot ham n-grams
            features, frequencies = zip(*ham_top)
            axes[i, 1].barh(range(len(features)), frequencies, color=self.colors['ham'], alpha=0.8)
            axes[i, 1].set_yticks(range(len(features)))
            axes[i, 1].set_yticklabels(features)
            axes[i, 1].set_title(f'Top {n}-grams in Ham Messages', fontweight='bold')
            axes[i, 1].set_xlabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'ngram_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def evaluate_model_performance(self, model_name: str, model_path: str, 
                                 vectorizer_path: str, feature_selector_path: str,
                                 X_test, y_test, save_dir: Path):
        """Comprehensive model performance evaluation."""
        print(f"🔍 Evaluating {model_name} performance...")
        
        try:
            # Load models
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            
            feature_selector = None
            if feature_selector_path and Path(feature_selector_path).exists():
                feature_selector = joblib.load(feature_selector_path)
                print(f"  ✅ Loaded feature selector: {Path(feature_selector_path).name}")
            
            # Transform test data
            X_test_transformed = vectorizer.transform(X_test)
            
            # Apply feature selection if available
            if feature_selector is not None:
                X_test_transformed = feature_selector.transform(X_test_transformed)
                print(f"  📏 Applied feature selection: {X_test_transformed.shape[1]} features")
            
            # Convert to dense if needed
            if hasattr(X_test_transformed, 'toarray'):
                X_test_transformed = X_test_transformed.toarray()
            
            # Make predictions
            y_pred = model.predict(X_test_transformed)
            y_scores = model.decision_function(X_test_transformed)
            
            # Convert labels to binary format
            y_test_binary = (y_test == 'spam').astype(int)
            y_pred_binary = (y_pred == 'spam').astype(int) if isinstance(y_pred[0], str) else y_pred
            
            # Normalize scores for ROC curve
            scaler = MinMaxScaler()
            y_scores_norm = scaler.fit_transform(y_scores.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            metrics = {
                'Accuracy': accuracy_score(y_test_binary, y_pred_binary),
                'Precision': precision_score(y_test_binary, y_pred_binary),
                'Recall': recall_score(y_test_binary, y_pred_binary),
                'F1-Score': f1_score(y_test_binary, y_pred_binary)
            }
            
            # ROC and PR curves
            fpr, tpr, _ = roc_curve(y_test_binary, y_scores_norm)
            roc_auc = auc(fpr, tpr)
            precision_curve, recall_curve, _ = precision_recall_curve(y_test_binary, y_scores_norm)
            pr_auc = auc(recall_curve, precision_curve)
            
            metrics['ROC AUC'] = roc_auc
            metrics['PR AUC'] = pr_auc
            
            # Create performance visualization
            self._create_performance_plots(
                model_name, y_test_binary, y_pred_binary, y_scores_norm,
                fpr, tpr, roc_auc, precision_curve, recall_curve, pr_auc,
                metrics, save_dir
            )
            
            return metrics
            
        except Exception as e:
            print(f"  ❌ Error evaluating {model_name}: {e}")
            return None
    
    def _create_performance_plots(self, model_name, y_test, y_pred, y_scores,
                                fpr, tpr, roc_auc, precision_curve, recall_curve, pr_auc,
                                metrics, save_dir):
        """Create detailed performance visualization plots."""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'{model_name} - Performance Analysis', fontsize=18, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix', fontweight='bold')
        axes[0, 0].set_ylabel('Actual')
        axes[0, 0].set_xlabel('Predicted')
        
        # 2. ROC Curve
        axes[0, 1].plot(fpr, tpr, color=self.colors['primary'], lw=2, 
                       label=f'ROC Curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve', fontweight='bold')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        axes[1, 0].plot(recall_curve, precision_curve, color=self.colors['secondary'], lw=2,
                       label=f'PR Curve (AUC = {pr_auc:.3f})')
        axes[1, 0].set_xlim([0.0, 1.0])
        axes[1, 0].set_ylim([0.0, 1.05])
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve', fontweight='bold')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Metrics Summary
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = axes[1, 1].bar(metric_names, metric_values, 
                             color=[self.colors['primary'], self.colors['secondary'], 
                                   self.colors['accent'], self.colors['spam'], 
                                   self.colors['ham'], self.colors['primary']][:len(metric_names)])
        
        axes[1, 1].set_title('Performance Metrics', fontweight='bold')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].set_ylim([0, 1])
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        safe_name = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        plt.savefig(save_dir / f'{safe_name}_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_comprehensive_comparison(self, all_results: dict, save_dir: Path):
        """Create comprehensive comparison across all phases."""
        print("📈 Creating comprehensive phase comparison...")
        
        if not all_results:
            print("  ⚠️ No results available for comparison")
            return
        
        # Prepare data for plotting
        phases = list(all_results.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC', 'PR AUC']
        
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Phase Comparison Dashboard', fontsize=20, fontweight='bold')
        
        # 1. Overall metrics comparison
        metric_data = {metric: [all_results[phase].get(metric, 0) for phase in phases] 
                      for metric in metrics[:4]}  # Main metrics
        
        x = np.arange(len(phases))
        width = 0.2
        
        for i, (metric, values) in enumerate(metric_data.items()):
            axes[0, 0].bar(x + i*width, values, width, label=metric, alpha=0.8)
        
        axes[0, 0].set_xlabel('Phases')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Main Performance Metrics', fontweight='bold')
        axes[0, 0].set_xticks(x + width * 1.5)
        axes[0, 0].set_xticklabels(phases, rotation=45)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. F1-Score progression
        f1_scores = [all_results[phase].get('F1-Score', 0) for phase in phases]
        axes[0, 1].plot(phases, f1_scores, marker='o', linewidth=3, markersize=8,
                       color=self.colors['primary'])
        axes[0, 1].set_title('F1-Score Progression', fontweight='bold')
        axes[0, 1].set_ylabel('F1-Score')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0.8, 1.0])
        
        # Add value labels
        for i, score in enumerate(f1_scores):
            axes[0, 1].annotate(f'{score:.3f}', (i, score), 
                               textcoords="offset points", xytext=(0,10), ha='center')
        
        # 3. Precision vs Recall
        precisions = [all_results[phase].get('Precision', 0) for phase in phases]
        recalls = [all_results[phase].get('Recall', 0) for phase in phases]
        
        axes[1, 0].scatter(recalls, precisions, s=200, alpha=0.7, 
                          c=range(len(phases)), cmap='viridis')
        
        # Add labels for each point
        for i, phase in enumerate(phases):
            axes[1, 0].annotate(f'Phase {i+1}', (recalls[i], precisions[i]),
                               xytext=(5, 5), textcoords='offset points')
        
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision vs Recall Trade-off', fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_xlim([0.7, 1.0])
        axes[1, 0].set_ylim([0.7, 1.0])
        
        # 4. AUC scores comparison
        roc_aucs = [all_results[phase].get('ROC AUC', 0) for phase in phases]
        pr_aucs = [all_results[phase].get('PR AUC', 0) for phase in phases]
        
        x = np.arange(len(phases))
        width = 0.35
        
        axes[1, 1].bar(x - width/2, roc_aucs, width, label='ROC AUC', 
                      color=self.colors['secondary'], alpha=0.8)
        axes[1, 1].bar(x + width/2, pr_aucs, width, label='PR AUC', 
                      color=self.colors['accent'], alpha=0.8)
        
        axes[1, 1].set_xlabel('Phases')
        axes[1, 1].set_ylabel('AUC Score')
        axes[1, 1].set_title('AUC Scores Comparison', fontweight='bold')
        axes[1, 1].set_xticks(x)
        axes[1, 1].set_xticklabels(phases, rotation=45)
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim([0.9, 1.0])
        
        plt.tight_layout()
        plt.savefig(save_dir / 'comprehensive_phase_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_interactive_dashboard(self, df: pd.DataFrame, results: dict, save_dir: Path):
        """Create interactive Plotly dashboard."""
        print("🌐 Creating interactive dashboard...")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=[
                'Class Distribution', 'Performance Metrics Comparison',
                'Message Length by Class', 'F1-Score Progression',
                'Precision vs Recall', 'ROC AUC Comparison'
            ],
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "box"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # 1. Class distribution pie chart
        class_counts = df['label'].value_counts()
        fig.add_trace(
            go.Pie(labels=class_counts.index, values=class_counts.values,
                  marker_colors=[self.colors['spam'], self.colors['ham']]),
            row=1, col=1
        )
        
        # 2. Performance metrics comparison
        if results:
            phases = list(results.keys())
            metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            for metric in metrics:
                values = [results[phase].get(metric, 0) for phase in phases]
                fig.add_trace(
                    go.Bar(x=phases, y=values, name=metric),
                    row=1, col=2
                )
        
        # 3. Message length distribution
        for label in ['spam', 'ham']:
            data = df[df['label'] == label]['message'].str.len()
            fig.add_trace(
                go.Box(y=data, name=label, 
                      marker_color=self.colors[label]),
                row=2, col=1
            )
        
        # 4. F1-Score progression
        if results:
            f1_scores = [results[phase].get('F1-Score', 0) for phase in phases]
            fig.add_trace(
                go.Scatter(x=phases, y=f1_scores, mode='lines+markers',
                          name='F1-Score', line=dict(width=3),
                          marker=dict(size=10)),
                row=2, col=2
            )
        
        # 5. Precision vs Recall
        if results:
            precisions = [results[phase].get('Precision', 0) for phase in phases]
            recalls = [results[phase].get('Recall', 0) for phase in phases]
            fig.add_trace(
                go.Scatter(x=recalls, y=precisions, mode='markers+text',
                          text=phases, textposition="top center",
                          marker=dict(size=15), name='Phases'),
                row=3, col=1
            )
        
        # 6. ROC AUC comparison
        if results:
            roc_aucs = [results[phase].get('ROC AUC', 0) for phase in phases]
            fig.add_trace(
                go.Bar(x=phases, y=roc_aucs, name='ROC AUC',
                      marker_color=self.colors['primary']),
                row=3, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title_text="SMS Spam Classification - Interactive Dashboard",
            title_x=0.5,
            showlegend=True
        )
        
        # Save interactive dashboard
        fig.write_html(save_dir / 'interactive_dashboard.html')
        print(f"  ✅ Interactive dashboard saved: {save_dir / 'interactive_dashboard.html'}")

def main():
    """Execute comprehensive Phase 4 visualization analysis."""
    print("="*70)
    print("🎨 PHASE 4: ADVANCED VISUALIZATION AND PERFORMANCE ANALYSIS")
    print("="*70)
    
    # Initialize components
    viz_suite = AdvancedVisualizationSuite()
    viz_suite.setup_matplotlib_style()
    data_loader = SMSDataLoader()
    
    # Create output directory
    output_dir = Path("visualizations")
    output_dir.mkdir(exist_ok=True)
    
    print("📂 Loading and preparing data...")
    # Load data
    df = data_loader.load_data()
    if df is None:
        print("❌ Error: Could not load data")
        return
    
    df = data_loader.basic_clean()
    print(f"✅ Dataset loaded: {len(df)} messages")
    
    # Prepare train/test split
    X = df['message']
    y = df['label']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"📊 Train set: {len(X_train)}, Test set: {len(X_test)}")
    
    # Phase 4.1: Dataset Distribution Analysis
    print("\n" + "="*60)
    print("📊 4.1 DATASET DISTRIBUTION ANALYSIS")
    print("="*60)
    
    dataset_stats = viz_suite.analyze_dataset_distribution(df, output_dir)
    
    # Phase 4.2: Advanced Word Clouds
    print("\n" + "="*60)
    print("☁️ 4.2 ADVANCED WORD CLOUD ANALYSIS")
    print("="*60)
    
    viz_suite.create_advanced_wordclouds(df, output_dir)
    
    # Phase 4.3: N-gram Pattern Analysis
    print("\n" + "="*60)
    print("🔤 4.3 N-GRAM PATTERN ANALYSIS")
    print("="*60)
    
    viz_suite.analyze_ngrams_patterns(df, output_dir, n_grams=[1, 2])
    
    # Phase 4.4: Model Performance Evaluation
    print("\n" + "="*60)
    print("🔍 4.4 MODEL PERFORMANCE EVALUATION")
    print("="*60)
    
    # Define model configurations
    model_configs = [
        {
            'name': 'Phase 1 (Baseline SVM)',
            'model_path': 'models/svm_spam_classifier.joblib',
            'vectorizer_path': 'models/tfidf_vectorizer.joblib',
            'feature_selector_path': None
        },
        {
            'name': 'Phase 2 (Optimized)',
            'model_path': 'models/optimized_svm_classifier.joblib',
            'vectorizer_path': 'models/optimized_tfidf_vectorizer.joblib',
            'feature_selector_path': 'models/optimized_feature_selector.joblib'
        },
        {
            'name': 'Phase 3 (Advanced)',
            'model_path': 'models/phase3_final_svm_classifier.joblib',
            'vectorizer_path': 'models/phase3_final_tfidf_vectorizer.joblib',
            'feature_selector_path': 'models/phase3_final_feature_selector.joblib'
        }
    ]
    
    # Evaluate each model
    all_results = {}
    
    for config in model_configs:
        model_name = config['name']
        
        # Check if model files exist
        if (Path(config['model_path']).exists() and 
            Path(config['vectorizer_path']).exists()):
            
            metrics = viz_suite.evaluate_model_performance(
                model_name=model_name,
                model_path=config['model_path'],
                vectorizer_path=config['vectorizer_path'],
                feature_selector_path=config['feature_selector_path'],
                X_test=X_test,
                y_test=y_test,
                save_dir=output_dir
            )
            
            if metrics:
                all_results[model_name] = metrics
                print(f"✅ {model_name} evaluation complete")
                print(f"   📈 F1-Score: {metrics['F1-Score']:.4f}")
                print(f"   📈 Accuracy: {metrics['Accuracy']:.4f}")
            else:
                print(f"❌ {model_name} evaluation failed")
        else:
            print(f"⚠️ {model_name} model files not found, skipping...")
    
    # Phase 4.5: Comprehensive Comparison
    print("\n" + "="*60)
    print("📈 4.5 COMPREHENSIVE PHASE COMPARISON")
    print("="*60)
    
    viz_suite.create_comprehensive_comparison(all_results, output_dir)
    
    # Phase 4.6: Interactive Dashboard
    print("\n" + "="*60)
    print("🌐 4.6 INTERACTIVE DASHBOARD CREATION")
    print("="*60)
    
    try:
        viz_suite.create_interactive_dashboard(df, all_results, output_dir)
    except Exception as e:
        print(f"⚠️ Interactive dashboard creation issue: {e}")
        print("📊 Static visualizations completed successfully")
    
    # Generate comprehensive report
    print("\n" + "="*60)
    print("📝 GENERATING COMPREHENSIVE REPORT")
    print("="*60)
    
    report_content = generate_final_report(dataset_stats, all_results, output_dir)
    
    with open(output_dir / "comprehensive_analysis_report.md", "w", encoding='utf-8') as f:
        f.write(report_content)
    
    # Display final summary
    print("\n" + "="*70)
    print("🎉 PHASE 4 ANALYSIS COMPLETE!")
    print("="*70)
    print(f"📁 Output directory: {output_dir.absolute()}")
    print(f"📊 Total files generated: {len(list(output_dir.glob('*')))}")
    print("\n📋 Generated files:")
    
    for file_path in sorted(output_dir.iterdir()):
        if file_path.is_file():
            file_size = file_path.stat().st_size / 1024  # KB
            print(f"  📄 {file_path.name} ({file_size:.1f} KB)")
    
    if all_results:
        best_model = max(all_results.keys(), key=lambda x: all_results[x]['F1-Score'])
        best_f1 = all_results[best_model]['F1-Score']
        print(f"\n🏆 Best performing model: {best_model}")
        print(f"   📈 F1-Score: {best_f1:.4f}")
    
    print("\n✨ Analysis complete! Check the visualizations directory for all outputs.")

def generate_final_report(dataset_stats: dict, results: dict, output_dir: Path) -> str:
    """Generate comprehensive final analysis report."""
    
    current_time = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    total_files = len(list(output_dir.glob("*")))
    
    # Find best model
    if results:
        best_model = max(results.keys(), key=lambda x: results[x]['F1-Score'])
        best_metrics = results[best_model]
    else:
        best_model = "No models evaluated"
        best_metrics = {}
    
    report = f"""# SMS Spam Classification - Comprehensive Analysis Report

*Generated on: {current_time}*

## 🎯 Executive Summary

This report presents a comprehensive visualization and performance analysis of the SMS spam classification project. The analysis encompasses data distribution patterns, linguistic features, model performance evolution, and comparative evaluation across multiple optimization phases.

### 📊 Dataset Overview
- **Total Messages**: {dataset_stats.get('total_messages', 'N/A'):,}
- **Spam Messages**: {dataset_stats.get('spam_count', 'N/A'):,} ({dataset_stats.get('spam_percentage', 0):.1f}%)
- **Ham Messages**: {dataset_stats.get('ham_count', 'N/A'):,} ({100 - dataset_stats.get('spam_percentage', 0):.1f}%)

### 🏆 Best Performing Model
**{best_model}**"""

    if best_metrics:
        report += f"""
- **Accuracy**: {best_metrics.get('Accuracy', 0):.4f}
- **Precision**: {best_metrics.get('Precision', 0):.4f}
- **Recall**: {best_metrics.get('Recall', 0):.4f}
- **F1-Score**: {best_metrics.get('F1-Score', 0):.4f}
- **ROC AUC**: {best_metrics.get('ROC AUC', 0):.4f}"""

    report += f"""

## 📈 Analysis Components

### 1. Dataset Distribution Analysis
- **Message Length Patterns**: 
  - Average Spam Length: {dataset_stats.get('avg_spam_length', 0):.0f} characters
  - Average Ham Length: {dataset_stats.get('avg_ham_length', 0):.0f} characters
- **Word Count Analysis**:
  - Average Spam Words: {dataset_stats.get('avg_spam_words', 0):.1f}
  - Average Ham Words: {dataset_stats.get('avg_ham_words', 0):.1f}
- **Linguistic Features**: Exclamation marks, uppercase ratios, and punctuation patterns

### 2. Text Pattern Analysis
- **Word Clouds**: Visual representation of most frequent terms in spam vs ham messages
- **N-gram Analysis**: Unigram and bigram frequency patterns
- **Linguistic Indicators**: Key features that distinguish spam from legitimate messages

### 3. Model Performance Evaluation"""

    if results:
        report += "\n"
        for model_name, metrics in results.items():
            report += f"""
#### {model_name}
- Accuracy: {metrics.get('Accuracy', 0):.4f}
- Precision: {metrics.get('Precision', 0):.4f}  
- Recall: {metrics.get('Recall', 0):.4f}
- F1-Score: {metrics.get('F1-Score', 0):.4f}
- ROC AUC: {metrics.get('ROC AUC', 0):.4f}
- PR AUC: {metrics.get('PR AUC', 0):.4f}"""

    report += f"""

### 4. Visualization Deliverables

#### Static Visualizations
- `dataset_distribution_analysis.png` - Comprehensive data distribution charts
- `advanced_wordclouds.png` - Enhanced word clouds with custom styling
- `ngram_analysis.png` - N-gram frequency analysis
- `comprehensive_phase_comparison.png` - Multi-phase performance comparison
- Individual model performance charts for each phase

#### Interactive Components  
- `interactive_dashboard.html` - Interactive Plotly dashboard with dynamic exploration

## 🔍 Key Findings

### Data Insights
1. **Class Imbalance**: Significant imbalance with {dataset_stats.get('spam_percentage', 0):.1f}% spam messages
2. **Message Characteristics**: Clear differentiation in length and linguistic patterns
3. **Vocabulary Patterns**: Distinct word usage patterns between spam and ham messages

### Performance Insights
1. **Model Evolution**: Progressive improvement across optimization phases
2. **Metric Balance**: Successful achievement of high precision and recall
3. **Robustness**: Consistent performance across different evaluation metrics

### Technical Implementation
- **Reproducibility**: All visualizations generated with fixed random seeds
- **Scalability**: Modular design for easy extension and modification
- **Quality**: Publication-ready plots with professional styling

## 📋 Recommendations

### For Model Development
1. Consider ensemble methods to combine strengths of different phases
2. Implement feature importance analysis using SHAP or permutation importance
3. Explore deep learning approaches for comparison

### For Production Deployment
1. Monitor for concept drift in spam patterns
2. Implement real-time performance tracking
3. Regular model retraining with new data

### For Further Analysis
1. Analyze misclassified examples for insights
2. Explore multilingual spam detection
3. Investigate temporal patterns in spam characteristics

## 🛠️ Technical Specifications

### Tools and Libraries
- **Data Processing**: pandas, numpy, scikit-learn
- **Static Visualization**: matplotlib, seaborn
- **Interactive Visualization**: plotly
- **Text Analysis**: WordCloud, CountVectorizer
- **Model Evaluation**: scikit-learn metrics

### Reproducibility
All analyses can be reproduced by running:
```bash
python train_phase4_visualization.py
```

### Output Files
Total files generated: {total_files}

## 📞 Conclusion

The comprehensive visualization analysis demonstrates the successful evolution of the SMS spam classification system through multiple optimization phases. The analysis provides actionable insights for both technical understanding and business presentation, supporting informed decision-making for production deployment.

The visualization suite serves as a robust foundation for ongoing model monitoring, performance evaluation, and stakeholder communication.

---

*This report was automatically generated as part of the Phase 4 analysis pipeline.*
*For questions or additional analysis, please refer to the interactive dashboard or contact the development team.*
"""

    return report

if __name__ == "__main__":
    main()