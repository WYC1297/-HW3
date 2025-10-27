#!/usr/bin/env python3
"""
SMS Spam Classification Visualization CLI Script
Generates comprehensive visualizations and analysis reports
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import sys
import argparse
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    precision_recall_curve, precision_score, recall_score, f1_score
)
from sklearn.feature_extraction.text import TfidfVectorizer
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
current_dir = Path(__file__).parent.parent
sys.path.append(str(current_dir / 'src'))

try:
    from data_loader import SMSDataLoader
    from advanced_preprocessor import AdvancedTextPreprocessor
except ImportError as e:
    print(f"❌ Required modules not found: {e}")
    print("Please ensure src/ directory exists with proper modules.")
    sys.exit(1)

class SpamVisualizationGenerator:
    """Comprehensive visualization generator for SMS spam classification"""
    
    def __init__(self, output_dir="reports/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_loader = SMSDataLoader()
        self.preprocessor = AdvancedTextPreprocessor()
        
        # Set style for matplotlib
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
        # Colors for consistency
        self.colors = {
            'spam': '#FF6B6B',
            'ham': '#4ECDC4',
            'primary': '#45B7D1',
            'secondary': '#96CEB4'
        }
    
    def load_data(self):
        """Load and prepare the dataset"""
        print("📊 Loading and cleaning dataset...")
        df = self.data_loader.load_data()
        if df is None:
            raise ValueError("Failed to load dataset")
        
        df = self.data_loader.basic_clean()
        print(f"✅ Dataset loaded: {len(df)} messages")
        return df
    
    def load_best_model(self):
        """Load the best performing model"""
        model_configs = [
            {
                'name': 'Advanced Model',
                'model_path': 'models/phase3_final_svm_classifier.joblib',
                'vectorizer_path': 'models/phase3_final_tfidf_vectorizer.joblib',
                'feature_selector_path': 'models/phase3_final_feature_selector.joblib'
            },
            {
                'name': 'Optimized Model',
                'model_path': 'models/optimized_svm_classifier.joblib',
                'vectorizer_path': 'models/optimized_tfidf_vectorizer.joblib',
                'feature_selector_path': 'models/optimized_feature_selector.joblib'
            },
            {
                'name': 'Baseline Model',
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
                    model = joblib.load(model_path)
                    vectorizer = joblib.load(vectorizer_path)
                    feature_selector = None
                    
                    if config['feature_selector_path']:
                        selector_path = Path(config['feature_selector_path'])
                        if selector_path.exists():
                            feature_selector = joblib.load(selector_path)
                    
                    print(f"✅ Loaded {config['name']}")
                    return model, vectorizer, feature_selector, config['name']
                    
            except Exception as e:
                print(f"⚠️ Failed to load {config['name']}: {e}")
                continue
        
        raise ValueError("No models could be loaded")
    
    def generate_class_distribution(self, df):
        """Generate class distribution visualizations"""
        print("📊 Generating class distribution charts...")
        
        # Calculate statistics
        class_counts = df['label'].value_counts()
        total = len(df)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart
        bars = ax1.bar(class_counts.index, class_counts.values, 
                      color=[self.colors['ham'], self.colors['spam']])
        ax1.set_title('SMS Message Class Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Message Type')
        ax1.set_ylabel('Count')
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, count in zip(bars, class_counts.values):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 50,
                    f'{count:,}\n({count/total*100:.1f}%)',
                    ha='center', va='bottom', fontweight='bold')
        
        # Pie chart
        wedges, texts, autotexts = ax2.pie(class_counts.values, labels=class_counts.index, 
                                          autopct='%1.1f%%', startangle=90,
                                          colors=[self.colors['ham'], self.colors['spam']])
        ax2.set_title('Class Distribution Percentage', fontsize=14, fontweight='bold')
        
        # Enhance pie chart text
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'class_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive Plotly version
        fig_plotly = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Message Count by Class', 'Class Distribution'),
            specs=[[{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Bar chart
        fig_plotly.add_trace(
            go.Bar(x=class_counts.index, y=class_counts.values,
                  marker_color=[self.colors['ham'], self.colors['spam']],
                  text=[f'{count:,}<br>({count/total*100:.1f}%)' for count in class_counts.values],
                  textposition='outside'),
            row=1, col=1
        )
        
        # Pie chart
        fig_plotly.add_trace(
            go.Pie(labels=class_counts.index, values=class_counts.values,
                  marker_colors=[self.colors['ham'], self.colors['spam']]),
            row=1, col=2
        )
        
        fig_plotly.update_layout(
            title_text="SMS Spam Classification - Class Distribution Analysis",
            height=500,
            showlegend=False
        )
        
        fig_plotly.write_html(str(self.output_dir / 'class_distribution_interactive.html'))
        
        return class_counts
    
    def generate_token_frequency_analysis(self, df):
        """Generate top token frequency charts for spam vs ham"""
        print("🔤 Generating token frequency analysis...")
        
        # Prepare data
        spam_messages = df[df['label'] == 'spam']['message']
        ham_messages = df[df['label'] == 'ham']['message']
        
        # Create TF-IDF vectorizer for token analysis
        vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        
        # Fit on all data
        all_messages = pd.concat([spam_messages, ham_messages])
        vectorizer.fit(all_messages)
        
        # Transform spam and ham separately
        spam_tfidf = vectorizer.transform(spam_messages)
        ham_tfidf = vectorizer.transform(ham_messages)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Calculate mean TF-IDF scores
        spam_scores = np.array(spam_tfidf.mean(axis=0)).flatten()
        ham_scores = np.array(ham_tfidf.mean(axis=0)).flatten()
        
        # Get top tokens for each class
        top_n = 20
        spam_top_indices = spam_scores.argsort()[-top_n:][::-1]
        ham_top_indices = ham_scores.argsort()[-top_n:][::-1]
        
        spam_top_tokens = [(feature_names[i], spam_scores[i]) for i in spam_top_indices]
        ham_top_tokens = [(feature_names[i], ham_scores[i]) for i in ham_top_indices]
        
        # Create visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Spam tokens
        spam_tokens, spam_values = zip(*spam_top_tokens)
        y_pos = np.arange(len(spam_tokens))
        
        bars1 = ax1.barh(y_pos, spam_values, color=self.colors['spam'], alpha=0.8)
        ax1.set_yticks(y_pos)
        ax1.set_yticklabels(spam_tokens)
        ax1.set_xlabel('Mean TF-IDF Score')
        ax1.set_title('Top 20 Tokens in SPAM Messages', fontsize=14, fontweight='bold')
        ax1.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars1, spam_values)):
            ax1.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', ha='left', fontsize=9)
        
        # Ham tokens
        ham_tokens, ham_values = zip(*ham_top_tokens)
        
        bars2 = ax2.barh(y_pos, ham_values, color=self.colors['ham'], alpha=0.8)
        ax2.set_yticks(y_pos)
        ax2.set_yticklabels(ham_tokens)
        ax2.set_xlabel('Mean TF-IDF Score')
        ax2.set_title('Top 20 Tokens in HAM Messages', fontsize=14, fontweight='bold')
        ax2.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars2, ham_values)):
            ax2.text(value + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{value:.3f}', va='center', ha='left', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'token_frequency_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive Plotly version
        fig_plotly = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Top Tokens in SPAM', 'Top Tokens in HAM'),
            specs=[[{"type": "bar"}, {"type": "bar"}]]
        )
        
        # Spam tokens
        fig_plotly.add_trace(
            go.Bar(y=spam_tokens, x=spam_values, orientation='h',
                  marker_color=self.colors['spam'], name='SPAM',
                  text=[f'{v:.3f}' for v in spam_values],
                  textposition='outside'),
            row=1, col=1
        )
        
        # Ham tokens
        fig_plotly.add_trace(
            go.Bar(y=ham_tokens, x=ham_values, orientation='h',
                  marker_color=self.colors['ham'], name='HAM',
                  text=[f'{v:.3f}' for v in ham_values],
                  textposition='outside'),
            row=1, col=2
        )
        
        fig_plotly.update_layout(
            title_text="Token Frequency Analysis - SPAM vs HAM",
            height=600,
            showlegend=False
        )
        
        fig_plotly.write_html(str(self.output_dir / 'token_frequency_interactive.html'))
        
        return spam_top_tokens, ham_top_tokens
    
    def generate_model_performance_analysis(self, df, model, vectorizer, feature_selector, model_name):
        """Generate comprehensive model performance visualizations"""
        print("🎯 Generating model performance analysis...")
        
        # Prepare data
        X = df['message']
        y = df['label']
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Transform data
        X_test_transformed = vectorizer.transform(X_test)
        if feature_selector is not None:
            X_test_transformed = feature_selector.transform(X_test_transformed)
        
        # Convert to dense if needed
        if hasattr(X_test_transformed, 'toarray'):
            X_test_transformed = X_test_transformed.toarray()
        
        # Make predictions
        y_pred = model.predict(X_test_transformed)
        
        # Convert to binary
        y_test_binary = (y_test == 'spam').astype(int)
        y_pred_binary = (y_pred == 'spam').astype(int) if isinstance(y_pred[0], str) else y_pred
        
        # Get prediction scores
        if hasattr(model, 'decision_function'):
            y_scores = model.decision_function(X_test_transformed)
        else:
            y_proba = model.predict_proba(X_test_transformed)
            y_scores = y_proba[:, 1]
        
        # 1. Confusion Matrix
        self._plot_confusion_matrix(y_test_binary, y_pred_binary, model_name)
        
        # 2. ROC and PR Curves
        self._plot_roc_and_pr_curves(y_test_binary, y_scores, model_name)
        
        # 3. Threshold Sweep Analysis
        self._plot_threshold_sweep(y_test_binary, y_scores, model_name)
        
        return y_test_binary, y_pred_binary, y_scores
    
    def _plot_confusion_matrix(self, y_true, y_pred, model_name):
        """Plot confusion matrix"""
        cm = confusion_matrix(y_true, y_pred)
        
        # Calculate percentages
        cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        
        # Add custom annotations with counts and percentages
        for i in range(len(cm)):
            for j in range(len(cm[0])):
                text = f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)'
                ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                       fontsize=12, fontweight='bold', color='black' if cm[i, j] < cm.max()/2 else 'white')
        
        ax.set_title(f'Confusion Matrix - {model_name}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive version
        fig_plotly = go.Figure(data=go.Heatmap(
            z=cm,
            x=['Ham', 'Spam'],
            y=['Ham', 'Spam'],
            colorscale='Blues',
            text=[[f'{cm[i, j]}<br>({cm_percent[i, j]:.1f}%)' for j in range(len(cm[0]))] for i in range(len(cm))],
            texttemplate="%{text}",
            textfont={"size": 14},
            hovertemplate='True: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>'
        ))
        
        fig_plotly.update_layout(
            title=f'Confusion Matrix - {model_name}',
            xaxis_title='Predicted Label',
            yaxis_title='True Label',
            height=500
        )
        
        fig_plotly.write_html(str(self.output_dir / f'confusion_matrix_{model_name.lower().replace(" ", "_")}_interactive.html'))
    
    def _plot_roc_and_pr_curves(self, y_true, y_scores, model_name):
        """Plot ROC and Precision-Recall curves"""
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate PR curve
        precision, recall, _ = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)
        
        # Create matplotlib version
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # ROC Curve
        ax1.plot(fpr, tpr, color=self.colors['primary'], lw=3, 
                label=f'ROC Curve (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Classifier')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.set_title(f'ROC Curve - {model_name}', fontweight='bold')
        ax1.legend(loc="lower right")
        ax1.grid(alpha=0.3)
        
        # PR Curve
        baseline_precision = np.sum(y_true) / len(y_true)
        ax2.plot(recall, precision, color=self.colors['secondary'], lw=3,
                label=f'PR Curve (AUC = {pr_auc:.3f})')
        ax2.axhline(y=baseline_precision, color='gray', lw=2, linestyle='--',
                   label=f'Baseline ({baseline_precision:.3f})')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.set_title(f'Precision-Recall Curve - {model_name}', fontweight='bold')
        ax2.legend(loc="lower left")
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'roc_pr_curves_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive Plotly version
        fig_plotly = make_subplots(
            rows=1, cols=2,
            subplot_titles=(f'ROC Curve (AUC = {roc_auc:.3f})', f'Precision-Recall Curve (AUC = {pr_auc:.3f})')
        )
        
        # ROC Curve
        fig_plotly.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', line=dict(width=3, color=self.colors['primary']),
                      name=f'ROC (AUC = {roc_auc:.3f})',
                      hovertemplate='FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>'),
            row=1, col=1
        )
        fig_plotly.add_trace(
            go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(dash='dash', color='gray'),
                      name='Random', showlegend=False),
            row=1, col=1
        )
        
        # PR Curve
        fig_plotly.add_trace(
            go.Scatter(x=recall, y=precision, mode='lines', line=dict(width=3, color=self.colors['secondary']),
                      name=f'PR (AUC = {pr_auc:.3f})',
                      hovertemplate='Recall: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'),
            row=1, col=2
        )
        fig_plotly.add_trace(
            go.Scatter(x=[0, 1], y=[baseline_precision, baseline_precision], mode='lines',
                      line=dict(dash='dash', color='gray'),
                      name=f'Baseline ({baseline_precision:.3f})', showlegend=False),
            row=1, col=2
        )
        
        fig_plotly.update_xaxes(title_text="False Positive Rate", row=1, col=1)
        fig_plotly.update_yaxes(title_text="True Positive Rate", row=1, col=1)
        fig_plotly.update_xaxes(title_text="Recall", row=1, col=2)
        fig_plotly.update_yaxes(title_text="Precision", row=1, col=2)
        
        fig_plotly.update_layout(
            title_text=f"Model Performance Curves - {model_name}",
            height=500
        )
        
        fig_plotly.write_html(str(self.output_dir / f'roc_pr_curves_{model_name.lower().replace(" ", "_")}_interactive.html'))
        
        return roc_auc, pr_auc
    
    def _plot_threshold_sweep(self, y_true, y_scores, model_name):
        """Generate threshold sweep analysis"""
        thresholds = np.linspace(0, 1, 101)
        precisions = []
        recalls = []
        f1_scores = []
        
        for threshold in thresholds:
            y_pred_thresh = (y_scores >= threshold).astype(int)
            
            # Handle edge cases
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
        
        # Create DataFrame for analysis
        threshold_df = pd.DataFrame({
            'Threshold': thresholds,
            'Precision': precisions,
            'Recall': recalls,
            'F1-Score': f1_scores
        })
        
        # Save to CSV
        threshold_df.to_csv(self.output_dir / f'threshold_sweep_{model_name.lower().replace(" ", "_")}.csv', index=False)
        
        # Plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.plot(thresholds, precisions, label='Precision', linewidth=3, color=self.colors['primary'])
        ax.plot(thresholds, recalls, label='Recall', linewidth=3, color=self.colors['secondary'])
        ax.plot(thresholds, f1_scores, label='F1-Score', linewidth=3, color=self.colors['spam'])
        
        # Find optimal F1 threshold
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        
        ax.axvline(x=optimal_threshold, color='red', linestyle='--', alpha=0.7,
                  label=f'Optimal F1 Threshold = {optimal_threshold:.3f}')
        ax.scatter([optimal_threshold], [optimal_f1], color='red', s=100, zorder=5)
        
        ax.set_xlabel('Classification Threshold')
        ax.set_ylabel('Score')
        ax.set_title(f'Threshold Sweep Analysis - {model_name}', fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        plt.tight_layout()
        plt.savefig(self.output_dir / f'threshold_sweep_{model_name.lower().replace(" ", "_")}.png', 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        # Interactive Plotly version
        fig_plotly = go.Figure()
        
        fig_plotly.add_trace(go.Scatter(x=thresholds, y=precisions, mode='lines', 
                                       name='Precision', line=dict(width=3, color=self.colors['primary']),
                                       hovertemplate='Threshold: %{x:.3f}<br>Precision: %{y:.3f}<extra></extra>'))
        fig_plotly.add_trace(go.Scatter(x=thresholds, y=recalls, mode='lines',
                                       name='Recall', line=dict(width=3, color=self.colors['secondary']),
                                       hovertemplate='Threshold: %{x:.3f}<br>Recall: %{y:.3f}<extra></extra>'))
        fig_plotly.add_trace(go.Scatter(x=thresholds, y=f1_scores, mode='lines',
                                       name='F1-Score', line=dict(width=3, color=self.colors['spam']),
                                       hovertemplate='Threshold: %{x:.3f}<br>F1-Score: %{y:.3f}<extra></extra>'))
        
        # Add optimal threshold line
        fig_plotly.add_vline(x=optimal_threshold, line_dash="dash", line_color="red",
                            annotation_text=f"Optimal F1: {optimal_threshold:.3f}")
        
        fig_plotly.update_layout(
            title=f'Threshold Sweep Analysis - {model_name}',
            xaxis_title='Classification Threshold',
            yaxis_title='Score',
            height=600,
            hovermode='x unified'
        )
        
        fig_plotly.write_html(str(self.output_dir / f'threshold_sweep_{model_name.lower().replace(" ", "_")}_interactive.html'))
        
        return threshold_df, optimal_threshold
    
    def generate_summary_report(self, df, model_name, class_counts, roc_auc, pr_auc, optimal_threshold):
        """Generate a comprehensive summary report"""
        print("📋 Generating summary report...")
        
        total_messages = len(df)
        spam_count = class_counts['spam']
        ham_count = class_counts['ham']
        spam_percentage = (spam_count / total_messages) * 100
        
        report = f"""
# SMS Spam Classification - Analysis Report

## Dataset Overview
- **Total Messages**: {total_messages:,}
- **Spam Messages**: {spam_count:,} ({spam_percentage:.1f}%)
- **Ham Messages**: {ham_count:,} ({100-spam_percentage:.1f}%)

## Model Performance
- **Model Used**: {model_name}
- **ROC AUC**: {roc_auc:.4f}
- **PR AUC**: {pr_auc:.4f}
- **Optimal Threshold**: {optimal_threshold:.3f}

## Generated Visualizations
1. **Class Distribution**: `class_distribution.png`
2. **Token Frequency Analysis**: `token_frequency_analysis.png`
3. **Confusion Matrix**: `confusion_matrix_{model_name.lower().replace(" ", "_")}.png`
4. **ROC & PR Curves**: `roc_pr_curves_{model_name.lower().replace(" ", "_")}.png`
5. **Threshold Sweep**: `threshold_sweep_{model_name.lower().replace(" ", "_")}.png`

## Interactive Visualizations
- HTML files with interactive Plotly charts are also generated for web viewing.

## Data Files
- **Threshold Analysis**: `threshold_sweep_{model_name.lower().replace(" ", "_")}.csv`

---
Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
        """
        
        with open(self.output_dir / 'analysis_summary.md', 'w') as f:
            f.write(report.strip())
        
        print(f"📊 Summary report saved to: {self.output_dir / 'analysis_summary.md'}")
    
    def run_complete_analysis(self):
        """Run the complete visualization pipeline"""
        print("🚀 Starting comprehensive SMS spam visualization analysis...")
        print("=" * 60)
        
        try:
            # Load data and model
            df = self.load_data()
            model, vectorizer, feature_selector, model_name = self.load_best_model()
            
            # Generate visualizations
            class_counts = self.generate_class_distribution(df)
            spam_tokens, ham_tokens = self.generate_token_frequency_analysis(df)
            y_true, y_pred, y_scores = self.generate_model_performance_analysis(
                df, model, vectorizer, feature_selector, model_name
            )
            
            # Get performance metrics from ROC/PR analysis
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            precision, recall, _ = precision_recall_curve(y_true, y_scores)
            pr_auc = auc(recall, precision)
            
            # Generate threshold analysis
            threshold_df, optimal_threshold = self._plot_threshold_sweep(y_true, y_scores, model_name)
            
            # Generate summary report
            self.generate_summary_report(df, model_name, class_counts, roc_auc, pr_auc, optimal_threshold)
            
            print("=" * 60)
            print("✅ Analysis complete! Check the following files:")
            print(f"📂 Output directory: {self.output_dir.absolute()}")
            
            # List generated files
            generated_files = list(self.output_dir.glob('*'))
            for file in sorted(generated_files):
                print(f"   📄 {file.name}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error during analysis: {e}")
            import traceback
            traceback.print_exc()
            return False

def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(description='Generate comprehensive SMS spam classification visualizations')
    parser.add_argument('--output-dir', '-o', default='reports/visualizations',
                       help='Output directory for visualizations (default: reports/visualizations)')
    parser.add_argument('--format', '-f', choices=['png', 'html', 'both'], default='both',
                       help='Output format for visualizations (default: both)')
    
    args = parser.parse_args()
    
    # Create visualization generator
    generator = SpamVisualizationGenerator(output_dir=args.output_dir)
    
    # Run analysis
    success = generator.run_complete_analysis()
    
    if success:
        print("\n🎉 Visualization generation completed successfully!")
        return 0
    else:
        print("\n❌ Visualization generation failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())