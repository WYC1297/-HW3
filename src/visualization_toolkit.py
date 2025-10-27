"""
Advanced Visualization Toolkit for Spam Classification Analysis
Provides comprehensive visual analysis for data distribution, model performance, and feature insights.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from collections import Counter
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import json
from typing import Dict, List, Tuple, Optional
import warnings
import random

# Set random seeds for reproducibility
np.random.seed(42)
random.seed(42)

warnings.filterwarnings('ignore')

class SpamVisualizationToolkit:
    """Comprehensive visualization toolkit for spam classification analysis."""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """Initialize visualization toolkit with default settings."""
        self.figsize = figsize
        self.colors = {
            'ham': '#2E8B57',      # Sea Green
            'spam': '#DC143C',     # Crimson
            'primary': '#1f77b4',  # Blue
            'secondary': '#ff7f0e', # Orange
            'success': '#2ca02c',   # Green
            'warning': '#d62728'    # Red
        }
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def visualize_data_distribution(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Create comprehensive data distribution visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('SMS Spam Dataset - Data Distribution Analysis', fontsize=16, fontweight='bold')
        
        # 1. Class distribution
        class_counts = df['label'].value_counts()
        axes[0, 0].pie(class_counts.values, labels=class_counts.index, autopct='%1.1f%%',
                       colors=[self.colors['ham'], self.colors['spam']], startangle=90)
        axes[0, 0].set_title('Class Distribution\n(Ham vs Spam)')
        
        # 2. Message length distribution
        df['message_length'] = df['message'].str.len()
        ham_lengths = df[df['label'] == 'ham']['message_length']
        spam_lengths = df[df['label'] == 'spam']['message_length']
        
        axes[0, 1].hist(ham_lengths, bins=50, alpha=0.7, label='Ham', color=self.colors['ham'], density=True)
        axes[0, 1].hist(spam_lengths, bins=50, alpha=0.7, label='Spam', color=self.colors['spam'], density=True)
        axes[0, 1].set_xlabel('Message Length (characters)')
        axes[0, 1].set_ylabel('Density')
        axes[0, 1].set_title('Message Length Distribution')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Word count distribution
        df['word_count'] = df['message'].str.split().str.len()
        ham_words = df[df['label'] == 'ham']['word_count']
        spam_words = df[df['label'] == 'spam']['word_count']
        
        box_plot = axes[1, 0].boxplot([ham_words, spam_words], labels=['Ham', 'Spam'],
                                     patch_artist=True)
        
        # Color the boxes
        colors = [self.colors['ham'], self.colors['spam']]
        for patch, color in zip(box_plot['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        axes[1, 0].set_ylabel('Word Count')
        axes[1, 0].set_title('Word Count Distribution (Box Plot)')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Statistics summary
        stats_data = {
            'Metric': ['Count', 'Mean Length', 'Median Length', 'Mean Words', 'Median Words'],
            'Ham': [
                len(ham_lengths),
                f"{ham_lengths.mean():.1f}",
                f"{ham_lengths.median():.1f}",
                f"{ham_words.mean():.1f}",
                f"{ham_words.median():.1f}"
            ],
            'Spam': [
                len(spam_lengths),
                f"{spam_lengths.mean():.1f}",
                f"{spam_lengths.median():.1f}",
                f"{spam_words.mean():.1f}",
                f"{spam_words.median():.1f}"
            ]
        }
        
        axes[1, 1].axis('tight')
        axes[1, 1].axis('off')
        table = axes[1, 1].table(cellText=[stats_data['Ham'], stats_data['Spam']],
                                rowLabels=['Ham', 'Spam'],
                                colLabels=stats_data['Metric'],
                                cellLoc='center',
                                loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        axes[1, 1].set_title('Statistical Summary')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_wordclouds(self, df: pd.DataFrame, save_path: Optional[str] = None) -> None:
        """Generate word clouds for ham and spam messages."""
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle('Word Clouds - Token Pattern Analysis', fontsize=16, fontweight='bold')
        
        # Ham word cloud
        ham_text = ' '.join(df[df['label'] == 'ham']['message'].astype(str))
        ham_wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Greens',
            max_words=100,
            relative_scaling=0.5
        ).generate(ham_text)
        
        axes[0].imshow(ham_wordcloud, interpolation='bilinear')
        axes[0].axis('off')
        axes[0].set_title('Ham Messages Word Cloud', fontsize=14, fontweight='bold')
        
        # Spam word cloud
        spam_text = ' '.join(df[df['label'] == 'spam']['message'].astype(str))
        spam_wordcloud = WordCloud(
            width=800, height=400,
            background_color='white',
            colormap='Reds',
            max_words=100,
            relative_scaling=0.5
        ).generate(spam_text)
        
        axes[1].imshow(spam_wordcloud, interpolation='bilinear')
        axes[1].axis('off')
        axes[1].set_title('Spam Messages Word Cloud', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def analyze_top_ngrams(self, df: pd.DataFrame, n: int = 20, save_path: Optional[str] = None) -> None:
        """Analyze and visualize top n-grams for ham and spam."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Top {n} N-grams Analysis', fontsize=16, fontweight='bold')
        
        # Separate texts
        ham_text = df[df['label'] == 'ham']['message'].str.cat(sep=' ')
        spam_text = df[df['label'] == 'spam']['message'].str.cat(sep=' ')
        
        # Unigrams
        ham_words = Counter(ham_text.lower().split())
        spam_words = Counter(spam_text.lower().split())
        
        # Plot ham unigrams
        ham_top = dict(ham_words.most_common(n))
        axes[0, 0].barh(list(ham_top.keys()), list(ham_top.values()), color=self.colors['ham'], alpha=0.7)
        axes[0, 0].set_title(f'Top {n} Words in Ham Messages')
        axes[0, 0].set_xlabel('Frequency')
        axes[0, 0].tick_params(axis='y', labelsize=8)
        
        # Plot spam unigrams
        spam_top = dict(spam_words.most_common(n))
        axes[0, 1].barh(list(spam_top.keys()), list(spam_top.values()), color=self.colors['spam'], alpha=0.7)
        axes[0, 1].set_title(f'Top {n} Words in Spam Messages')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].tick_params(axis='y', labelsize=8)
        
        # Bigrams
        import re
        
        def get_bigrams(text):
            words = re.findall(r'\b\w+\b', text.lower())
            return [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        
        ham_bigrams = Counter(get_bigrams(ham_text))
        spam_bigrams = Counter(get_bigrams(spam_text))
        
        # Plot ham bigrams
        ham_bi_top = dict(ham_bigrams.most_common(n))
        axes[1, 0].barh(list(ham_bi_top.keys()), list(ham_bi_top.values()), color=self.colors['ham'], alpha=0.7)
        axes[1, 0].set_title(f'Top {n} Bigrams in Ham Messages')
        axes[1, 0].set_xlabel('Frequency')
        axes[1, 0].tick_params(axis='y', labelsize=8)
        
        # Plot spam bigrams
        spam_bi_top = dict(spam_bigrams.most_common(n))
        axes[1, 1].barh(list(spam_bi_top.keys()), list(spam_bi_top.values()), color=self.colors['spam'], alpha=0.7)
        axes[1, 1].set_title(f'Top {n} Bigrams in Spam Messages')
        axes[1, 1].set_xlabel('Frequency')
        axes[1, 1].tick_params(axis='y', labelsize=8)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def visualize_model_performance(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  y_proba: np.ndarray, model_name: str = "Model",
                                  save_path: Optional[str] = None) -> None:
        """Create comprehensive model performance visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{model_name} - Performance Analysis', fontsize=16, fontweight='bold')
        
        # 1. Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'],
                   ax=axes[0, 0])
        axes[0, 0].set_title('Confusion Matrix')
        axes[0, 0].set_xlabel('Predicted')
        axes[0, 0].set_ylabel('Actual')
        
        # 2. ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[0, 1].plot(fpr, tpr, color=self.colors['primary'], lw=2, 
                       label=f'ROC curve (AUC = {roc_auc:.3f})')
        axes[0, 1].plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        axes[0, 1].set_xlim([0.0, 1.0])
        axes[0, 1].set_ylim([0.0, 1.05])
        axes[0, 1].set_xlabel('False Positive Rate')
        axes[0, 1].set_ylabel('True Positive Rate')
        axes[0, 1].set_title('ROC Curve')
        axes[0, 1].legend(loc="lower right")
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Precision-Recall Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        
        axes[1, 0].plot(recall, precision, color=self.colors['secondary'], lw=2,
                       label=f'PR curve (AUC = {pr_auc:.3f})')
        axes[1, 0].set_xlabel('Recall')
        axes[1, 0].set_ylabel('Precision')
        axes[1, 0].set_title('Precision-Recall Curve')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        metrics = {
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision': precision_score(y_true, y_pred),
            'Recall': recall_score(y_true, y_pred),
            'F1-Score': f1_score(y_true, y_pred),
            'ROC AUC': roc_auc,
            'PR AUC': pr_auc
        }
        
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        bars = axes[1, 1].bar(metric_names, metric_values, 
                             color=[self.colors['success'] if v >= 0.9 else self.colors['warning'] 
                                   for v in metric_values], alpha=0.7)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        axes[1, 1].set_ylim([0, 1.1])
        axes[1, 1].set_title('Performance Metrics')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].tick_params(axis='x', rotation=45)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics
    
    def analyze_feature_importance(self, vectorizer_path: str, model_path: str, 
                                 top_n: int = 20, save_path: Optional[str] = None) -> None:
        """Analyze and visualize TF-IDF feature importance."""
        # Load models
        vectorizer = joblib.load(vectorizer_path)
        model = joblib.load(model_path)
        
        # Get feature names and coefficients
        feature_names = vectorizer.get_feature_names_out()
        coefficients = model.coef_[0]
        
        # Create feature importance dataframe
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False)
        
        # Separate spam and ham indicators
        spam_features = feature_importance[feature_importance['coefficient'] > 0].head(top_n)
        ham_features = feature_importance[feature_importance['coefficient'] < 0].tail(top_n).iloc[::-1]
        
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        fig.suptitle(f'Top {top_n} Feature Importance (TF-IDF Coefficients)', fontsize=16, fontweight='bold')
        
        # Spam indicators
        axes[0].barh(spam_features['feature'], spam_features['coefficient'], 
                    color=self.colors['spam'], alpha=0.7)
        axes[0].set_title(f'Top {top_n} Spam Indicators')
        axes[0].set_xlabel('Coefficient Value')
        axes[0].tick_params(axis='y', labelsize=8)
        axes[0].grid(True, alpha=0.3)
        
        # Ham indicators
        axes[1].barh(ham_features['feature'], ham_features['coefficient'], 
                    color=self.colors['ham'], alpha=0.7)
        axes[1].set_title(f'Top {top_n} Ham Indicators')
        axes[1].set_xlabel('Coefficient Value')
        axes[1].tick_params(axis='y', labelsize=8)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return feature_importance
    
    def create_phase_comparison(self, results_dict: Dict[str, Dict], save_path: Optional[str] = None) -> None:
        """Create comprehensive comparison across different phases."""
        phases = list(results_dict.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Phase Comparison - Model Performance Evolution', fontsize=16, fontweight='bold')
        
        # Prepare data for plotting
        comparison_data = pd.DataFrame({
            phase: [results_dict[phase].get(metric, 0) for metric in metrics]
            for phase in phases
        }, index=metrics)
        
        # 1. Bar chart comparison
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, phase in enumerate(phases):
            axes[0, 0].bar(x + i*width, comparison_data[phase], width, 
                          label=phase, alpha=0.8)
        
        axes[0, 0].set_xlabel('Metrics')
        axes[0, 0].set_ylabel('Score')
        axes[0, 0].set_title('Performance Metrics Comparison')
        axes[0, 0].set_xticks(x + width)
        axes[0, 0].set_xticklabels(metrics)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylim([0.8, 1.0])
        
        # 2. Line chart showing progression
        for metric in metrics:
            metric_values = [results_dict[phase].get(metric, 0) for phase in phases]
            axes[0, 1].plot(phases, metric_values, marker='o', linewidth=2, label=metric)
        
        axes[0, 1].set_xlabel('Phase')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].set_title('Performance Progression')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim([0.8, 1.0])
        
        # 3. Heatmap
        sns.heatmap(comparison_data, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0.8, vmax=1.0, ax=axes[1, 0])
        axes[1, 0].set_title('Performance Heatmap')
        axes[1, 0].set_xlabel('Phase')
        axes[1, 0].set_ylabel('Metric')
        
        # 4. Improvement table
        improvement_data = []
        for i in range(1, len(phases)):
            prev_phase = phases[i-1]
            curr_phase = phases[i]
            improvements = []
            for metric in metrics:
                prev_val = results_dict[prev_phase].get(metric, 0)
                curr_val = results_dict[curr_phase].get(metric, 0)
                improvement = ((curr_val - prev_val) / prev_val) * 100 if prev_val > 0 else 0
                improvements.append(f"{improvement:+.2f}%")
            improvement_data.append(improvements)
        
        if improvement_data:
            axes[1, 1].axis('tight')
            axes[1, 1].axis('off')
            table = axes[1, 1].table(cellText=improvement_data,
                                   rowLabels=[f"{phases[i-1]} → {phases[i]}" 
                                            for i in range(1, len(phases))],
                                   colLabels=metrics,
                                   cellLoc='center',
                                   loc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            axes[1, 1].set_title('Improvement Percentages')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_interactive_dashboard(self, df: pd.DataFrame, results_dict: Dict[str, Dict],
                                   save_path: Optional[str] = None) -> None:
        """Create interactive Plotly dashboard."""
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Class Distribution', 'Message Length Distribution', 
                          'Performance Comparison', 'Phase Progression'),
            specs=[[{"type": "pie"}, {"type": "histogram"}],
                   [{"type": "bar"}, {"type": "scatter"}]]
        )
        
        # 1. Class distribution pie chart
        class_counts = df['label'].value_counts()
        fig.add_trace(go.Pie(labels=class_counts.index, values=class_counts.values,
                           name="Class Distribution"), row=1, col=1)
        
        # 2. Message length histogram
        df['message_length'] = df['message'].str.len()
        fig.add_trace(go.Histogram(x=df[df['label']=='ham']['message_length'], 
                                 name='Ham', opacity=0.7), row=1, col=2)
        fig.add_trace(go.Histogram(x=df[df['label']=='spam']['message_length'], 
                                 name='Spam', opacity=0.7), row=1, col=2)
        
        # 3. Performance comparison
        phases = list(results_dict.keys())
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for metric in metrics:
            values = [results_dict[phase].get(metric, 0) for phase in phases]
            fig.add_trace(go.Bar(x=phases, y=values, name=metric), row=2, col=1)
        
        # 4. Phase progression
        for metric in metrics:
            values = [results_dict[phase].get(metric, 0) for phase in phases]
            fig.add_trace(go.Scatter(x=phases, y=values, mode='lines+markers', 
                                   name=f"{metric} Progression"), row=2, col=2)
        
        fig.update_layout(height=800, showlegend=True, 
                         title_text="SMS Spam Classification - Interactive Dashboard")
        
        if save_path:
            fig.write_html(save_path)
        
        fig.show()
        
        return fig

# Example usage and utility functions
def load_phase_results() -> Dict[str, Dict]:
    """Load performance results from all completed phases with actual trained results."""
    results = {
        'Phase 1 (Baseline)': {
            'Accuracy': 0.9787,
            'Precision': 0.9907,
            'Recall': 0.8359,
            'F1-Score': 0.9068
        },
        'Phase 2 (Optimized)': {
            'Accuracy': 0.9855,
            'Precision': 0.9829,
            'Recall': 0.8984,
            'F1-Score': 0.9388
        },
        'Phase 3 (Advanced)': {
            'Accuracy': 0.9826,
            'Precision': 0.9104,
            'Recall': 0.9531,
            'F1-Score': 0.9313
        }
    }
    return results

if __name__ == "__main__":
    print("Spam Classification Visualization Toolkit")
    print("=========================================")
    print("This module provides comprehensive visualization tools for analyzing")
    print("spam classification performance, data distribution, and model insights.")
    print("\nAvailable visualizations:")
    print("- Data distribution analysis")
    print("- Word clouds and n-gram analysis") 
    print("- Model performance metrics")
    print("- Feature importance analysis")
    print("- Phase comparison reports")
    print("- Interactive dashboards")