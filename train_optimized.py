"""
Phase 2 Optimized Model - Advanced SVM with improved preprocessing
Combines all Phase 2 improvements and compares with baseline
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
import joblib
from pathlib import Path
import time
import re

class OptimizedSMSClassifier:
    """Phase 2 optimized SMS spam classifier"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.feature_selector = None
        self.is_fitted = False
        
    def advanced_preprocess(self, text: str) -> str:
        """Advanced text preprocessing based on Phase 2 improvements"""
        if pd.isna(text) or text == '':
            return ''
        
        text = str(text).lower()
        
        # Replace URLs with token
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', ' URL ', text)
        
        # Replace email addresses with token
        text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', ' EMAIL ', text)
        
        # Replace phone numbers with token
        text = re.sub(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b|\b\d{5,}\b', ' PHONE ', text)
        
        # Replace money amounts with token
        text = re.sub(r'[£$€¥₹][\d,]+\.?\d*|\b\d+(?:\.\d+)?\s*(?:pounds?|dollars?|euros?|cents?|pence)\b', ' MONEY ', text, flags=re.IGNORECASE)
        
        # Replace remaining numbers with token
        text = re.sub(r'\b\d+\b', ' NUMBER ', text)
        
        # Remove extra punctuation but keep apostrophes
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Handle basic contractions
        text = text.replace("won't", "will not")
        text = text.replace("can't", "cannot")
        text = text.replace("n't", " not")
        text = text.replace("'re", " are")
        text = text.replace("'ve", " have")
        text = text.replace("'ll", " will")
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def train_optimized_model(self, messages: list, labels: np.ndarray, use_grid_search: bool = False):
        """Train the optimized model with Phase 2 improvements"""
        
        print("🚀 Training Phase 2 Optimized SVM Model")
        print("="*50)
        
        # Preprocess messages
        print("🔄 Applying advanced preprocessing...")
        processed_messages = [self.advanced_preprocess(msg) for msg in messages]
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed_messages, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )
        
        print(f"Training size: {len(X_train_text)}")
        print(f"Test size: {len(X_test_text)}")
        
        # Advanced vectorization
        print("🔤 Creating optimized TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=3000,
            ngram_range=(1, 2),  # Include bigrams
            min_df=2,
            max_df=0.9,
            sublinear_tf=True,  # Apply sublinear tf scaling
            use_idf=True,
            smooth_idf=True
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train_text)
        X_test_vec = self.vectorizer.transform(X_test_text)
        
        print(f"Initial features: {X_train_vec.shape[1]}")
        
        # Feature selection
        print("🎯 Applying feature selection...")
        self.feature_selector = SelectKBest(chi2, k=min(1500, X_train_vec.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train_vec, y_train)
        X_test_selected = self.feature_selector.transform(X_test_vec)
        
        print(f"Selected features: {X_train_selected.shape[1]}")
        
        # Train SVM with optimized parameters
        print("🤖 Training optimized SVM...")
        
        if use_grid_search:
            print("Running grid search for hyperparameter optimization...")
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'linear']
            }
            
            self.model = GridSearchCV(
                SVC(random_state=self.random_state, probability=True),
                param_grid,
                cv=3,
                scoring='f1',
                n_jobs=-1
            )
        else:
            # Use optimized parameters based on common best practices
            self.model = SVC(
                kernel='rbf',
                C=10,  # Slightly higher C for better performance
                gamma='scale',
                random_state=self.random_state,
                probability=True
            )
        
        self.model.fit(X_train_selected, y_train)
        self.is_fitted = True
        
        # Evaluate
        print("📈 Evaluating optimized model...")
        y_pred = self.model.predict(X_test_selected)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        # Detailed metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'feature_count': X_train_selected.shape[1],
            'vocab_size': len(self.vectorizer.vocabulary_)
        }
        
        print(f"\n📊 OPTIMIZED MODEL RESULTS")
        print(f"{'='*35}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"Features used: {X_train_selected.shape[1]}")
        
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              Ham  Spam")
        print(f"Actual Ham    {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Spam   {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        if use_grid_search:
            print(f"\nBest parameters: {self.model.best_params_}")
        
        return results, X_test_selected, y_test, y_pred
    
    def save_model(self, model_dir: str = "models"):
        """Save the optimized model"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        Path(model_dir).mkdir(exist_ok=True)
        
        joblib.dump(self.model, f"{model_dir}/optimized_svm_classifier.joblib")
        joblib.dump(self.vectorizer, f"{model_dir}/optimized_tfidf_vectorizer.joblib")
        joblib.dump(self.feature_selector, f"{model_dir}/optimized_feature_selector.joblib")
        
        print(f"✅ Optimized model saved to {model_dir}/")

def compare_with_baseline():
    """Compare Phase 2 optimized model with Phase 1 baseline"""
    
    print("\n🔬 BASELINE vs OPTIMIZED COMPARISON")
    print("="*50)
    
    # Load data
    data_path = "data/sms_spam_no_header.csv"
    df = pd.read_csv(data_path, header=None, names=['label', 'message'])
    df = df.drop_duplicates()
    df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
    
    messages = df['message'].tolist()
    labels = df['label_binary'].values
    
    # Train baseline model (Phase 1)
    print("\n📊 Training baseline model (Phase 1)...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Simple preprocessing for baseline
    X_train_simple = [text.lower() for text in X_train_text]
    X_test_simple = [text.lower() for text in X_test_text]
    
    baseline_vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1,1))
    X_train_baseline = baseline_vectorizer.fit_transform(X_train_simple).toarray()
    X_test_baseline = baseline_vectorizer.transform(X_test_simple).toarray()
    
    baseline_svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
    baseline_svm.fit(X_train_baseline, y_train)
    
    y_pred_baseline = baseline_svm.predict(X_test_baseline)
    baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
    baseline_f1 = f1_score(y_test, y_pred_baseline)
    
    # Calculate baseline precision and recall
    from sklearn.metrics import precision_score, recall_score
    baseline_precision = precision_score(y_test, y_pred_baseline)
    baseline_recall = recall_score(y_test, y_pred_baseline)
    
    # Train optimized model (Phase 2)
    print("\n🚀 Training optimized model (Phase 2)...")
    optimizer = OptimizedSMSClassifier(random_state=42)
    optimized_results, _, _, _ = optimizer.train_optimized_model(messages, labels)
    
    # Comparison summary
    print(f"\n📈 FINAL COMPARISON SUMMARY")
    print(f"{'='*60}")
    print(f"{'Metric':<15} {'Baseline':<12} {'Optimized':<12} {'Improvement':<15}")
    print(f"{'-'*64}")
    
    acc_improvement = ((optimized_results['accuracy'] - baseline_accuracy) / baseline_accuracy) * 100
    f1_improvement = ((optimized_results['f1_score'] - baseline_f1) / baseline_f1) * 100
    precision_improvement = ((optimized_results['precision'] - baseline_precision) / baseline_precision) * 100
    recall_improvement = ((optimized_results['recall'] - baseline_recall) / baseline_recall) * 100
    
    print(f"{'Accuracy':<15} {baseline_accuracy:<12.4f} {optimized_results['accuracy']:<12.4f} {acc_improvement:<15.2f}%")
    print(f"{'F1-Score':<15} {baseline_f1:<12.4f} {optimized_results['f1_score']:<12.4f} {f1_improvement:<15.2f}%")
    print(f"{'Precision':<15} {baseline_precision:<12.4f} {optimized_results['precision']:<12.4f} {precision_improvement:<15.2f}%")
    print(f"{'Recall':<15} {baseline_recall:<12.4f} {optimized_results['recall']:<12.4f} {recall_improvement:<15.2f}%")
    print(f"{'Features':<15} {X_train_baseline.shape[1]:<12} {optimized_results['feature_count']:<12} {'+' + str(optimized_results['feature_count'] - X_train_baseline.shape[1]):<15}")
    
    # Save optimized model
    optimizer.save_model()
    
    return {
        'baseline': {
            'accuracy': baseline_accuracy, 
            'f1_score': baseline_f1,
            'precision': baseline_precision,
            'recall': baseline_recall
        },
        'optimized': optimized_results,
        'improvements': {
            'accuracy': acc_improvement, 
            'f1_score': f1_improvement,
            'precision': precision_improvement,
            'recall': recall_improvement
        }
    }

if __name__ == "__main__":
    results = compare_with_baseline()
    print(f"\n✅ Phase 2 optimization and comparison completed!")