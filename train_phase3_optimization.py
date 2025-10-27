"""
Phase 3: Advanced Model Optimization Training Script
Target: Achieve Precision & Recall > 93%
Based on Phase 2 with additional optimizations
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from imblearn.over_sampling import SMOTE
import joblib
from pathlib import Path
import re

class Phase3AdvancedClassifier:
    """Phase 3 advanced SMS spam classifier with SMOTE balancing"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.model = None
        self.vectorizer = None
        self.feature_selector = None
        self.smote = None
        self.is_fitted = False
        
    def advanced_preprocess(self, text: str) -> str:
        """Enhanced text preprocessing based on Phase 2 + additional techniques"""
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
    
    def train_phase3_model(self, messages: list, labels: np.ndarray):
        """Train Phase 3 advanced model with SMOTE balancing"""
        
        print("🚀 Training Phase 3 Advanced SVM Model")
        print("🎯 Target: Precision & Recall > 93%")
        print("="*60)
        
        # Enhanced preprocessing
        print("� Applying enhanced preprocessing...")
        processed_messages = [self.advanced_preprocess(msg) for msg in messages]
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed_messages, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )
        
        print(f"Training size: {len(X_train_text)}")
        print(f"Test size: {len(X_test_text)}")
        
        # Advanced vectorization with optimized parameters
        print("� Creating advanced TF-IDF features...")
        self.vectorizer = TfidfVectorizer(
            max_features=5000,  # More features for better coverage
            ngram_range=(1, 3),  # Include trigrams
            min_df=2,
            max_df=0.85,  # Slightly more restrictive
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )
        
        X_train_vec = self.vectorizer.fit_transform(X_train_text)
        X_test_vec = self.vectorizer.transform(X_test_text)
        
        print(f"Initial features: {X_train_vec.shape[1]}")
        
        # Feature selection with more features
        print("🎯 Applying advanced feature selection...")
        self.feature_selector = SelectKBest(chi2, k=min(1500, X_train_vec.shape[1]))
        X_train_selected = self.feature_selector.fit_transform(X_train_vec, y_train)
        X_test_selected = self.feature_selector.transform(X_test_vec)
        
        print(f"Selected features: {X_train_selected.shape[1]}")
        
        # Apply SMOTE for class balancing
        print("⚖️ Applying SMOTE class balancing...")
        self.smote = SMOTE(random_state=self.random_state)
        X_train_balanced, y_train_balanced = self.smote.fit_resample(X_train_selected, y_train)
        
        print(f"Balanced training size: {X_train_balanced.shape[0]}")
        print(f"Class distribution after SMOTE: {np.bincount(y_train_balanced)}")
        
        # Train SVM with grid search for optimal parameters
        print("🤖 Training advanced SVM with grid search...")
        
        param_grid = {
            'C': [1, 10, 50],
            'gamma': ['scale', 'auto', 0.001, 0.01],
            'kernel': ['rbf']
        }
        
        self.model = GridSearchCV(
            SVC(random_state=self.random_state, probability=True),
            param_grid,
            cv=3,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        self.model.fit(X_train_balanced, y_train_balanced)
        self.is_fitted = True
        
        # Evaluate on original test set
        print("📈 Evaluating advanced model...")
        y_pred = self.model.predict(X_test_selected)
        
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        
        # Detailed metrics
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        results = {
            'accuracy': accuracy,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'confusion_matrix': cm,
            'feature_count': X_train_selected.shape[1],
            'vocab_size': len(self.vectorizer.vocabulary_),
            'best_params': self.model.best_params_,
            'target_achieved': precision >= 0.93 and recall >= 0.93
        }
        
        print(f"\n📊 PHASE 3 ADVANCED MODEL RESULTS")
        print(f"{'='*45}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"Precision: {precision:.4f} {'✅' if precision >= 0.93 else '❌'}")
        print(f"Recall: {recall:.4f} {'✅' if recall >= 0.93 else '❌'}")
        print(f"Features used: {X_train_selected.shape[1]}")
        print(f"Best SVM params: {self.model.best_params_}")
        
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              Ham  Spam")
        print(f"Actual Ham    {cm[0,0]:4d}  {cm[0,1]:4d}")
        print(f"       Spam   {cm[1,0]:4d}  {cm[1,1]:4d}")
        
        target_status = "✅ TARGET ACHIEVED" if results['target_achieved'] else "❌ TARGET NOT ACHIEVED"
        print(f"\n🎯 {target_status}: Precision & Recall > 93%")
        
        return results
    
    def save_model(self, model_dir: str = "models", prefix: str = "phase3_final"):
        """Save the Phase 3 advanced model"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet")
        
        Path(model_dir).mkdir(exist_ok=True)
        
        joblib.dump(self.model, f"{model_dir}/{prefix}_svm_classifier.joblib")
        joblib.dump(self.vectorizer, f"{model_dir}/{prefix}_tfidf_vectorizer.joblib")
        joblib.dump(self.feature_selector, f"{model_dir}/{prefix}_feature_selector.joblib")
        
        print(f"✅ Phase 3 model saved to {model_dir}/")

def main():
    print("🚀 Phase 3: Advanced Model Optimization")
    print("🎯 Target: Precision & Recall > 93%")
    print("=" * 60)
    
    # Load data
    data_path = "data/sms_spam_no_header.csv"
    df = pd.read_csv(data_path, header=None, names=['label', 'message'])
    df = df.drop_duplicates()
    df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
    
    messages = df['message'].tolist()
    labels = df['label_binary'].values
    
    print(f"� Dataset loaded: {len(df)} messages")
    print(f"📋 Class distribution: {dict(df['label'].value_counts())}")
    
    # Train Phase 3 model
    classifier = Phase3AdvancedClassifier(random_state=42)
    results = classifier.train_phase3_model(messages, labels)
    
    # Save model
    classifier.save_model()
    
    return results

if __name__ == "__main__":
    results = main()
    if results['target_achieved']:
        print("\n🎉 Phase 3 optimization completed successfully!")
        print("✅ Models saved for future use")
    else:
        print("\n📝 Phase 3 models saved for analysis")
        print("💡 Consider additional optimization strategies")