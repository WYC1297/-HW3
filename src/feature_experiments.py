"""
Feature extraction experiments for Phase 2
Compare different vectorization methods and parameters
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.feature_selection import SelectKBest, chi2
from src.advanced_preprocessor import AdvancedTextPreprocessor
import time
from typing import Dict, List, Tuple, Any

class FeatureExtractionExperiments:
    """Class for conducting feature extraction experiments"""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.results = []
        
    def load_and_prepare_data(self) -> Tuple[List[str], np.ndarray]:
        """Load and prepare the SMS spam dataset"""
        
        print("📥 Loading SMS spam dataset...")
        data_path = "data/sms_spam_no_header.csv"
        df = pd.read_csv(data_path, header=None, names=['label', 'message'])
        
        # Clean data
        df = df.drop_duplicates()
        df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
        
        messages = df['message'].tolist()
        labels = df['label_binary'].values
        
        print(f"Dataset loaded: {len(messages)} messages")
        return messages, labels
    
    def experiment_vectorizers(self, messages: List[str], labels: np.ndarray) -> Dict[str, Any]:
        """
        Experiment with different vectorization methods
        
        Args:
            messages: List of raw messages
            labels: Binary labels (0=ham, 1=spam)
            
        Returns:
            Dict containing experiment results
        """
        print("\n🧪 Running vectorization experiments...")
        
        # Initialize advanced preprocessor
        preprocessor = AdvancedTextPreprocessor(use_stemming=True, use_lemmatization=False)
        
        # Preprocess all messages
        print("🔄 Preprocessing messages...")
        processed_messages = [preprocessor.preprocess_text(msg) for msg in messages]
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            processed_messages, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )
        
        # Define vectorizer configurations to test
        vectorizer_configs = [
            {
                'name': 'TF-IDF_Unigrams_1K',
                'vectorizer': TfidfVectorizer(max_features=1000, ngram_range=(1, 1), min_df=2, max_df=0.95)
            },
            {
                'name': 'TF-IDF_Bigrams_2K',
                'vectorizer': TfidfVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2, max_df=0.95)
            },
            {
                'name': 'TF-IDF_Trigrams_3K',
                'vectorizer': TfidfVectorizer(max_features=3000, ngram_range=(1, 3), min_df=2, max_df=0.95)
            },
            {
                'name': 'TF-IDF_Optimized_5K',
                'vectorizer': TfidfVectorizer(
                    max_features=5000, 
                    ngram_range=(1, 2), 
                    min_df=3, 
                    max_df=0.9,
                    sublinear_tf=True,  # Apply sublinear tf scaling
                    use_idf=True,
                    smooth_idf=True
                )
            },
            {
                'name': 'Count_Bigrams_2K',
                'vectorizer': CountVectorizer(max_features=2000, ngram_range=(1, 2), min_df=2, max_df=0.95)
            }
        ]
        
        experiment_results = []
        
        for config in vectorizer_configs:
            print(f"\n🔧 Testing {config['name']}...")
            
            start_time = time.time()
            
            # Vectorize
            vectorizer = config['vectorizer']
            X_train = vectorizer.fit_transform(X_train_text).toarray()
            X_test = vectorizer.transform(X_test_text).toarray()
            
            # Train SVM
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=self.random_state)
            svm.fit(X_train, y_train)
            
            # Evaluate
            y_pred = svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # Calculate confusion matrix metrics
            cm = confusion_matrix(y_test, y_pred)
            tn, fp, fn, tp = cm.ravel()
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            end_time = time.time()
            
            result = {
                'name': config['name'],
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'feature_count': X_train.shape[1],
                'vocab_size': len(vectorizer.vocabulary_),
                'training_time': end_time - start_time,
                'confusion_matrix': cm
            }
            
            experiment_results.append(result)
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Features: {X_train.shape[1]}")
            print(f"  Time: {end_time - start_time:.2f}s")
        
        return {
            'results': experiment_results,
            'train_size': len(X_train_text),
            'test_size': len(X_test_text)
        }
    
    def compare_preprocessing_methods(self, messages: List[str], labels: np.ndarray) -> Dict[str, Any]:
        """
        Compare different preprocessing approaches
        
        Args:
            messages: List of raw messages
            labels: Binary labels
            
        Returns:
            Dict containing comparison results
        """
        print("\n🔬 Comparing preprocessing methods...")
        
        # Split data first
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            messages, labels, test_size=0.2, random_state=self.random_state, stratify=labels
        )
        
        # Define preprocessing configurations
        preprocessing_configs = [
            {
                'name': 'Basic_Lowercase',
                'preprocessor': lambda text: text.lower()
            },
            {
                'name': 'Advanced_Stemming',
                'preprocessor': AdvancedTextPreprocessor(use_stemming=True, use_lemmatization=False).preprocess_text
            },
            {
                'name': 'Advanced_Lemmatization',
                'preprocessor': AdvancedTextPreprocessor(use_stemming=False, use_lemmatization=True).preprocess_text
            }
        ]
        
        comparison_results = []
        
        for config in preprocessing_configs:
            print(f"\n🔧 Testing {config['name']}...")
            
            start_time = time.time()
            
            # Preprocess
            preprocessor = config['preprocessor']
            X_train_processed = [preprocessor(text) for text in X_train_text]
            X_test_processed = [preprocessor(text) for text in X_test_text]
            
            # Vectorize with consistent parameters
            vectorizer = TfidfVectorizer(
                max_features=2000,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            X_train = vectorizer.fit_transform(X_train_processed).toarray()
            X_test = vectorizer.transform(X_test_processed).toarray()
            
            # Train and evaluate
            svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=self.random_state)
            svm.fit(X_train, y_train)
            
            y_pred = svm.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            end_time = time.time()
            
            result = {
                'name': config['name'],
                'accuracy': accuracy,
                'f1_score': f1,
                'vocab_size': len(vectorizer.vocabulary_),
                'processing_time': end_time - start_time
            }
            
            comparison_results.append(result)
            
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Vocab size: {len(vectorizer.vocabulary_)}")
        
        return {'preprocessing_results': comparison_results}
    
    def print_experiment_summary(self, vectorizer_results: Dict, preprocessing_results: Dict):
        """Print a summary of all experiments"""
        
        print("\n" + "="*60)
        print("📊 FEATURE EXTRACTION EXPERIMENT SUMMARY")
        print("="*60)
        
        # Vectorizer experiments
        print("\n🔧 Vectorization Method Comparison:")
        print(f"{'Method':<20} {'Accuracy':<10} {'F1-Score':<10} {'Features':<10} {'Time(s)':<8}")
        print("-" * 68)
        
        best_accuracy = 0
        best_f1 = 0
        best_method = ""
        
        for result in vectorizer_results['results']:
            print(f"{result['name']:<20} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} "
                  f"{result['feature_count']:<10} {result['training_time']:<8.2f}")
            
            if result['f1_score'] > best_f1:
                best_f1 = result['f1_score']
                best_accuracy = result['accuracy']
                best_method = result['name']
        
        print(f"\n🏆 Best performing method: {best_method}")
        print(f"   Accuracy: {best_accuracy:.4f}, F1-Score: {best_f1:.4f}")
        
        # Preprocessing comparison
        print("\n🔬 Preprocessing Method Comparison:")
        print(f"{'Method':<20} {'Accuracy':<10} {'F1-Score':<10} {'Vocab Size':<12}")
        print("-" * 52)
        
        for result in preprocessing_results['preprocessing_results']:
            print(f"{result['name']:<20} {result['accuracy']:<10.4f} {result['f1_score']:<10.4f} "
                  f"{result['vocab_size']:<12}")

def main():
    """Run all feature extraction experiments"""
    
    print("🚀 Starting Feature Extraction Experiments")
    print("="*50)
    
    # Initialize experiment runner
    experiments = FeatureExtractionExperiments(random_state=42)
    
    # Load data
    messages, labels = experiments.load_and_prepare_data()
    
    # Run vectorizer experiments
    vectorizer_results = experiments.experiment_vectorizers(messages, labels)
    
    # Run preprocessing comparison
    preprocessing_results = experiments.compare_preprocessing_methods(messages, labels)
    
    # Print summary
    experiments.print_experiment_summary(vectorizer_results, preprocessing_results)
    
    print(f"\n✅ All experiments completed!")
    
    return vectorizer_results, preprocessing_results

if __name__ == "__main__":
    vectorizer_results, preprocessing_results = main()