"""
Machine Learning models for SMS spam classification
Currently implements SVM baseline model
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import os
from pathlib import Path
from typing import Tuple, Dict, Any

class SVMSpamClassifier:
    """SVM-based spam classifier"""
    
    def __init__(self, random_state: int = 42):
        """
        Initialize SVM classifier
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.model = SVC(
            kernel='rbf',
            random_state=random_state,
            probability=True  # Enable probability predictions
        )
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            stop_words='english'
        )
        self.is_fitted = False
        
    def prepare_data(self, df: pd.DataFrame, test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training and testing
        
        Args:
            df: DataFrame with 'message' and 'label_binary' columns
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("📊 Preparing data for SVM training...")
        
        # Basic text preprocessing (simple approach for Phase 1)
        messages = df['message'].str.lower().fillna('')
        labels = df['label_binary'].values
        
        print(f"Dataset shape: {messages.shape}")
        print(f"Label distribution: {np.bincount(labels)}")
        
        # Split data
        X_train_text, X_test_text, y_train, y_test = train_test_split(
            messages, labels, 
            test_size=test_size, 
            random_state=self.random_state,
            stratify=labels
        )
        
        print(f"Training set size: {len(X_train_text)}")
        print(f"Test set size: {len(X_test_text)}")
        
        # Vectorize texts
        print("🔄 Vectorizing texts with TF-IDF...")
        X_train = self.vectorizer.fit_transform(X_train_text).toarray()
        X_test = self.vectorizer.transform(X_test_text).toarray()
        
        print(f"Feature matrix shape: {X_train.shape}")
        print(f"Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return X_train, X_test, y_train, y_test
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the SVM model
        
        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("🚀 Training SVM model...")
        print(f"Training data shape: {X_train.shape}")
        
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        print("✅ SVM model training completed!")
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate the trained model
        
        Args:
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dict containing evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        print("📈 Evaluating SVM model...")
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]  # Probability of spam
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': conf_matrix,
            'classification_report': class_report,
            'predictions': y_pred,
            'probabilities': y_prob,
            'true_labels': y_test
        }
        
        # Print results
        print(f"\n📊 SVM Model Evaluation Results:")
        print(f"{'='*50}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"              Predicted")
        print(f"              Ham  Spam")
        print(f"Actual Ham    {conf_matrix[0,0]:4d}  {conf_matrix[0,1]:4d}")
        print(f"       Spam   {conf_matrix[1,0]:4d}  {conf_matrix[1,1]:4d}")
        print(f"\nClassification Report:")
        print(class_report)
        
        return results
    
    def predict(self, messages: list) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict spam/ham for new messages
        
        Args:
            messages: List of SMS messages
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # Preprocess and vectorize
        processed_messages = [msg.lower() for msg in messages]
        X = self.vectorizer.transform(processed_messages).toarray()
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)[:, 1]
        
        return predictions, probabilities
    
    def save_model(self, model_dir: str = "models") -> None:
        """
        Save the trained model and vectorizer
        
        Args:
            model_dir: Directory to save the model
        """
        if not self.is_fitted:
            raise ValueError("Model not trained. Call train() first.")
        
        # Create models directory
        Path(model_dir).mkdir(exist_ok=True)
        
        # Save model and vectorizer
        model_path = Path(model_dir) / "svm_spam_classifier.joblib"
        vectorizer_path = Path(model_dir) / "tfidf_vectorizer.joblib"
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.vectorizer, vectorizer_path)
        
        print(f"✅ Model saved to {model_path}")
        print(f"✅ Vectorizer saved to {vectorizer_path}")
    
    def load_model(self, model_dir: str = "models") -> None:
        """
        Load a trained model and vectorizer
        
        Args:
            model_dir: Directory containing the saved model
        """
        model_path = Path(model_dir) / "svm_spam_classifier.joblib"
        vectorizer_path = Path(model_dir) / "tfidf_vectorizer.joblib"
        
        if not model_path.exists() or not vectorizer_path.exists():
            raise FileNotFoundError("Model files not found. Train the model first.")
        
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.is_fitted = True
        
        print(f"✅ Model loaded from {model_path}")
        print(f"✅ Vectorizer loaded from {vectorizer_path}")

# Test the SVM classifier
if __name__ == "__main__":
    print("🧪 Testing SVM Spam Classifier...")
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'message': [
            "FREE! Win a £1000 cash prize now!",
            "Hey, how are you doing today?",
            "URGENT! You have won a lottery! Call now!",
            "Can we meet for lunch tomorrow?",
            "Congratulations! You've won £500. Text WIN to claim."
        ],
        'label_binary': [1, 0, 1, 0, 1]  # 1=spam, 0=ham
    })
    
    # Initialize classifier
    classifier = SVMSpamClassifier()
    
    # Prepare data
    X_train, X_test, y_train, y_test = classifier.prepare_data(sample_data, test_size=0.4)
    
    # Train model
    classifier.train(X_train, y_train)
    
    # Evaluate model
    results = classifier.evaluate(X_test, y_test)
    
    # Test prediction
    test_messages = ["Free money now!", "Hello friend"]
    predictions, probabilities = classifier.predict(test_messages)
    
    print(f"\n🔮 Prediction Test:")
    for msg, pred, prob in zip(test_messages, predictions, probabilities):
        label = "SPAM" if pred == 1 else "HAM"
        print(f"'{msg}' -> {label} (confidence: {prob:.3f})")
    
    print(f"\n✅ SVM classifier test completed!")