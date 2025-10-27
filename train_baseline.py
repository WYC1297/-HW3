"""
Complete SVM baseline training script
Trains on real SMS data and saves the model
"""

import pandas as pd
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
from pathlib import Path
import numpy as np

def train_svm_baseline():
    """Train SVM baseline model on SMS spam data"""
    
    print("🚀 Starting SVM Baseline Training")
    print("="*50)
    
    # Load data
    print("📥 Loading data...")
    data_path = "data/sms_spam_no_header.csv"
    df = pd.read_csv(data_path, header=None, names=['label', 'message'])
    
    # Basic cleaning
    print("🧹 Basic data cleaning...")
    df = df.drop_duplicates()
    df['label_binary'] = df['label'].map({'ham': 0, 'spam': 1})
    
    print(f"Dataset shape: {df.shape}")
    print(f"Label distribution: {df['label_binary'].value_counts().to_dict()}")
    
    # Simple preprocessing
    print("🔄 Preprocessing texts...")
    messages = df['message'].str.lower().fillna('')
    labels = df['label_binary'].values
    
    # Split data
    print("📊 Splitting data...")
    X_train_text, X_test_text, y_train, y_test = train_test_split(
        messages, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training set size: {len(X_train_text)}")
    print(f"Test set size: {len(X_test_text)}")
    
    # Vectorize
    print("🔤 Vectorizing texts...")
    vectorizer = TfidfVectorizer(
        max_features=3000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    X_train = vectorizer.fit_transform(X_train_text).toarray()
    X_test = vectorizer.transform(X_test_text).toarray()
    
    print(f"Feature matrix shape: {X_train.shape}")
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # Train SVM
    print("🤖 Training SVM model...")
    svm = SVC(kernel='rbf', random_state=42, probability=True)
    svm.fit(X_train, y_train)
    
    # Evaluate
    print("📈 Evaluating model...")
    y_pred = svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 EVALUATION RESULTS")
    print(f"{'='*30}")
    print(f"Accuracy: {accuracy:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"              Predicted")
    print(f"              Ham  Spam")
    print(f"Actual Ham    {cm[0,0]:4d}  {cm[0,1]:4d}")
    print(f"       Spam   {cm[1,0]:4d}  {cm[1,1]:4d}")
    
    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))
    
    # Save model
    print("💾 Saving model...")
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)
    
    joblib.dump(svm, model_dir / "svm_spam_classifier.joblib")
    joblib.dump(vectorizer, model_dir / "tfidf_vectorizer.joblib")
    
    print(f"✅ Model saved to {model_dir}")
    
    # Test predictions
    print("\n🔮 Testing predictions...")
    test_messages = [
        "FREE! Win £1000 now! Call 08001234567",
        "Hey, how are you doing today?",
        "URGENT! You won a prize! Text WIN to 12345",
        "Can we meet for lunch tomorrow?"
    ]
    
    test_processed = [msg.lower() for msg in test_messages]
    test_vectors = vectorizer.transform(test_processed).toarray()
    predictions = svm.predict(test_vectors)
    probabilities = svm.predict_proba(test_vectors)[:, 1]
    
    print(f"\n📋 Prediction Results:")
    print(f"{'='*60}")
    for msg, pred, prob in zip(test_messages, predictions, probabilities):
        label = "SPAM" if pred == 1 else "HAM"
        confidence = prob if pred == 1 else (1 - prob)
        print(f"{label:4s} | {confidence:.3f} | {msg}")
    
    print(f"\n✅ SVM Baseline Training Completed!")
    
    return {
        'accuracy': accuracy,
        'confusion_matrix': cm,
        'model': svm,
        'vectorizer': vectorizer
    }

if __name__ == "__main__":
    results = train_svm_baseline()