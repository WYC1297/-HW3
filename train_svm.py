"""
Main training script for SVM spam classifier
Combines data loading, preprocessing, training, and evaluation
"""

from src.data_loader import SMSDataLoader
from src.svm_classifier import SVMSpamClassifier

def main():
    """Main training pipeline"""
    
    # Load the real dataset
    print("📥 Loading SMS spam dataset...")
    loader = SMSDataLoader()
    data = loader.load_data()
    cleaned_data = loader.basic_clean()
    
    # Initialize SVM classifier
    print("\n🤖 Initializing SVM classifier...")
    classifier = SVMSpamClassifier(random_state=42)
    
    # Prepare data and train model
    print("\n📊 Preparing data and training model...")
    X_train, X_test, y_train, y_test = classifier.prepare_data(cleaned_data, test_size=0.2)
    
    # Train the model
    classifier.train(X_train, y_train)
    
    # Evaluate the model
    results = classifier.evaluate(X_test, y_test)
    
    # Save the model
    print("\n💾 Saving trained model...")
    classifier.save_model()
    
    # Test with sample messages
    print("\n🔮 Testing with sample messages...")
    test_messages = [
        "FREE! Win £1000 cash prize now! Call 08001234567",
        "Hey, how are you doing today?",
        "URGENT! You have won a lottery! Text WIN to 12345",
        "Can we meet for lunch tomorrow?",
        "Congratulations! Click here to claim your reward!"
    ]
    
    predictions, probabilities = classifier.predict(test_messages)
    
    print(f"\n📋 Prediction Results:")
    print(f"{'='*60}")
    for msg, pred, prob in zip(test_messages, predictions, probabilities):
        label = "SPAM" if pred == 1 else "HAM"
        confidence = prob if pred == 1 else (1 - prob)
        print(f"{label:4s} | {confidence:.3f} | {msg}")
    
    print(f"\n✅ Training pipeline completed successfully!")
    return results

if __name__ == "__main__":
    results = main()