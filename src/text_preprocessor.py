"""
Text preprocessing module for SMS spam classification
Handles tokenization, cleaning, and feature extraction
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from typing import List, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    print("Downloading NLTK stopwords...")
    nltk.download('stopwords')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading NLTK POS tagger...")
    nltk.download('averaged_perceptron_tagger')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

class SMSTextPreprocessor:
    """Class for preprocessing SMS text data"""
    
    def __init__(self):
        """Initialize the text preprocessor"""
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.vectorizer = None
        
    def clean_text(self, text: str) -> str:
        """
        Clean and normalize text
        
        Args:
            text: Raw text string
            
        Returns:
            str: Cleaned text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Remove phone numbers (basic pattern)
        text = re.sub(r'\b\d{3,}\b', '', text)
        
        # Remove punctuation except apostrophes
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def tokenize_and_stem(self, text: str) -> List[str]:
        """
        Tokenize text and apply stemming
        
        Args:
            text: Cleaned text string
            
        Returns:
            List[str]: List of stemmed tokens
        """
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and stem
        tokens = [
            self.stemmer.stem(token) 
            for token in tokens 
            if token not in self.stop_words and len(token) > 2
        ]
        
        return tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete text preprocessing pipeline
        
        Args:
            text: Raw text string
            
        Returns:
            str: Preprocessed text ready for vectorization
        """
        # Clean text
        cleaned = self.clean_text(text)
        
        # Tokenize and stem
        tokens = self.tokenize_and_stem(cleaned)
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def fit_vectorizer(self, texts: List[str], method: str = 'tfidf', max_features: int = 5000) -> None:
        """
        Fit vectorizer on training texts
        
        Args:
            texts: List of preprocessed texts
            method: 'tfidf' or 'count'
            max_features: Maximum number of features
        """
        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),  # Include bigrams
                min_df=2,  # Ignore terms that appear in less than 2 documents
                max_df=0.95  # Ignore terms that appear in more than 95% of documents
            )
        else:
            self.vectorizer = CountVectorizer(
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
        
        print(f"Fitting {method} vectorizer with max_features={max_features}...")
        self.vectorizer.fit(texts)
        print(f"✅ Vectorizer fitted. Vocabulary size: {len(self.vectorizer.vocabulary_)}")
    
    def transform_texts(self, texts: List[str]) -> np.ndarray:
        """
        Transform texts to feature vectors
        
        Args:
            texts: List of preprocessed texts
            
        Returns:
            np.ndarray: Feature matrix
        """
        if self.vectorizer is None:
            raise ValueError("Vectorizer not fitted. Call fit_vectorizer first.")
        
        return self.vectorizer.transform(texts).toarray()
    
    def preprocess_dataset(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Complete preprocessing pipeline for the dataset
        
        Args:
            df: DataFrame with 'message' and 'label_binary' columns
            
        Returns:
            Tuple of (features, labels, preprocessed_texts)
        """
        print("🔄 Starting text preprocessing pipeline...")
        
        # Preprocess all texts
        print("1. Cleaning and preprocessing texts...")
        preprocessed_texts = [self.preprocess_text(text) for text in df['message']]
        
        # Fit vectorizer on all texts
        print("2. Fitting vectorizer...")
        self.fit_vectorizer(preprocessed_texts)
        
        # Transform texts to features
        print("3. Transforming texts to feature vectors...")
        features = self.transform_texts(preprocessed_texts)
        labels = df['label_binary'].values
        
        print(f"✅ Preprocessing complete!")
        print(f"   - Feature matrix shape: {features.shape}")
        print(f"   - Labels shape: {labels.shape}")
        print(f"   - Vocabulary size: {len(self.vectorizer.vocabulary_)}")
        
        return features, labels, preprocessed_texts

# Test the preprocessor
if __name__ == "__main__":
    # Sample text for testing
    sample_texts = [
        "FREE! Win a £1000 cash prize or a prize worth £5000. To enter send WIN to 85233 NOW! Only 150p/min. Over 18s only.",
        "Hey there! How are you doing today? Hope you're having a great day!",
        "URGENT! You have won a lottery prize. Call 123-456-7890 immediately to claim your reward."
    ]
    
    # Initialize preprocessor
    preprocessor = SMSTextPreprocessor()
    
    # Test preprocessing
    print("🧪 Testing text preprocessing:")
    for i, text in enumerate(sample_texts, 1):
        processed = preprocessor.preprocess_text(text)
        print(f"\n{i}. Original: {text}")
        print(f"   Processed: {processed}")
    
    # Test vectorization
    print(f"\n🧪 Testing vectorization:")
    preprocessor.fit_vectorizer([preprocessor.preprocess_text(text) for text in sample_texts])
    features = preprocessor.transform_texts([preprocessor.preprocess_text(text) for text in sample_texts])
    print(f"Feature matrix shape: {features.shape}")
    print(f"Sample features (first 10): {features[0][:10]}")