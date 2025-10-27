"""
Advanced text preprocessing module for Phase 2
Enhanced cleaning, tokenization, and feature extraction
"""

import pandas as pd
import numpy as np
import re
import string
import nltk
from typing import List, Tuple, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split
import joblib
from pathlib import Path

# Download required NLTK data if not already available
def download_nltk_data():
    """Download required NLTK data"""
    required_data = [
        ('tokenizers/punkt', 'punkt'),
        ('tokenizers/punkt_tab', 'punkt_tab'),
        ('corpora/stopwords', 'stopwords'),
        ('taggers/averaged_perceptron_tagger', 'averaged_perceptron_tagger'),
        ('corpora/wordnet', 'wordnet')
    ]
    
    for path, package in required_data:
        try:
            nltk.data.find(path)
        except LookupError:
            print(f"Downloading NLTK {package}...")
            nltk.download(package)

# Initialize NLTK data
download_nltk_data()

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

class AdvancedTextPreprocessor:
    """Advanced text preprocessing for Phase 2 optimization"""
    
    def __init__(self, use_stemming: bool = True, use_lemmatization: bool = False):
        """
        Initialize the advanced text preprocessor
        
        Args:
            use_stemming: Whether to use stemming
            use_lemmatization: Whether to use lemmatization
        """
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer() if use_stemming else None
        self.lemmatizer = WordNetLemmatizer() if use_lemmatization else None
        self.use_stemming = use_stemming
        self.use_lemmatization = use_lemmatization
        
        # Compile regex patterns for efficiency
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        self.phone_pattern = re.compile(r'\b(?:\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})\b|\b\d{5,}\b')
        self.money_pattern = re.compile(r'[£$€¥₹][\d,]+\.?\d*|\b\d+(?:\.\d+)?\s*(?:pounds?|dollars?|euros?|cents?|pence)\b', re.IGNORECASE)
        self.number_pattern = re.compile(r'\b\d+\b')
        
    def advanced_clean_text(self, text: str) -> str:
        """
        Advanced text cleaning with specific patterns for spam detection
        
        Args:
            text: Raw text string
            
        Returns:
            str: Cleaned text
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = self.url_pattern.sub(' URL ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' EMAIL ', text)
        
        # Replace phone numbers with token
        text = self.phone_pattern.sub(' PHONE ', text)
        
        # Replace money amounts with token
        text = self.money_pattern.sub(' MONEY ', text)
        
        # Replace remaining numbers with token
        text = self.number_pattern.sub(' NUMBER ', text)
        
        # Remove extra punctuation but keep some meaningful ones
        # Keep apostrophes for contractions
        text = re.sub(r"[^\w\s']", ' ', text)
        
        # Handle contractions (basic)
        contractions = {
            "won't": "will not",
            "can't": "cannot",
            "n't": " not",
            "'re": " are",
            "'ve": " have",
            "'ll": " will",
            "'d": " would",
            "'m": " am"
        }
        
        for contraction, expansion in contractions.items():
            text = text.replace(contraction, expansion)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def advanced_tokenize(self, text: str) -> List[str]:
        """
        Advanced tokenization with stemming/lemmatization
        
        Args:
            text: Cleaned text string
            
        Returns:
            List[str]: List of processed tokens
        """
        if not text:
            return []
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Filter tokens
        processed_tokens = []
        for token in tokens:
            # Skip if too short or is stop word
            if len(token) < 2 or token in self.stop_words:
                continue
            
            # Skip if all punctuation
            if all(c in string.punctuation for c in token):
                continue
            
            # Apply stemming or lemmatization
            if self.use_stemming and self.stemmer:
                token = self.stemmer.stem(token)
            elif self.use_lemmatization and self.lemmatizer:
                token = self.lemmatizer.lemmatize(token)
            
            processed_tokens.append(token)
        
        return processed_tokens
    
    def preprocess_text(self, text: str) -> str:
        """
        Complete advanced preprocessing pipeline
        
        Args:
            text: Raw text string
            
        Returns:
            str: Preprocessed text ready for vectorization
        """
        # Clean text
        cleaned = self.advanced_clean_text(text)
        
        # Tokenize and process
        tokens = self.advanced_tokenize(cleaned)
        
        # Join tokens back to string
        return ' '.join(tokens)
    
    def analyze_text_features(self, texts: List[str]) -> Dict[str, Any]:
        """
        Analyze text features for optimization insights
        
        Args:
            texts: List of raw texts
            
        Returns:
            Dict containing analysis results
        """
        print("🔍 Analyzing text features...")
        
        # Apply preprocessing
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Calculate statistics
        original_lengths = [len(text) for text in texts]
        processed_lengths = [len(text) for text in processed_texts]
        token_counts = [len(text.split()) for text in processed_texts]
        
        # Count special tokens
        url_count = sum(1 for text in texts if self.url_pattern.search(text))
        email_count = sum(1 for text in texts if self.email_pattern.search(text))
        phone_count = sum(1 for text in texts if self.phone_pattern.search(text))
        money_count = sum(1 for text in texts if self.money_pattern.search(text))
        
        analysis = {
            'total_texts': len(texts),
            'avg_original_length': np.mean(original_lengths),
            'avg_processed_length': np.mean(processed_lengths),
            'avg_token_count': np.mean(token_counts),
            'reduction_ratio': np.mean(processed_lengths) / np.mean(original_lengths),
            'url_count': url_count,
            'email_count': email_count,
            'phone_count': phone_count,
            'money_count': money_count,
            'sample_processed': processed_texts[:3]
        }
        
        return analysis

# Test the advanced preprocessor
if __name__ == "__main__":
    print("🧪 Testing Advanced Text Preprocessor...")
    
    # Sample spam and ham messages
    test_messages = [
        "FREE! Win £1000 cash prize now! Call 08001234567 or visit http://example.com",
        "URGENT! You have won $500! Text WIN to 12345. Contact us at spam@example.com",
        "Hey, how are you doing today? Can we meet for lunch?",
        "Thanks for the message. I'll call you at 555-123-4567 later.",
        "Check out this website: https://legitimate-site.com for more info"
    ]
    
    # Test different configurations
    print("\n📊 Testing with Stemming:")
    preprocessor_stem = AdvancedTextPreprocessor(use_stemming=True, use_lemmatization=False)
    
    for i, msg in enumerate(test_messages[:3], 1):
        processed = preprocessor_stem.preprocess_text(msg)
        print(f"{i}. Original: {msg}")
        print(f"   Processed: {processed}")
        print()
    
    # Analyze features
    analysis = preprocessor_stem.analyze_text_features(test_messages)
    print("📈 Analysis Results:")
    for key, value in analysis.items():
        if key != 'sample_processed':
            print(f"  {key}: {value}")
    
    print("\n✅ Advanced text preprocessor test completed!")