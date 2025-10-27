"""
Data loading and preprocessing module for SMS spam classification
Handles loading, basic preprocessing, and data exploration
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import re
import string

class SMSDataLoader:
    """Class for loading and basic preprocessing of SMS spam dataset"""
    
    def __init__(self, data_path: str = "data/sms_spam_no_header.csv"):
        """
        Initialize the data loader
        
        Args:
            data_path: Path to the CSV file containing SMS data
        """
        self.data_path = Path(data_path)
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load the SMS dataset from CSV file
        
        Returns:
            pandas.DataFrame: Loaded dataset with proper column names
        """
        try:
            # Load the CSV file - comma separated with quotes, no header
            self.df = pd.read_csv(self.data_path, header=None, names=['label', 'message'])
            
            print(f"✅ Dataset loaded successfully!")
            print(f"📊 Shape: {self.df.shape}")
            print(f"📋 Columns: {list(self.df.columns)}")
            
            return self.df
            
        except FileNotFoundError:
            print(f"❌ Error: File {self.data_path} not found!")
            return None
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return None
    
    def explore_data(self) -> None:
        """Explore the loaded dataset and print basic statistics"""
        if self.df is None:
            print("❌ No data loaded. Please call load_data() first.")
            return
        
        print("\n" + "="*50)
        print("📊 DATA EXPLORATION")
        print("="*50)
        
        # Basic info
        print(f"Dataset shape: {self.df.shape}")
        print(f"Missing values: {self.df.isnull().sum().sum()}")
        
        # Label distribution
        print(f"\n📈 Label distribution:")
        label_counts = self.df['label'].value_counts()
        print(label_counts)
        print(f"Percentage distribution:")
        print((label_counts / len(self.df) * 100).round(2))
        
        # Message length statistics
        self.df['message_length'] = self.df['message'].str.len()
        print(f"\n📏 Message length statistics:")
        print(self.df['message_length'].describe())
        
        # Sample messages
        print(f"\n📝 Sample messages:")
        for label in self.df['label'].unique():
            print(f"\n{label.upper()} examples:")
            samples = self.df[self.df['label'] == label]['message'].head(2)
            for i, msg in enumerate(samples, 1):
                print(f"  {i}. {msg[:100]}{'...' if len(msg) > 100 else ''}")
    
    def basic_clean(self) -> pd.DataFrame:
        """
        Perform basic data cleaning
        
        Returns:
            pandas.DataFrame: Cleaned dataset
        """
        if self.df is None:
            print("❌ No data loaded. Please call load_data() first.")
            return None
        
        print("\n🧹 Performing basic data cleaning...")
        
        # Remove duplicates
        initial_size = len(self.df)
        self.df = self.df.drop_duplicates()
        duplicates_removed = initial_size - len(self.df)
        print(f"Removed {duplicates_removed} duplicate rows")
        
        # Remove empty messages
        empty_messages = self.df['message'].isna().sum()
        self.df = self.df.dropna(subset=['message'])
        print(f"Removed {empty_messages} empty messages")
        
        # Strip whitespace
        self.df['message'] = self.df['message'].str.strip()
        
        # Convert labels to binary (ham=0, spam=1)
        label_mapping = {'ham': 0, 'spam': 1}
        self.df['label_binary'] = self.df['label'].map(label_mapping)
        
        print(f"✅ Cleaning complete. Final dataset shape: {self.df.shape}")
        
        return self.df
    
    def get_data(self) -> Optional[pd.DataFrame]:
        """Return the loaded and cleaned dataset"""
        return self.df

# Test the data loader
if __name__ == "__main__":
    # Initialize data loader
    loader = SMSDataLoader()
    
    # Load and explore data
    data = loader.load_data()
    if data is not None:
        loader.explore_data()
        cleaned_data = loader.basic_clean()
        print(f"\n🎯 Ready for preprocessing: {len(cleaned_data)} messages")