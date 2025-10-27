"""
Data download script for SMS spam dataset
Downloads the dataset from the specified GitHub repository
"""

import requests
import os
from pathlib import Path

def download_dataset():
    """Download the SMS spam dataset from GitHub repository"""
    
    # Dataset URL
    url = "https://raw.githubusercontent.com/PacktPublishing/Hands-On-Artificial-Intelligence-for-Cybersecurity/refs/heads/master/Chapter03/datasets/sms_spam_no_header.csv"
    
    # Create data directory if it doesn't exist
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    # Download file
    filename = "sms_spam_no_header.csv"
    filepath = data_dir / filename
    
    print(f"Downloading dataset from: {url}")
    print(f"Saving to: {filepath}")
    
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for bad status codes
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        print(f"✅ Dataset downloaded successfully!")
        print(f"📁 File saved as: {filepath}")
        print(f"📊 File size: {os.path.getsize(filepath)} bytes")
        
        return filepath
        
    except requests.exceptions.RequestException as e:
        print(f"❌ Error downloading dataset: {e}")
        return None

if __name__ == "__main__":
    download_dataset()