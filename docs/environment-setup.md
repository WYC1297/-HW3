# Environment Setup Documentation

## HW3 Spam Classification Project - Environment Setup

### Overview
This document describes the setup process for the HW3 virtual environment used in the spam email classification project.

### Prerequisites
- Python 3.7+ installed on the system
- Access to command line/PowerShell

### Setup Steps

#### 1. Create Virtual Environment
```bash
# Navigate to project directory
cd c:\Users\WANG\Desktop\hw3

# Create virtual environment named "HW3"
python -m venv HW3
```

#### 2. Activate Virtual Environment
```bash
# On Windows PowerShell
.\HW3\Scripts\Activate.ps1

# On Windows Command Prompt
HW3\Scripts\activate.bat

# On Linux/Mac
source HW3/bin/activate
```

#### 3. Upgrade Base Packages
```bash
python -m pip install --upgrade pip setuptools wheel
```

#### 4. Install Project Dependencies
```bash
pip install -r requirements.txt
```

### Verification
After setup, verify the environment:

```bash
# Check Python version
python --version

# Check installed packages
pip list

# Verify environment is active (should show (HW3) prefix)
```

### Environment Details
- **Environment Name**: HW3
- **Python Version**: 3.13.7
- **Environment Type**: Virtual Environment (venv)
- **Location**: `c:\Users\WANG\Desktop\hw3\HW3\`

### Key Dependencies Installed
- scikit-learn (>=1.3.0) - Machine learning framework
- numpy (>=1.24.0) - Numerical computing
- pandas (>=2.0.0) - Data manipulation
- requests (>=2.31.0) - HTTP library for data download
- nltk (>=3.8.0) - Natural language processing
- pytest (>=7.4.0) - Testing framework

### Deactivation
To deactivate the environment when done:
```bash
deactivate
```

### Notes
- Always activate the HW3 environment before working on the project
- The environment is isolated and won't affect system Python packages
- Additional packages for future phases (Streamlit, visualization) are commented in requirements.txt