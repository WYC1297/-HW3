"""
Streamlit App Entry Point with Path Configuration
"""

import sys
from pathlib import Path

# Add src directory to Python path
current_dir = Path(__file__).parent
src_path = current_dir / 'src'
sys.path.insert(0, str(src_path))

# Import and run the main app
from streamlit_app import EnhancedSpamDashboard

if __name__ == "__main__":
    dashboard = EnhancedSpamDashboard()
    dashboard.run_dashboard()