# Enhanced SMS Spam Classification System

## 🎯 Overview
This project provides a comprehensive SMS spam classification system with advanced visualizations and interactive dashboards.

## 📁 Project Structure

```
├── scripts/
│   └── visualize_spam.py          # CLI visualization generator
├── app/
│   └── streamlit_app.py           # Enhanced Streamlit dashboard
├── reports/
│   └── visualizations/            # Generated analysis reports
├── models/                        # Trained ML models
├── src/                          # Source modules
└── streamlit_app.py              # Main dashboard (copy of app/streamlit_app.py)
```

## 🚀 Quick Start

### 1. Generate Comprehensive Visualizations
```bash
# Activate environment
.\HW3\Scripts\activate

# Generate all visualizations and reports
python scripts/visualize_spam.py

# Optional: specify custom output directory
python scripts/visualize_spam.py --output-dir custom/path
```

### 2. Launch Interactive Dashboard
```bash
# Start the enhanced Streamlit dashboard
streamlit run streamlit_app.py
```

## 📊 Generated Visualizations

The CLI script `scripts/visualize_spam.py` generates:

### Static Visualizations (PNG)
- **Class Distribution**: Bar chart and pie chart of spam vs ham messages
- **Token Frequency Analysis**: Top tokens comparison between spam and ham
- **Confusion Matrix**: Model performance on test data
- **ROC & PR Curves**: Receiver Operating Characteristic and Precision-Recall curves
- **Threshold Sweep**: Performance metrics across different classification thresholds

### Interactive Visualizations (HTML)
- All above charts in interactive Plotly format with hover details and zoom capabilities

### Data Files
- **Threshold Analysis CSV**: Detailed performance metrics across thresholds
- **Analysis Summary**: Comprehensive markdown report

## 🎛️ Interactive Dashboard Features

The enhanced Streamlit dashboard provides:

### 📊 Five Main Tabs

1. **🏠 Overview**
   - Dataset statistics and sample messages
   - Interactive class distribution charts

2. **📊 Visualizations** 
   - Token frequency analysis
   - Word clouds for spam vs ham
   - Interactive charts with zoom and hover

3. **🎯 Model Analysis**
   - Performance metrics comparison
   - ROC and PR curves for all models
   - Confusion matrices
   - Threshold sweep analysis

4. **🔮 Live Inference**
   - Real-time SMS classification
   - **Adjustable threshold slider** (0.0 - 1.0)
   - **Probability gauge** showing spam likelihood
   - Model selection (Baseline, Optimized, Advanced)
   - Example messages to test
   - Message analysis (character count, words, etc.)

5. **📋 Reports**
   - View generated visualization files
   - Access analysis reports
   - Generate new reports

### 🎚️ Advanced Features

- **Threshold Control**: Adjust classification threshold with live updates
- **Multi-Model Support**: Compare Baseline, Optimized, and Advanced models
- **Confidence Visualization**: Interactive gauge showing prediction confidence
- **Professional Styling**: Gradient headers, color-coded results
- **Error Handling**: Robust model loading with fallbacks

## 🔧 CLI Script Usage

```bash
# Basic usage
python scripts/visualize_spam.py

# With custom output directory
python scripts/visualize_spam.py --output-dir my_reports

# Help
python scripts/visualize_spam.py --help
```

### CLI Features:
- **Automatic Model Loading**: Finds best available model (Advanced → Optimized → Baseline)
- **Comprehensive Analysis**: Generates 12+ visualization files
- **Performance Metrics**: ROC AUC, PR AUC, optimal thresholds
- **Interactive & Static**: Both PNG and HTML versions
- **Summary Report**: Detailed markdown analysis

## 📈 Model Performance

Current best model (Advanced Model):
- **ROC AUC**: 0.9910
- **PR AUC**: 0.9415
- **Optimal Threshold**: 0.020
- **Dataset**: 5,160 messages (12.4% spam, 87.6% ham)

## 🎨 Visualization Examples

### Class Distribution
- Balanced view of spam vs ham distribution
- Percentage breakdowns and counts

### Token Frequency Analysis
- Top 15-20 most discriminative tokens
- Side-by-side comparison of spam vs ham patterns

### Performance Curves
- ROC curves comparing all models
- Precision-Recall curves with baseline references
- Threshold sweep showing optimal operating points

### Confusion Matrices
- Heatmaps with counts and percentages
- Interactive hover details

## 🔮 Live Inference Features

### Text Input
- Real-time SMS message classification
- Support for any message length

### Threshold Control
- Slider from 0.0 to 1.0
- Live updates as threshold changes
- Visual threshold line in gauge

### Model Selection
- Choose between available models
- Compare predictions across models

### Confidence Visualization
- Interactive gauge showing spam probability
- Color-coded results (red for spam, green for ham)
- Threshold indicator line

### Message Analysis
- Character count, word count
- Special character analysis (exclamations, numbers)

## 📋 Output Files

All files are saved to `reports/visualizations/`:

- `class_distribution.png` / `.html`
- `token_frequency_analysis.png` / `.html`
- `confusion_matrix_[model].png` / `.html`
- `roc_pr_curves_[model].png` / `.html`
- `threshold_sweep_[model].png` / `.html` / `.csv`
- `analysis_summary.md`

## 🎯 Key Improvements

1. **Professional CLI Tool**: Comprehensive visualization generation
2. **Enhanced Dashboard**: 5-tab interface with advanced features
3. **Threshold Control**: Adjustable classification threshold
4. **Live Inference**: Real-time prediction with confidence gauges
5. **Multi-Model Support**: Compare different model phases
6. **Interactive Charts**: Plotly visualizations with hover and zoom
7. **Comprehensive Reports**: Detailed analysis and summaries
8. **Professional Styling**: Modern UI with gradient effects

## 🚀 Getting Started

1. Ensure all models are trained and saved in `models/` directory
2. Run `python scripts/visualize_spam.py` to generate comprehensive reports
3. Launch `streamlit run streamlit_app.py` for interactive exploration
4. Use the **Live Inference** tab to test custom messages with threshold control

The system now provides both automated analysis generation and interactive exploration capabilities!