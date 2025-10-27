# SMS Spam Classification System

A comprehensive SMS spam classification system with machine learning models, interactive visualizations, and live inference capabilities.

## 🚀 Live Demo

**🎯 Interactive Dashboard**: [https://sms-spam-classifier.streamlit.app/](https://your-username-sms-spam-classifier.streamlit.app/)

## 📊 Features

### 🤖 Machine Learning Models
- **Baseline SVM**: Basic TF-IDF features
- **Optimized SVM**: Feature selection and parameter tuning
- **Advanced SVM**: SMOTE balancing and grid search optimization

### 📈 Interactive Dashboard
- **Live Inference**: Real-time SMS classification with adjustable threshold
- **Model Analysis**: Performance comparison across all models
- **Visualizations**: Interactive charts, word clouds, and analytics
- **Smart Example Generator**: Generate random spam/ham examples from real data

### 📊 Comprehensive Analytics
- ROC and Precision-Recall curves
- Confusion matrices with detailed metrics
- Threshold sweep analysis
- Token frequency analysis
- Class distribution visualizations

## 🎯 Performance

**Best Model (Advanced SVM)**:
- ROC AUC: **0.9910**
- PR AUC: **0.9415**
- Optimal Threshold: **0.020**
- Dataset: 5,160 messages (12.4% spam, 87.6% ham)

## 🚀 Quick Start

### Online Dashboard
Visit the live dashboard: **[SMS Spam Classifier](https://your-username-sms-spam-classifier.streamlit.app/)**

### Local Installation

1. **Clone the repository**:
```bash
git clone https://github.com/your-username/sms-spam-classifier.git
cd sms-spam-classifier
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the dashboard**:
```bash
streamlit run streamlit_app.py
```

4. **Generate visualizations**:
```bash
python scripts/visualize_spam.py
```

## 📁 Project Structure

```
sms-spam-classifier/
├── streamlit_app.py           # Main Streamlit dashboard
├── scripts/
│   └── visualize_spam.py      # CLI visualization generator
├── app/
│   └── streamlit_app.py       # Enhanced dashboard
├── src/                       # Source modules
├── models/                    # Trained ML models
├── data/                      # Dataset
├── reports/visualizations/    # Generated reports
├── requirements.txt           # Dependencies
└── README.md                  # This file
```

## 🎛️ Dashboard Features

### 🏠 Overview Tab
- Dataset statistics and metrics
- Interactive class distribution charts
- Sample message examples

### 📊 Visualizations Tab
- Token frequency analysis
- Word clouds for spam vs ham
- Interactive Plotly charts

### 🎯 Model Analysis Tab
- Performance metrics comparison
- ROC and PR curves for all models
- Confusion matrices
- Threshold sweep analysis

### 🔮 Live Inference Tab
- **Smart Example Generator**: Choose spam/ham and generate random examples
- **Adjustable Threshold**: Real-time threshold control (0.0-1.0)
- **Multi-Model Support**: Compare predictions across models
- **Confidence Gauge**: Visual spam probability indicator
- **Message Analysis**: Character count, word analysis

### 📋 Reports Tab
- View generated visualization files
- Access analysis reports
- Download comprehensive analytics

## 🎲 Smart Example Generator

### Features:
- **Type Selection**: Choose to generate Spam or Ham examples
- **Random Generation**: Pull real examples from the dataset
- **One-Click Use**: Automatically fill text input with generated examples
- **Color Coding**: Visual distinction between spam (red) and ham (green)
- **Preview**: Display generated examples with smart truncation

## 📈 Model Performance Details

| Model | Accuracy | Precision | Recall | F1-Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| Baseline | 0.9767 | 0.9890 | 0.8601 | 0.9203 | 0.9847 |
| Optimized | 0.9845 | 0.9890 | 0.9070 | 0.9462 | 0.9910 |
| Advanced | 0.9845 | 0.9890 | 0.9070 | 0.9462 | 0.9910 |

## 🔧 Technologies Used

- **Machine Learning**: scikit-learn, imbalanced-learn
- **Web Framework**: Streamlit
- **Visualization**: Plotly, Matplotlib, Seaborn
- **NLP**: NLTK, TF-IDF Vectorization
- **Data Processing**: Pandas, NumPy

## 📊 Dataset

- **Source**: SMS Spam Collection Dataset
- **Total Messages**: 5,160 (after cleaning)
- **Spam Rate**: 12.4%
- **Features**: Text preprocessing, TF-IDF vectorization, feature selection

## 🎨 Visualizations Generated

### Static Visualizations (PNG)
- Class distribution charts
- Token frequency analysis
- Confusion matrices
- ROC & PR curves
- Threshold sweep plots

### Interactive Visualizations (HTML)
- All above charts in interactive Plotly format
- Hover details and zoom capabilities
- Professional styling and animations

## 🔮 Advanced Features

- **Threshold Control**: Live adjustment with immediate feedback
- **Model Comparison**: Side-by-side performance analysis
- **Real-time Classification**: Instant spam/ham prediction
- **Comprehensive Analytics**: 12+ visualization types
- **Professional UI**: Modern design with gradient effects

## 📝 Usage Examples

### Live Classification
1. Navigate to the "🔮 Live Inference" tab
2. Select your preferred model
3. Adjust the classification threshold
4. Enter an SMS message or use the smart generator
5. View real-time prediction with confidence score

### Generate Analytics
```bash
# Generate comprehensive visualization reports
python scripts/visualize_spam.py --output-dir reports/visualizations

# Launch interactive dashboard
streamlit run streamlit_app.py
```

## 🚀 Deployment

This project is deployed on:
- **Streamlit Cloud**: [https://your-username-sms-spam-classifier.streamlit.app/](https://your-username-sms-spam-classifier.streamlit.app/)
- **GitHub**: [https://github.com/your-username/sms-spam-classifier](https://github.com/your-username/sms-spam-classifier)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📞 Contact

- **Project Link**: [https://github.com/your-username/sms-spam-classifier](https://github.com/your-username/sms-spam-classifier)
- **Live Demo**: [https://your-username-sms-spam-classifier.streamlit.app/](https://your-username-sms-spam-classifier.streamlit.app/)

## 🙏 Acknowledgments

- SMS Spam Collection Dataset
- Streamlit for the amazing web framework
- Plotly for interactive visualizations
- scikit-learn for machine learning tools