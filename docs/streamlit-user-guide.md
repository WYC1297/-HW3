# SMS Spam Classifier - Streamlit Web Application

## 🚀 Quick Start

### Launch the Application
```bash
# Activate virtual environment
cd c:\Users\WANG\Desktop\hw3
HW3\Scripts\activate

# Run the Streamlit app
python -m streamlit run streamlit_app.py
```

The application will be available at: `http://localhost:8501`

## 📱 Application Features

### 🏠 Home Page
- **Project Overview**: Comprehensive introduction to the SMS spam classification system
- **Performance Metrics**: Key statistics and model performance summary
- **Quick Test**: Instant message classification with confidence scoring

### 🔍 Classifier Page
- **Real-time Classification**: Enter any text message for instant spam/ham detection
- **Model Selection**: Choose from Phase 1, 2, or 3 optimized models
- **Sample Messages**: Pre-loaded examples of spam and ham messages
- **Detailed Analysis**: Character analysis, word counts, and suspicious pattern detection
- **Prediction History**: Track recent classifications with timestamps

### 📊 Dashboard Page
- **Performance Comparison**: Interactive charts comparing all model phases
- **Metrics Evolution**: Line charts showing improvement across development phases
- **Detailed Metrics Table**: Comprehensive performance statistics
- **Key Insights**: Analysis of optimization achievements and improvements

### 🔍 Data Explorer Page
- **Dataset Overview**: Statistical summary of the training data
- **Class Distribution**: Visual representation of ham vs spam balance
- **Message Analysis**: Length distribution and word count patterns
- **Sample Messages**: Browse actual training examples
- **Word Frequency**: Most common words and spam/ham indicators

### ℹ️ About Page
- **Project Documentation**: Complete development history and technical details
- **Phase Breakdown**: Detailed explanation of each optimization phase
- **Technical Stack**: Tools and technologies used
- **Performance Timeline**: Achievement progression across phases

## 🎯 Key Features

### Real-time Classification
- Instant spam detection with confidence scores
- Support for multiple model versions
- Detailed text analysis and pattern detection
- Visual confidence gauge and metrics

### Interactive Visualizations
- Dynamic charts and graphs using Plotly
- Performance dashboards with drill-down capabilities
- Data exploration tools
- Responsive design for all screen sizes

### Model Insights
- Compare different optimization phases
- Understand model decision-making
- Track prediction history
- Analyze text patterns and features

## 📊 Model Performance

| Phase | Description | Accuracy | Precision | Recall | F1-Score |
|-------|-------------|----------|-----------|--------|----------|
| Phase 1 | Baseline SVM | 98.16% | 100.00% | 85.16% | 91.98% |
| Phase 2 | Optimized | 98.45% | 99.12% | 88.28% | 93.39% |
| Phase 3 | Advanced | 98.84% | 93.08% | 94.53% | 93.80% |

**🎯 Phase 3 Achievement**: Successfully balanced precision and recall, both exceeding 93% target!

## 🛠️ Technical Implementation

### Architecture
- **Frontend**: Streamlit with custom CSS styling
- **Backend**: scikit-learn models with joblib persistence
- **Visualization**: Plotly, matplotlib, seaborn
- **Data Processing**: pandas, numpy, NLTK

### Model Pipeline
1. **Text Preprocessing**: Advanced cleaning and normalization
2. **Vectorization**: Optimized TF-IDF with 3000 features
3. **Classification**: SVM with RBF kernel and balanced weights
4. **Post-processing**: Threshold optimization and probability scoring

### Web Interface
- **Responsive Design**: Works on desktop and mobile devices
- **Interactive Navigation**: Sidebar menu with multiple pages
- **Real-time Updates**: Instant predictions and visualizations
- **Professional Styling**: Custom CSS for polished appearance

## 📝 Usage Examples

### Basic Classification
1. Navigate to the "Classifier" page
2. Select your preferred model (Phase 3 recommended)
3. Enter a text message or choose a sample
4. Click "Classify Message" for instant results

### Model Comparison
1. Go to the "Dashboard" page
2. View interactive performance charts
3. Compare metrics across different phases
4. Analyze improvement trends

### Data Exploration
1. Visit the "Data Explorer" page
2. Examine dataset statistics and distributions
3. Browse sample messages by type
4. Analyze word frequency patterns

## 🔧 Advanced Features

### Prediction Analysis
- **Confidence Scoring**: Probability-based confidence metrics
- **Text Statistics**: Character and word analysis
- **Pattern Detection**: Identification of spam indicators
- **Historical Tracking**: Maintain prediction history

### Visualization Options
- **Interactive Charts**: Hover effects and drill-down capabilities
- **Multiple Chart Types**: Bar, line, pie, histogram, box plots
- **Custom Styling**: Consistent color schemes and branding
- **Export Options**: Download charts and data

## 🚨 Troubleshooting

### Common Issues

**1. Model Loading Errors**
- Ensure all model files are present in `models/` directory
- Check file permissions and paths
- Verify model compatibility

**2. Import Errors**
- Activate the HW3 virtual environment
- Install missing packages: `pip install -r requirements.txt`
- Check Python version compatibility

**3. Performance Issues**
- Large text inputs may slow processing
- Consider caching for repeated predictions
- Monitor memory usage with large datasets

### Error Messages
- **"Model not available"**: Check model file paths and loading
- **"Data not available"**: Verify dataset location and format
- **"Prediction error"**: Review text input and preprocessing

## 📈 Future Enhancements

### Planned Features
- **Batch Processing**: Upload and classify multiple messages
- **API Integration**: RESTful API for external applications
- **Advanced Analytics**: Deeper text analysis and insights
- **User Management**: Multi-user support and preferences
- **Export Functions**: Download predictions and reports

### Technical Improvements
- **Model Versioning**: Support for multiple model versions
- **A/B Testing**: Compare different algorithms
- **Real-time Retraining**: Update models with new data
- **Performance Monitoring**: Track prediction accuracy over time

## 📧 Support

For technical support or feature requests:
1. Check the troubleshooting section
2. Review the application logs
3. Refer to the project documentation
4. Contact the development team

---

**Built with ❤️ using Python, Streamlit, and scikit-learn**

*Version 1.0 - Phase 5 Complete*