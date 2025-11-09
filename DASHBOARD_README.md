# 🚛 Supply Chain Risk Intelligence Dashboard

A comprehensive web-based dashboard for supply chain risk analysis and resilience modeling using machine learning and graph neural networks.

## 🌟 Features

### 📊 **Multi-Page Dashboard**
- **Dashboard**: Overview with key metrics and insights
- **Data Overview**: Detailed analysis of risk and resilience datasets
- **Model Performance**: Comprehensive model evaluation and comparison
- **Model Inference**: Real-time predictions with interactive input forms
- **Feature Analysis**: Feature importance, correlations, and distributions
- **Graph Analysis**: GNN embeddings visualization and network insights
- **Reports**: Detailed statistical and quality reports

### 🤖 **Machine Learning Models**
- **CatBoost**: Classification and regression models
- **LightGBM**: High-performance gradient boosting
- **Graph Neural Networks**: Network analysis with PyTorch Geometric

### 📈 **Visualizations**
- Interactive Plotly charts and graphs
- Feature importance plots
- Correlation heatmaps
- Distribution analysis
- Confusion matrices
- Regression performance plots
- t-SNE and PCA embeddings

### 🔍 **Data Analysis**
- VIF (Variance Inflation Factor) analysis
- Statistical summaries
- Data quality assessment
- Feature correlation analysis

## 🚀 Quick Start

### Method 1: Using the Launcher Script (Recommended)
```bash
python run_dashboard.py
```

### Method 2: Manual Installation
```bash
# Install required packages
pip install streamlit==1.25.0 plotly==5.15.0 pillow==10.0.0

# Run the dashboard
streamlit run streamlit_app.py
```

### Method 3: Windows Batch File
```bash
run_dashboard.bat
```

## 📋 Prerequisites

1. **Run the Main Pipeline First**
   ```bash
   python main.py
   ```
   This generates all the necessary data, models, and visualizations.

2. **Python Requirements**
   - Python 3.8+
   - All packages from `requirements_txt.txt`

## 📁 Project Structure

```
├── streamlit_app.py          # Main dashboard application
├── run_dashboard.py          # Dashboard launcher script
├── run_dashboard.bat         # Windows batch launcher
├── main.py                   # Main ML pipeline
├── requirements_txt.txt      # Python dependencies
├── data/
│   ├── cleaned/              # Processed datasets
│   └── raw/                  # Original datasets
├── reports/                  # CSV reports and analysis
├── output/                   # Generated plots and visualizations
└── catboost_info/           # Model training logs
```

## 🎯 Dashboard Pages

### 🏠 Dashboard
- Key performance metrics
- Data distribution overview
- Quick insights and recommendations
- Real-time statistics

### 📊 Data Overview
- **Risk Data**: Environmental, operational, and supply chain factors
- **Resilience Data**: Supply chain resilience metrics and scores
- **Data Quality**: VIF analysis and data validation

### 🤖 Model Performance
- Model comparison summary
- Classification results with confusion matrices
- Regression performance with prediction plots
- Feature importance rankings

### 🔍 Model Inference
- Interactive prediction interface
- Real-time risk and resilience scoring
- Automated recommendations
- Input validation and preprocessing

### 📈 Feature Analysis
- Feature importance across all models
- Correlation analysis between variables
- Feature distribution visualization
- Statistical significance testing

### 🌐 Graph Analysis
- GNN embedding visualizations (t-SNE, PCA)
- Network statistics and insights
- Relationship modeling results
- Graph structure analysis

### 📋 Reports
- Comprehensive statistical summaries
- Model performance reports
- Data quality assessments
- Downloadable CSV reports

## 🎨 UI Features

### 🌈 **Attractive Design**
- Modern gradient-based styling
- Responsive layout for all screen sizes
- Color-coded risk levels and alerts
- Professional metric cards

### 📱 **Interactive Elements**
- Real-time input forms
- Dynamic visualizations
- Downloadable reports
- Tabbed navigation

### 🚨 **Smart Alerts**
- Risk level indicators
- Automated recommendations
- Data quality warnings
- Model performance insights

## 🔧 Customization

### Adding New Models
1. Train your model in `main.py`
2. Save results in the `reports/` directory
3. Add visualization logic to `streamlit_app.py`

### Custom Visualizations
- Modify the plotting functions in `streamlit_app.py`
- Add new chart types using Plotly
- Include custom CSS styling

### Additional Features
- Extend the prediction interface
- Add new data sources
- Implement custom metrics

## 📊 Data Sources

The dashboard works with:
- **Risk Dataset**: 3,000 records with 23 features
- **Resilience Dataset**: 1,000 records with 27 features
- **Graph Data**: Heterogeneous network with suppliers, buyers, and products

## 🛠️ Technical Stack

- **Frontend**: Streamlit with custom CSS
- **Visualization**: Plotly, Matplotlib, Seaborn
- **ML Framework**: CatBoost, LightGBM, PyTorch Geometric
- **Data Processing**: Pandas, NumPy
- **Statistics**: SciPy, Statsmodels

## 🔍 Troubleshooting

### Common Issues

1. **"Data not available" messages**
   - Run `python main.py` first to generate all data and models

2. **Import errors**
   - Install missing packages: `pip install streamlit plotly pillow`

3. **Port conflicts**
   - Streamlit runs on port 8501 by default
   - Change port: `streamlit run streamlit_app.py --server.port 8502`

4. **Memory issues**
   - Large datasets may require more RAM
   - Consider reducing data size or using sampling

### Performance Tips

- Run the main pipeline periodically to update models
- Cache large datasets using Streamlit's caching
- Use the launcher script for automated setup

## 📈 Future Enhancements

- [ ] Real-time data streaming
- [ ] Advanced alerting system
- [ ] Model retraining interface
- [ ] API integration
- [ ] Export to PDF/PowerPoint
- [ ] Multi-user authentication
- [ ] Database connectivity
- [ ] Advanced filtering options

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Support

For questions or issues:
- Create an issue in the repository
- Check the troubleshooting section
- Review the code documentation

---

**Built with ❤️ using Streamlit, CatBoost, LightGBM, and PyTorch Geometric**