# MOF Property Prediction Using Machine Learning

[![Python 3.8+](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Machine learning pipeline for predicting Metal-Organic Framework (MOF) formation energy from synthesis parameters and surface chemistry. Achieves **82.28% prediction accuracy** with interpretable insights into structure-property relationships.

## 🎯 Key Features

- **High Accuracy**: 82.28% R² score using Ridge Regression (±1.31 kJ/mol error)
- **Comprehensive Pipeline**: From raw data to production-ready models
- **Multiple Algorithms**: Comparison of Ridge Regression, Gradient Boosting, Random Forest, and SVM
- **Feature Importance**: Quantitative analysis of factors controlling MOF properties
- **Well-Documented**: Complete documentation, tutorials, and reproducible notebooks
- **Research-Ready**: Publication-quality analysis and methodology

## 📊 Research Findings

### Surface Chemistry Controls MOF Formation Energy

```
COOH-like (acidic):  -49.65 kJ/mol  ← MORE STABLE (kinetic phase)
Neutral:             -46.69 kJ/mol
NH2-like (basic):    -43.87 kJ/mol  ← LESS STABLE (thermodynamic phase)

Difference: 5.78 kJ/mol between extremes
```

**Conclusion**: Surface chemistry is the PRIMARY control variable for MOF formation energy and crystal phase selectivity.

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/yourusername/mof_ml_project.git
cd mof_ml_project

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```python
import joblib
import pandas as pd

# Load the best-performing model
model = joblib.load('models/ridge_model.pkl')

# Load test data
X_test = pd.read_csv('data/processed/X_test.csv')

# Make predictions
predictions = model.predict(X_test)

# Display results
print(f"Predictions: {predictions}")
```

### Run Notebooks

```bash
# Start Jupyter
jupyter notebook

# Navigate to notebooks/ and open:
# 01_data_loading_exploration.ipynb - Data overview
# 02_feature_engineering.ipynb - Feature creation
# 03_model_training.ipynb - Model development
# 04_model_evaluation.ipynb - Performance analysis
# 05_analysis_insights.ipynb - Key findings
```

## 📁 Project Structure

```
mof_ml_project/
├── data/
│   ├── raw/                          # Original datasets
│   └── processed/                    # Engineered features and splits
├── notebooks/                        # Jupyter notebooks (step-by-step)
├── src/                              # Source code modules
│   ├── data_loader.py               # Data loading utilities
│   ├── feature_engineer.py          # Feature engineering functions
│   ├── models.py                    # Model definitions and training
│   ├── evaluate.py                  # Evaluation metrics
│   └── utils.py                     # Helper functions
├── models/                           # Trained model files
│   ├── ridge_model.pkl              # Best performer (R²=0.8228)
│   ├── gradient_boosting_model.pkl
│   ├── random_forest_model.pkl
│   └── svm_model.pkl
├── results/                          # Analysis outputs
│   ├── model_comparison.csv
│   ├── feature_importance.csv
│   └── figures/
├── tests/                            # Unit tests
├── docs/                             # Documentation
├── requirements.txt                  # Dependencies
├── setup.py                          # Package configuration
└── README.md                         # This file
```

## 📈 Model Performance

| Model | Test R² | Test RMSE | Test MAE | Training Time |
|-------|---------|-----------|----------|---------------|
| **Ridge Regression** ⭐ | **0.8228** | **1.6343** | **1.3140** | <0.1s |
| Gradient Boosting | 0.7698 | 1.8628 | 1.5072 | 0.47s |
| Random Forest | 0.7533 | 1.9284 | 1.5915 | 0.5s |
| Support Vector Machine | 0.7135 | 2.0780 | 1.6992 | 0.35s |

## 🔍 Top Features Influencing MOF Properties

1. **Surface chemistry (COOH-like)** - 23.6% importance
2. **Surface chemistry (NH2-like)** - 15.5% importance
3. **Surface area (m²/g)** - 10.8% importance
4. **Metal: Dysprosium** - 5.8% importance
5. **Topology: pcu** - 5.4% importance

## 📚 Documentation

- **[Installation Guide](docs/installation.md)** - Setup and requirements
- **[Usage Guide](docs/usage.md)** - How to use the models and code
- **[API Reference](docs/api_reference.md)** - Function and class documentation
- **[Contributing](CONTRIBUTING.md)** - How to contribute to the project
- **[FAQ](docs/faq.md)** - Frequently asked questions

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_models.py

# Run with coverage
pytest --cov=src tests/
```

## 📊 Dataset

- **Total Samples**: 1,500 MOF configurations
- **Features Engineered**: 40 (from raw synthesis parameters)
- **Target Variable**: Formation Energy (kJ/mol)
- **Data Split**: 70% train, 15% validation, 15% test
- **Metal Types**: Cu, Zn, Fe, Al, Cr, Mn, Co, Ni, Dy, Tb
- **Crystal Topologies**: bcu, mtn, pcu, cmo, dia

## 🎓 Key Publications & References

The methodology and findings build on:
- MOF synthesis and characterization literature
- Machine learning best practices for materials science
- Feature engineering for chemical/materials systems

## 💡 Applications

1. **MOF Design**: Predict formation energy for new compositions
2. **Materials Screening**: Identify promising MOF candidates
3. **Experiment Planning**: Guide synthesis parameter optimization
4. **Process Control**: Inform manufacturing decisions

## 🤝 Contributing

Contributions are welcome! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@software{mof_ml_2025,
  author = {Pratyush Bhartiya},
  title = {MOF Property Prediction Using Machine Learning},
  year = {2025},
  url = {https://github.com/yourusername/mof_ml_project}
}
```

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## ⚖️ Code of Conduct

Please note that this project is released with a [Contributor Code of Conduct](CODE_OF_CONDUCT.md). By participating in this project, you agree to abide by its terms.

## 🙋 Support & Issues

- **Questions?** Open an [Issue](https://github.com/yourusername/mof_ml_project/issues/new)
- **Bug Reports**: Use the [Bug Report](https://github.com/yourusername/mof_ml_project/issues/new?template=bug_report.md) template
- **Feature Requests**: Use the [Feature Request](https://github.com/yourusername/mof_ml_project/issues/new?template=feature_request.md) template

## 👥 Authors

- **Pratyush Bhartiya** - Initial work and research

## 🙏 Acknowledgments

- IIT Delhi Chemical Engineering department
- MOF research community
- Open-source ML community (scikit-learn, pandas, etc.)

## 📞 Contact

- **Email**: ch7231112@iitd.ac.in
- **LinkedIn**: [pratyush1710](https://www.linkedin.com/in/pratyush1710/)
- **GitHub**: [TechieMaNiAc1710](https://github.com/TechieMaNiAc1710)

---

**Last Updated**: December 19, 2025
**Status**: ✅ Production-Ready
