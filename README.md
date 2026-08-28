# COVID-19 Vaccination Prediction

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3%2B-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28%2B-red)
![License](https://img.shields.io/badge/License-MIT-green)

> Machine Learning pipeline for predicting COVID-19 vaccination rates globally, with focus on Iran as a case study.

## Live Demo

Run the Streamlit app locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Project Overview

This project implements a complete machine learning pipeline to predict daily COVID-19 vaccination counts using historical vaccination data. The project follows industry best practices for data science workflows, including data preprocessing, feature engineering, model training, evaluation, and deployment.

## Features

### Notebook
- **Comprehensive EDA**: Detailed exploratory data analysis with visualizations
- **Feature Engineering**: Time-based features, lag features, and rolling statistics
- **Multiple Models**: Random Forest, XGBoost, and LightGBM comparison
- **Time Series Handling**: Proper train/test split respecting temporal order
- **Model Evaluation**: RMSE, MAE, R2, and MAPE metrics
- **Production Ready**: Saved models and scaler for deployment

### Streamlit Web App
- **Country Selection**: Choose any country from the dataset
- **Interactive Dashboard**: 5 tabs with different visualizations
- **Real-time Predictions**: Watch model training and predictions
- **Model Comparison**: Compare multiple ML models side-by-side
- **Feature Importance**: Understand what drives predictions
- **Dark Theme**: Modern, attractive UI with gradient backgrounds
- **Responsive Design**: Works on desktop and mobile

## Project Structure

```
Covid-Vaccination/
├── Project1.ipynb              # Main notebook with complete pipeline
├── app.py                      # Streamlit web application
├── .streamlit/
│   └── config.toml             # Streamlit theme configuration
├── country_vaccinations.csv    # Primary dataset
├── country_vaccinations_by_manufacturer.csv  # Manufacturer data
├── requirements.txt            # Python dependencies
├── best_vaccination_model.pkl  # Trained model (generated)
├── scaler.pkl                  # Feature scaler (generated)
└── README.md                   # This file
```

## Dataset

- **Source**: Our World in Data (OWID)
- **Records**: Daily vaccination data for 200+ countries
- **Features**: Country, date, total vaccinations, people vaccinated, daily vaccinations, vaccine types, etc.
- **Target Country**: Iran

## Methodology

### 1. Data Loading & Preprocessing
- Load and filter data for target country (Iran)
- Handle missing values using forward/backward fill
- Create date-based features (month, day, dayofweek, quarter)

### 2. Feature Engineering
- **Lag Features**: 1, 3, 7, 14, 30-day lags
- **Rolling Statistics**: Mean and std for 3, 7, 14, 30-day windows
- **Cyclical Encoding**: Sine/cosine transformations for periodic features
- **Rate of Change**: Percentage change and differences

### 3. Model Training
- **Random Forest**: Ensemble of decision trees
- **XGBoost**: Gradient boosting with regularization
- **LightGBM**: Fast gradient boosting framework
- **Cross-Validation**: TimeSeriesSplit for temporal data

### 4. Evaluation Metrics
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- R2 (Coefficient of Determination)
- MAPE (Mean Absolute Percentage Error)

## Installation

```bash
# Clone the repository
git clone https://github.com/AlirezasDev/Covid-Vaccination.git
cd Covid-Vaccination

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Running the Notebook
```bash
jupyter notebook Project1.ipynb
```

### Running the Web App
```bash
streamlit run app.py
```

### Using the Saved Model
```python
import joblib
import pandas as pd

model = joblib.load('best_vaccination_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prepare your features
X_new = scaler.transform(your_features)
predictions = model.predict(X_new)
```

## Results

| Model | RMSE | MAE | R2 | MAPE |
|-------|------|-----|----|----|
| Random Forest | - | - | - | - |
| XGBoost | - | - | - | - |
| LightGBM | - | - | - | - |

*Results will be populated after running the notebook*

## Technologies Used

- Python 3.8+
- Pandas & NumPy
- Scikit-learn
- XGBoost
- LightGBM
- Matplotlib & Seaborn
- Streamlit
- Plotly
- Jupyter Notebook

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Our World in Data](https://ourworldindata.org/covid-vaccinations) for the dataset
- [Scikit-learn](https://scikit-learn.org/) documentation
- [XGBoost](https://xgboost.readthedocs.io/) documentation
- [LightGBM](https://lightgbm.readthedocs.io/) documentation
- [Streamlit](https://streamlit.io/) documentation

---

**Author**: Alireza Sepehri  
**Email**: alireza_sepehri@mathdep.iust.ac.ir  
**GitHub**: [@AlirezasDev](https://github.com/AlirezasDev)