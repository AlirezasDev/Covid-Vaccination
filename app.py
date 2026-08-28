import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb
import lightgbm as lgb
import joblib
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="COVID-19 Vaccination Prediction",
    page_icon="v",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    .stMetric {
        background-color: #1e1e1e;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #333;
    }
    .stMetric label {
        color: #888 !important;
    }
    .stMetric div {
        color: #fff !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #fff !important;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 5px 5px 0 0;
    }
    div[data-testid="stSidebar"] {
        background-color: #1a1a1a;
    }
    .css-1d391kg {
        background-color: #1a1a1a;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_data():
    df = pd.read_csv('country_vaccinations.csv')
    df['date'] = pd.to_datetime(df['date'])
    return df


@st.cache_data
def prepare_country_data(df, country):
    country_df = df[df['country'] == country].copy()
    country_df = country_df.sort_values('date').reset_index(drop=True)
    return country_df


def create_features(df, target_col):
    df = df.copy()
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['dayofyear'] = df['date'].dt.dayofyear
    df['quarter'] = df['date'].dt.quarter
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    
    for lag in [1, 3, 7, 14, 30]:
        df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
    
    for window in [3, 7, 14, 30]:
        df[f'{target_col}_rolling_{window}_mean'] = df[target_col].rolling(window=window, min_periods=1).mean()
        df[f'{target_col}_rolling_{window}_std'] = df[target_col].rolling(window=window, min_periods=1).std()
    
    df[f'{target_col}_pct_change'] = df[target_col].pct_change()
    df[f'{target_col}_diff'] = df[target_col].diff()
    
    df = df.dropna().reset_index(drop=True)
    return df


def train_models(X_train, y_train, X_val, y_val):
    models = {
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=15, min_samples_split=5,
            min_samples_leaf=2, random_state=42, n_jobs=-1
        ),
        'XGBoost': xgb.XGBRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1
        ),
        'LightGBM': lgb.LGBMRegressor(
            n_estimators=300, max_depth=8, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=42,
            n_jobs=-1, verbose=-1
        )
    }
    
    trained_models = {}
    results = {}
    
    for name, model in models.items():
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)])
        trained_models[name] = model
        
        y_pred = model.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mape = np.mean(np.abs((y_val - y_pred) / np.maximum(np.abs(y_val), 1))) * 100
        
        results[name] = {
            'RMSE': rmse, 'MAE': mae, 'R2': r2, 'MAPE': mape,
            'predictions': y_pred
        }
    
    return trained_models, results


def main():
    st.sidebar.markdown("# Control Panel")
    st.sidebar.markdown("---")
    
    df = load_data()
    
    countries = sorted(df['country'].unique())
    selected_country = st.sidebar.selectbox("Select Country", countries, index=countries.index('Iran'))
    
    country_df = prepare_country_data(df, selected_country)
    
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                padding: 30px; border-radius: 15px; margin-bottom: 30px;'>
        <h1 style='color: white; text-align: center; margin: 0;'>
            COVID-19 Vaccination Prediction
        </h1>
        <p style='color: rgba(255,255,255,0.8); text-align: center; margin: 10px 0 0 0;'>
            Machine Learning Dashboard for {selected_country}
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Overview", "EDA", "Predictions", "Model Comparison", "Feature Importance"
    ])
    
    with tab1:
        st.markdown("## Dataset Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Records", f"{len(country_df):,}")
        with col2:
            st.metric("Date Range", f"{country_df['date'].min().strftime('%Y-%m-%d')} to {country_df['date'].max().strftime('%Y-%m-%d')}")
        with col3:
            st.metric("Avg Daily Vaccinations", f"{country_df['daily_vaccinations'].mean():,.0f}")
        with col4:
            st.metric("Max Daily Vaccinations", f"{country_df['daily_vaccinations'].max():,.0f}")
        
        st.markdown("---")
        st.markdown("### Vaccination Trend Over Time")
        
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                           subplot_titles=("Daily Vaccinations", "7-Day Rolling Average"),
                           vertical_spacing=0.1)
        
        fig.add_trace(
            go.Scatter(x=country_df['date'], y=country_df['daily_vaccinations'],
                      mode='lines', name='Daily', line=dict(width=1, color='#667eea')),
            row=1, col=1
        )
        
        rolling_mean = country_df['daily_vaccinations'].rolling(window=7, min_periods=1).mean()
        fig.add_trace(
            go.Scatter(x=country_df['date'], y=rolling_mean,
                      mode='lines', name='7-Day MA', line=dict(width=2, color='#e74c3c')),
            row=2, col=1
        )
        
        fig.update_layout(height=600, template='plotly_dark', 
                         showlegend=True,
                         paper_bgcolor='rgba(0,0,0,0)',
                         plot_bgcolor='rgba(0,0,0,0)')
        fig.update_xaxes(title_text='Date', row=2, col=1)
        fig.update_yaxes(title_text='Vaccinations', row=1, col=1)
        fig.update_yaxes(title_text='Vaccinations', row=2, col=1)
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.markdown("## Exploratory Data Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Distribution of Daily Vaccinations")
            fig = px.histogram(country_df, x='daily_vaccinations', nbins=30,
                             template='plotly_dark', color_discrete_sequence=['#667eea'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Log-Transformed Distribution")
            fig = px.histogram(country_df, x=np.log1p(country_df['daily_vaccinations'].dropna()), 
                             nbins=30, template='plotly_dark', color_discrete_sequence=['#e74c3c'])
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Correlation Heatmap")
        numeric_cols = country_df.select_dtypes(include=[np.number]).columns.tolist()
        if 'daily_vaccinations' in numeric_cols:
            corr_with_target = country_df[numeric_cols].corr()['daily_vaccinations'].abs().sort_values(ascending=False)
            top_features = corr_with_target.head(10).index.tolist()
            
            fig = px.imshow(country_df[top_features].corr(), 
                           template='plotly_dark', color_continuous_scale='RdBu_r',
                           aspect='auto')
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.markdown("## Model Training & Predictions")
        
        target_col = 'daily_vaccinations'
        processed_df = create_features(country_df, target_col)
        
        drop_cols = ['country', 'iso_code', 'vaccines', 'source_name', 'source_website', 'date', target_col]
        feature_cols = [col for col in processed_df.columns if col not in drop_cols]
        
        X = processed_df[feature_cols]
        y = processed_df[target_col]
        
        scaler = StandardScaler()
        X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns, index=X.index)
        
        split_idx = int(len(X_scaled) * 0.8)
        X_train, X_test = X_scaled.iloc[:split_idx], X_scaled.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        split_idx_val = int(len(X_train) * 0.8)
        X_train_final, X_val = X_train.iloc[:split_idx_val], X_train.iloc[split_idx_val:]
        y_train_final, y_val = y_train.iloc[:split_idx_val], y_train.iloc[split_idx_val:]
        
        with st.spinner("Training models..."):
            trained_models, results = train_models(X_train_final, y_train_final, X_val, y_val)
        
        best_model_name = min(results, key=lambda x: results[x]['RMSE'])
        best_model = trained_models[best_model_name]
        
        test_predictions = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, test_predictions))
        test_r2 = r2_score(y_test, test_predictions)
        
        st.success(f"Best Model: **{best_model_name}**")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Test RMSE", f"{test_rmse:,.2f}")
        with col2:
            st.metric("Test R2", f"{test_r2:.4f}")
        with col3:
            st.metric("Training Samples", f"{len(X_train):,}")
        with col4:
            st.metric("Test Samples", f"{len(X_test):,}")
        
        st.markdown("### Predictions vs Actual (Test Set)")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(y_test))), y=y_test.values,
                                mode='lines', name='Actual', line=dict(color='#667eea', width=2)))
        fig.add_trace(go.Scatter(x=list(range(len(test_predictions))), y=test_predictions,
                                mode='lines', name='Predicted', line=dict(color='#e74c3c', width=2, dash='dash')))
        fig.update_layout(title=f'{best_model_name} - Predictions vs Actual',
                         xaxis_title='Time Index', yaxis_title='Daily Vaccinations',
                         template='plotly_dark', height=500,
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Scatter Plot: Predicted vs Actual")
        fig = px.scatter(x=y_test.values, y=test_predictions, 
                        template='plotly_dark', color_discrete_sequence=['#667eea'])
        fig.add_trace(go.Scatter(x=[y_test.min(), y_test.max()], y=[y_test.min(), y_test.max()],
                                mode='lines', name='Perfect Prediction', line=dict(color='red', dash='dash')))
        fig.update_layout(xaxis_title='Actual', yaxis_title='Predicted',
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.markdown("## Model Comparison")
        
        metrics_df = pd.DataFrame({name: {k: v for k, v in res.items() if k not in ['predictions']}
                                  for name, res in results.items()}).T
        
        st.dataframe(metrics_df.style.highlight_min(axis=0, subset=['RMSE', 'MAE', 'MAPE'])
                    .highlight_max(axis=0, subset=['R2']),
                    use_container_width=True)
        
        fig = go.Figure()
        for name, res in results.items():
            fig.add_trace(go.Bar(name=name, x=['RMSE', 'MAE', 'R2', 'MAPE'],
                                y=[res['RMSE'], res['MAE'], res['R2'], res['MAPE']]))
        
        fig.update_layout(barmode='group', template='plotly_dark',
                         title='Model Metrics Comparison',
                         paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### Cross-Validation Results")
        cv_data = []
        for name, model in trained_models.items():
            tscv = TimeSeriesSplit(n_splits=5)
            scores = cross_val_score(model, X_scaled, y, cv=tscv,
                                    scoring='neg_root_mean_squared_error', n_jobs=-1)
            cv_data.append({
                'Model': name,
                'Mean RMSE': -scores.mean(),
                'Std RMSE': scores.std(),
                'Min RMSE': -scores.max(),
                'Max RMSE': -scores.min()
            })
        
        cv_df = pd.DataFrame(cv_data)
        st.dataframe(cv_df, use_container_width=True)
    
    with tab5:
        st.markdown("## Feature Importance")
        
        model_to_explain = st.selectbox("Select Model", list(trained_models.keys()))
        
        if hasattr(trained_models[model_to_explain], 'feature_importances_'):
            importances = pd.Series(trained_models[model_to_explain].feature_importances_,
                                   index=X.columns).sort_values(ascending=False)
            
            fig = px.bar(x=importances.head(15).values, y=importances.head(15).index,
                        orientation='h', template='plotly_dark',
                        color=importances.head(15).values,
                        color_continuous_scale='viridis')
            fig.update_layout(title=f'Top 15 Feature Importances - {model_to_explain}',
                             xaxis_title='Importance', yaxis_title='Feature',
                             paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             height=500)
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("### Feature Importance Table")
            importance_df = pd.DataFrame({
                'Feature': importances.index,
                'Importance': importances.values
            })
            st.dataframe(importance_df, use_container_width=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Download")
    
    if st.sidebar.button("Save Model"):
        joblib.dump(best_model, 'best_vaccination_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        st.sidebar.success("Model saved!")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **Author:** Alireza Sepehri  
    **Email:** alireza_sepehri@mathdep.iust.ac.ir  
    **GitHub:** [@AlirezasDev](https://github.com/AlirezasDev)
    """)


if __name__ == "__main__":
    main()