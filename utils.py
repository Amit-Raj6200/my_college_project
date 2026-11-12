# Utility functions for credit card fraud detection project
"""
Utility functions for data visualization and processing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

def load_and_prepare_data():
    """
    Load and prepare the credit card fraud dataset.
    Note: In production, you'd download from Kaggle.
    For demo, we'll generate synthetic data similar to the real dataset.
    """
    try:
        # Try to load real dataset if available
        df = pd.read_csv('creditcard.csv')
        return df
    except FileNotFoundError:
        # Generate synthetic data for demonstration
        st.warning("Real dataset not found. Generating synthetic data for demonstration...")
        return generate_synthetic_data()

def generate_synthetic_data(n_samples=10000):
    """
    Generate synthetic credit card transaction data similar to the real dataset
    
    Parameters:
    n_samples: Number of samples to generate
    
    Returns:
    DataFrame with synthetic transaction data
    """
    np.random.seed(42)
    
    # Generate V1 to V28 features (anonymized)
    n_features = 28
    X = np.random.randn(n_samples, n_features)
    
    # Generate Time feature
    time = np.random.uniform(0, 172792, n_samples)
    
    # Generate Amount feature (log-normal distribution)
    amount = np.random.lognormal(mean=4, sigma=1, size=n_samples)
    amount = amount * 100  # Scale to typical transaction amounts

    # Generate Class labels (fraudulent transactions are rare)
    fraud_ratio = 0.00172  # ~0.172% fraud rate like real dataset
    class_labels = np.random.choice([0, 1], size=n_samples, p=[1-fraud_ratio, fraud_ratio])
    
    # Modify features to create patterns for fraud detection
    # Fraudulent transactions have slightly different patterns
    for i in range(n_samples):
        if class_labels[i] == 1:
            # Add some pattern to V features for fraud cases
            X[i, :10] += np.random.randn(10) * 0.5
            amount[i] = amount[i] * np.random.uniform(0.5, 3)
    
    # Create DataFrame
    columns = [f'V{i}' for i in range(1, n_features + 1)]
    df = pd.DataFrame(X, columns=columns)
    df['Time'] = time
    df['Amount'] = amount
    df['Class'] = class_labels
    
    return df

def create_class_distribution_chart(df):
    """
    Create a pie chart showing class distribution
    
    Parameters:
    df: DataFrame with 'Class' column
    
    Returns:
    Plotly figure
    """
    class_counts = df['Class'].value_counts()
    labels = ['Legitimate (0)', 'Fraudulent (1)']
    values = [class_counts[0], class_counts[1] if 1 in class_counts else 0]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=.3,
        marker_colors=['#2ecc71', '#e74c3c']
    )])
    
    fig.update_layout(
        title="Transaction Class Distribution",
        annotations=[dict(text=f'Total: {len(df):,}', x=0.5, y=0.5, font_size=20, showarrow=False)]
    )
    
    return fig

def create_amount_distribution_chart(df):
    """
    Create box plot showing amount distribution for fraud vs legitimate transactions
    
    Parameters:
    df: DataFrame with 'Amount' and 'Class' columns
    
    Returns:
    Plotly figure
    """
    fraud_df = df[df['Class'] == 1]
    legit_df = df[df['Class'] == 0]
    
    fig = make_subplots(rows=1, cols=2, subplot_titles=('All Transactions', 'Fraudulent Transactions'))
    
    # Box plot for all transactions
    fig.add_trace(
        go.Box(y=df['Amount'], name='All Transactions', boxmean='sd'),
        row=1, col=1
    )
    
    # Distribution for fraudulent transactions
    fig.add_trace(
        go.Histogram(x=fraud_df['Amount'], nbinsx=50, name='Fraud Amounts', 
                     marker_color='red', opacity=0.7),
        row=1, col=2
    )
    
    fig.update_layout(
        title="Transaction Amount Analysis",
        height=500,
        showlegend=True
    )
    
    fig.update_xaxes(title_text="Transaction Type", row=1, col=1)
    fig.update_xaxes(title_text="Amount ($)", row=1, col=2)
    fig.update_yaxes(title_text="Amount ($)", row=1, col=1)
    
    return fig

def create_correlation_heatmap(df, n_features=10):
    """
    Create correlation heatmap for selected features
    
    Parameters:
    df: DataFrame
    n_features: Number of features to include
    
    Returns:
    matplotlib figure
    """
    # Select top features with highest correlation to Class
    correlations = df.corr()['Class'].abs().sort_values(ascending=False)
    top_features = correlations.head(n_features + 1).index.tolist()
    
    # Create correlation matrix for selected features
    corr_matrix = df[top_features].corr()
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, ax=ax, cbar_kws={"shrink": 0.8})
    ax.set_title(f'Correlation Heatmap (Top {n_features} Features with Class)', fontsize=14, fontweight='bold')
    
    return fig

def create_time_analysis_chart(df):
    """
    Create time-based analysis chart
    
    Parameters:
    df: DataFrame with 'Time' and 'Class' columns
    
    Returns:
    Plotly figure
    """
    fig = make_subplots(rows=2, cols=1, subplot_titles=('Fraud over Time', 'Density of Transactions'))
    
    # Time series of fraud occurrences
    fraud_over_time = df[df['Class'] == 1].groupby(df['Time'] // 10000).size()
    
    fig.add_trace(
        go.Scatter(x=fraud_over_time.index * 10000, y=fraud_over_time.values,
                   mode='lines+markers', name='Fraud Cases', line=dict(color='red', width=2)),
        row=1, col=1
    )
    
    # Density plot
    fig.add_trace(
        go.Histogram(x=df['Time'], nbinsx=50, name='All Transactions', 
                     marker_color='blue', opacity=0.5, histnorm='density'),
        row=2, col=1
    )
    
    fig.add_trace(
        go.Histogram(x=df[df['Class'] == 1]['Time'], nbinsx=50, name='Fraud Transactions',
                     marker_color='red', opacity=0.7, histnorm='density'),
        row=2, col=1
    )
    
    fig.update_layout(
        title="Time-based Analysis",
        height=600,
        showlegend=True,
        barmode='overlay'
    )
    
    fig.update_xaxes(title_text="Time (seconds)", row=1, col=1)
    fig.update_xaxes(title_text="Time (seconds)", row=2, col=1)
    fig.update_yaxes(title_text="Number of Frauds", row=1, col=1)
    fig.update_yaxes(title_text="Density", row=2, col=1)
    
    return fig

def create_feature_importance_chart(model, feature_names, top_n=15):
    """
    Create feature importance chart for tree-based models
    
    Parameters:
    model: Trained model with feature_importances_ attribute
    feature_names: List of feature names
    top_n: Number of top features to display
    
    Returns:
    Plotly figure
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Create DataFrame for feature importances
        feat_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values('Importance', ascending=False).head(top_n)
        
        fig = px.bar(feat_importance, x='Importance', y='Feature', 
                     orientation='h', title=f'Top {top_n} Feature Importances',
                     color='Importance', color_continuous_scale='Viridis')
        
        fig.update_layout(height=500, yaxis={'categoryorder': 'total ascending'})
        return fig
    else:
        return None
