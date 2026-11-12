"""
Machine Learning models for credit card fraud detection
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, confusion_matrix, roc_auc_score, roc_curve)
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import joblib
import streamlit as st
import warnings
warnings.filterwarnings('ignore')

class FraudDetectionModel:
    """
    Credit Card Fraud Detection Model class
    Handles data preprocessing, model training, and prediction
    """
    
    def __init__(self):
        """Initialize the model with None values"""
        self.scaler = None
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = None
        
    def preprocess_data(self, df):
        """
        Preprocess the data: handle missing values, scale features
        
        Parameters:
        df: Input DataFrame
        
        Returns:
        X_scaled: Scaled features
        y: Target variable
        """
        # Handle missing values
        df = df.dropna()
        
        # Separate features and target
        X = df.drop('Class', axis=1)
        y = df['Class']
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale 'Amount' and 'Time' if they exist
        if 'Amount' in X.columns and 'Time' in X.columns:
            self.scaler = StandardScaler()
            X[['Amount', 'Time']] = self.scaler.fit_transform(X[['Amount', 'Time']])
        
        return X, y
    
    def handle_imbalance(self, X, y):
        """
        Handle class imbalance using SMOTE
        
        Parameters:
        X: Features
        y: Target variable
        
        Returns:
        X_resampled, y_resampled: Resampled data
        """
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        return X_resampled, y_resampled
    
    def train_models(self, X_train, y_train):
        """
        Train multiple models and return their performance
        
        Parameters:
        X_train: Training features
        y_train: Training target
        
        Returns:
        Dictionary of trained models
        """
        # Initialize models with optimized parameters for imbalanced data
        models = {
            'Logistic Regression': LogisticRegression(
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            ),
            'XGBoost': xgb.XGBClassifier(
                scale_pos_weight=len(y_train[y_train==0])/max(len(y_train[y_train==1]), 1),
                random_state=42,
                eval_metric='logloss'
            )
        }
        
        # Train each model
        for name, model in models.items():
            with st.spinner(f'Training {name}...'):
                model.fit(X_train, y_train)
            self.models[name] = model
        
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """
        Evaluate all trained models and return metrics
        
        Parameters:
        X_test: Test features
        y_test: Test target
        
        Returns:
        DataFrame with model performances
        """
        results = []
        
        for name, model in self.models.items():
            # Make predictions
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Calculate metrics
            metrics = {
                'Model': name,
                'Accuracy': accuracy_score(y_test, y_pred),
                'Precision': precision_score(y_test, y_pred),
                'Recall': recall_score(y_test, y_pred),
                'F1-Score': f1_score(y_test, y_pred),
                'ROC-AUC': roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
            }
            results.append(metrics)
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Select best model (based on F1-Score)
        self.best_model_name = results_df.loc[results_df['F1-Score'].idxmax(), 'Model']
        self.best_model = self.models[self.best_model_name]
        
        return results_df
    
    def get_confusion_matrix(self, model, X_test, y_test):
        """
        Get confusion matrix for a specific model
        
        Parameters:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
        Returns:
        Confusion matrix array
        """
        y_pred = model.predict(X_test)
        return confusion_matrix(y_test, y_pred)
    
    def get_roc_curve_data(self, model, X_test, y_test):
        """
        Get ROC curve data for a specific model
        
        Parameters:
        model: Trained model
        X_test: Test features
        y_test: Test target
        
        Returns:
        fpr, tpr, thresholds
        """
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
            return fpr, tpr
        return None, None
    
    def predict_single(self, features_dict):
        """
        Predict fraud probability for a single transaction
        
        Parameters:
        features_dict: Dictionary with feature values
        
        Returns:
        prediction: 0 or 1 (Legitimate or Fraud)
        probability: Probability of being fraud
        """
        # Convert dictionary to array
        features = np.array([features_dict[feature] for feature in self.feature_names]).reshape(1, -1)
        
        # Scale Amount and Time if they exist
        if 'Amount' in features_dict and 'Time' in features_dict:
            amount_idx = self.feature_names.index('Amount')
            time_idx = self.feature_names.index('Time')
            scale_vals = np.array([[features[0, amount_idx], features[0, time_idx]]])
            scaled = self.scaler.transform(scale_vals)[0]
            features[0, amount_idx] = scaled[0]
            features[0, time_idx] = scaled[1]
        
        # Make prediction
        prediction = self.best_model.predict(features)[0]
        probability = self.best_model.predict_proba(features)[0][1]
        
        return prediction, probability
    
    def predict_bulk(self, df):
        """
        Predict fraud for multiple transactions
        
        Parameters:
        df: DataFrame with transaction data
        
        Returns:
        DataFrame with predictions
        """
        # Preprocess
        X = df[self.feature_names].copy()
        
        # Scale Amount and Time if they exist
        if 'Amount' in X.columns and 'Time' in X.columns:
            X[['Amount', 'Time']] = self.scaler.transform(X[['Amount', 'Time']])
        
        # Make predictions
        predictions = self.best_model.predict(X)
        probabilities = self.best_model.predict_proba(X)[:, 1]
        
        # Add to DataFrame
        result_df = df.copy()
        result_df['Prediction'] = predictions
        result_df['Fraud_Probability'] = probabilities
        result_df['Result'] = result_df['Prediction'].map({0: 'Legitimate', 1: 'Fraudulent'})
        
        return result_df
    
    def save_model(self, filepath='fraud_model.pkl'):
        """
        Save the trained model and scaler
        
        Parameters:
        filepath: Path to save the model
        """
        model_data = {
            'best_model': self.best_model,
            'best_model_name': self.best_model_name,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'all_models': self.models
        }
        joblib.dump(model_data, filepath)
        return filepath
    
    def load_model(self, filepath='fraud_model.pkl'):
        """
        Load a trained model
        
        Parameters:
        filepath: Path to the saved model
        """
        model_data = joblib.load(filepath)
        self.best_model = model_data['best_model']
        self.best_model_name = model_data['best_model_name']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.models = model_data['all_models']
        return True
