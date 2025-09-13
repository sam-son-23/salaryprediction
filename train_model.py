# Complete Model Training Script for Salary Prediction System
# Save this as: train_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import warnings
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

def prepare_combined_dataset():
    """Combine and prepare the datasets for training"""
    
    print("Loading datasets...")
    
    # Load both datasets
    try:
        synthetic_data = pd.read_csv('synthetic_salary_data.csv')
        real_data = pd.read_csv('salary_prediction_dataset_20250407_124051.csv')
        
        print(f"Synthetic data shape: {synthetic_data.shape}")
        print(f"Real data shape: {real_data.shape}")
        
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        print("Please ensure both CSV files are in the same directory as this script.")
        return None
    
    # Handle missing values in synthetic data
    synthetic_data['certifications'] = synthetic_data['certifications'].fillna('None')
    synthetic_data['skills'] = synthetic_data['skills'].fillna('Cloud')
    
    # Align real data structure to match synthetic data
    real_data_aligned = pd.DataFrame()
    real_data_aligned['current_company'] = real_data['current_company']
    real_data_aligned['target_company'] = real_data['target_company']
    real_data_aligned['years_of_experience'] = real_data['years_of_experience']
    real_data_aligned['current_salary'] = real_data['current_salary']
    real_data_aligned['expected_salary'] = real_data['expected_salary']
    real_data_aligned['gender'] = real_data['gender']
    real_data_aligned['location'] = real_data['location']
    real_data_aligned['current_role'] = real_data['current_role']
    real_data_aligned['sector'] = real_data['sector']
    real_data_aligned['current_company_tier'] = real_data['current_company_tier']
    real_data_aligned['target_company_tier'] = real_data['target_company_tier']
    
    # Add missing columns with realistic distributions
    np.random.seed(42)
    real_data_aligned['education'] = np.random.choice(
        ['Bachelor', 'Master', 'PhD'], 
        size=len(real_data_aligned),
        p=[0.6, 0.35, 0.05]  # More realistic distribution
    )
    real_data_aligned['certifications'] = np.random.choice(
        ['None', 'AWS', 'Azure', 'GCP', 'PMP', 'Scrum Master'], 
        size=len(real_data_aligned),
        p=[0.4, 0.2, 0.15, 0.1, 0.1, 0.05]
    )
    real_data_aligned['skills'] = np.random.choice(
        ['Cloud', 'AI', 'DevOps', 'Full Stack', 'Data Engineering', 'Security'], 
        size=len(real_data_aligned),
        p=[0.25, 0.15, 0.2, 0.2, 0.15, 0.05]
    )
    
    # Combine datasets
    combined_data = pd.concat([synthetic_data, real_data_aligned], ignore_index=True)
    
    print(f"Combined data shape: {combined_data.shape}")
    
    return combined_data

def clean_and_preprocess_data(df):
    """Clean and preprocess the dataset"""
    
    print("Cleaning and preprocessing data...")
    
    # Remove salary outliers using IQR method
    Q1 = df['expected_salary'].quantile(0.25)
    Q3 = df['expected_salary'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    df_clean = df[
        (df['expected_salary'] >= lower_bound) & 
        (df['expected_salary'] <= upper_bound)
    ]
    
    print(f"Removed {len(df) - len(df_clean)} outliers")
    print(f"Clean data shape: {df_clean.shape}")
    
    # Handle missing values
    df_clean = df_clean.dropna()
    
    return df_clean

def encode_categorical_features(df):
    """Encode categorical features using LabelEncoder"""
    
    print("Encoding categorical variables...")
    
    categorical_columns = [
        'current_company', 'target_company', 'gender', 'location', 
        'current_role', 'sector', 'education', 'certifications', 'skills'
    ]
    
    label_encoders = {}
    df_encoded = df.copy()
    
    for col in categorical_columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        label_encoders[col] = le
        print(f"  {col}: {len(le.classes_)} unique values")
    
    return df_encoded, label_encoders

def train_model(X_train, y_train):
    """Train the salary prediction model"""
    
    print("Training RandomForest model...")
    
    # You can try XGBoost if you have it installed:
    # from xgboost import XGBRegressor
    # model = XGBRegressor(
    #     n_estimators=100,
    #     max_depth=6,
    #     learning_rate=0.1,
    #     random_state=42,
    #     n_jobs=-1
    # )
    
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    
    return model

def evaluate_model(model, X_test, y_test, feature_columns):
    """Evaluate model performance"""
    
    print("\nEvaluating model...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model Performance Metrics:")
    print(f"  Mean Absolute Error: ₹{mae:,.0f}")
    print(f"  Root Mean Square Error: ₹{rmse:,.0f}")
    print(f"  R² Score: {r2:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")
    
    return feature_importance

def create_feature_importance_plot(feature_importance):
    """Create and save feature importance plot"""
    
    try:
        plt.figure(figsize=(10, 6))
        top_features = feature_importance.head(10)
        
        sns.barplot(data=top_features, x='importance', y='feature')
        plt.title('Top 10 Feature Importance - Salary Prediction Model')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        
        # Save the plot (this is what the Streamlit app tries to load)
        plt.savefig('shap_feature_importance.png', dpi=300, bbox_inches='tight')
        print("Feature importance plot saved as 'shap_feature_importance.png'")
        
        plt.close()
        
    except Exception as e:
        print(f"Could not create plot: {e}")

def main():
    """Main training pipeline"""
    
    print("=" * 60)
    print("SALARY PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    # Load and combine datasets
    df = prepare_combined_dataset()
    if df is None:
        return
    
    # Clean and preprocess data
    df_clean = clean_and_preprocess_data(df)
    
    # Encode categorical features
    df_encoded, label_encoders = encode_categorical_features(df_clean)
    
    # Define features and target
    feature_columns = [
        'current_company', 'target_company', 'years_of_experience', 
        'current_salary', 'gender', 'location', 'current_role', 
        'sector', 'education', 'certifications', 'skills',
        'current_company_tier', 'target_company_tier'
    ]
    
    X = df_encoded[feature_columns]
    y = df_encoded['expected_salary']
    
    print(f"\nFinal dataset for training:")
    print(f"  Features shape: {X.shape}")
    print(f"  Target shape: {y.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    feature_importance = evaluate_model(model, X_test, y_test, feature_columns)
    
    # Create feature importance plot
    create_feature_importance_plot(feature_importance)
    
    # Save model and encoders
    print("\nSaving model and encoders...")
    joblib.dump(model, 'salary_prediction_model.joblib')
    joblib.dump(label_encoders, 'label_encoders.joblib')
    
    # Test loading
    print("Testing model loading...")
    try:
        loaded_model = joblib.load('salary_prediction_model.joblib')
        loaded_encoders = joblib.load('label_encoders.joblib')
        print("✅ Model and encoders saved and loaded successfully!")
        
        # Quick prediction test
        sample_prediction = loaded_model.predict(X_test[:1])
        print(f"Sample prediction test: ₹{sample_prediction[0]:,.0f}")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("You can now run: streamlit run 3_streamlit_app.py")
    print("=" * 60)

if __name__ == "__main__":
    main()