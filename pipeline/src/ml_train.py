import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def train_and_save_models(data_path, output_dir):
    df = pd.read_csv(data_path)
    os.makedirs(output_dir, exist_ok=True)
    
    X = df.drop(columns=['Crop', 'Stage'], errors='ignore')
    y_crop = df['Crop']

    non_numeric_columns = X.select_dtypes(exclude='number').columns.tolist()
    if non_numeric_columns:
        raise ValueError(
            f"Data preparation must produce numeric model features: {non_numeric_columns}"
        )
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_crop, test_size=0.3, stratify=y_crop, random_state=42
    )
    
    # Build and Train Pipeline
    pipeline = RandomForestClassifier(
        class_weight='balanced',
        random_state=42,
    )
    
    pipeline.fit(X_train, y_train)
    
    # Save the trained model and test splits for evaluation step
    joblib.dump(pipeline, os.path.join(output_dir, 'crop_model.pkl'))
    joblib.dump((X_test, y_test), os.path.join(output_dir, 'test_data.pkl'))