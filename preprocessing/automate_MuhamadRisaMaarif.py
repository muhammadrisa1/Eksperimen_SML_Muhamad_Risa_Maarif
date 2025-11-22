import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import os 

def load_data(file_path):
    print(f"Loading data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Data loaded: {df.shape}")
    return df

def clean_data(df):
    print("\nCleaning data...")
    initial_shape = df.shape
    df = df.drop_duplicates()

    df = df.dropna()

    print(f"Cleaning done: {initial_shape} {df.shape}" )
    return df

def create_features(df):
    print("\nCreating features...")
    df = df.copy()
    
    # Time-based features
    df['transaction_hour'] = (df['Time'] // 3600) % 24
    df['is_off_hours'] = ((df['transaction_hour'] <= 6) | 
                          (df['transaction_hour'] >= 22)).astype(int)
    
    # Amount behavior
    df['amount_std_score'] = (df['Amount'] - df['Amount'].mean()) / df['Amount'].std()
    df['is_amount_outlier'] = (df['amount_std_score'].abs() > 3).astype(int)
    
    # V-features interactions
    df['v_negative_composite'] = df['V14'] + df['V12'] + df['V17']
    df['v_positive_composite'] = df['V4'] + df['V11'] + df['V2']
    
    # Ratio features
    df['v14_to_amount'] = df['V14'] / (df['Amount'] + 0.001)
    df['time_to_amount'] = df['Time'] / (df['Amount'] + 0.001)
    
    # Binary flags
    df['extreme_v14'] = (df['V14'] < df['V14'].quantile(0.01)).astype(int)
    df['extreme_v12'] = (df['V12'] < df['V12'].quantile(0.01)).astype(int)
    
    print(f"Features created: {df.shape}")
    return df

def select_features(df, top_n=15):
    print(f"\nSelecting top {top_n} features...")
    
    correlation = df.corr()['Class'].abs().sort_values(ascending=False)
    selected_features = correlation[1:top_n+1].index.tolist()
    
    X = df[selected_features]
    y = df['Class']
    
    print(f"Selected features: {selected_features}")
    return X, y, selected_features

def scale_features(X):
    print("\nScaling features...")
    
    X_scaled = X.copy()
    
    standard_cols = [col for col in X.columns if col.startswith('V') or col == 'Time']
    robust_cols = [col for col in X.columns if 'amount' in col.lower()]
    
    if standard_cols:
        scaler_std = StandardScaler()
        X_scaled[standard_cols] = scaler_std.fit_transform(X[standard_cols])
    
    if robust_cols:
        scaler_robust = RobustScaler()
        X_scaled[robust_cols] = scaler_robust.fit_transform(X[robust_cols])
    
    print("Scaling completed")
    return X_scaled

def split_data(X, y, test_size=0.2, random_state=42):
    print(f"\nSplitting data (test_size={test_size})...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"Train set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test

def balance_data(X_train, y_train, random_state=42):
    print("\nBalancing data with SMOTE...")
    
    print(f"Before - Normal: {(y_train == 0).sum()}, Fraud: {(y_train == 1).sum()}")
    
    smote = SMOTE(random_state=random_state)
    X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"After  - Normal: {(y_balanced == 0).sum()}, Fraud: {(y_balanced == 1).sum()}")
    print("Balancing completed")
    
    return X_balanced, y_balanced

def save_processed_data(X_train, y_train, X_test, y_test, output_dir='preprocessing'):
    print(f"\nSaving processed data to {output_dir}/...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    X_train.to_csv(f'{output_dir}/creditcard_train_x.csv', index=False)
    pd.DataFrame(y_train, columns=['Class']).to_csv(f'{output_dir}/creditcard_train_y.csv', index=False)
    X_test.to_csv(f'{output_dir}/creditcard_test_X.csv', index=False)
    y_test.to_csv(f'{output_dir}/creditcard_test_y.csv', index=False)
    print(" All files saved successfully!")

def preprocess_pipeline(input_file, output_dir='preprocessing'):
    print("AUTOMATED PREPROCESSING PIPELINE")
    
    # 1. Load data
    df = load_data(input_file)
    
    # 2. Clean data
    df_clean = clean_data(df)
    
    # 3. Feature engineering
    df_enhanced = create_features(df_clean)
    
    # 4. Feature selection
    X, y, selected_features = select_features(df_enhanced)
    
    # 5. Feature scaling
    X_scaled = scale_features(X)
    
    # 6. Train-test split
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)
    
    # 7. Balance training data
    X_train_balanced, y_train_balanced = balance_data(X_train, y_train)
    
    # 8. Save processed data
    save_processed_data(X_train_balanced, y_train_balanced, X_test, y_test, output_dir)
    
    print("PREPROCESSING COMPLETED SUCCESSFULLY!")
    
    return X_train_balanced, y_train_balanced, X_test, y_test

if __name__ == "__main__":
    X_train, y_train, X_test, y_test = preprocess_pipeline(
        input_file='creditcard_raw.csv',
        output_dir='.'
    )
    
    print(f"\nFinal shapes:")
    print(f"Train: X={X_train.shape}, y={y_train.shape}")
    print(f"Test:  X={X_test.shape} ")
