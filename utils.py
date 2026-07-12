import pandas as pd
import joblib

try:
    scaler = joblib.load('models/scaler.pkl')
    imputer = joblib.load('models/imputer.pkl')
    encoder = joblib.load('models/encoder.pkl')
except:
    print("Warning: Model assets not found. Ensure .pkl files exist in 'models/'.")

NUMERIC_COLS = ['age', 'cigsPerDay', 'totChol', 'sysBP', 'BMI', 'glucose']
DROP_COLS = ['prevalentStroke', 'diabetes', 'BPMeds']

def preprocess_input(data_dict):
   
    df = pd.DataFrame([data_dict])
    
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].map({'Female': 0, 'Male': 1})

    if 'education' in df.columns:
        df[['education']] = encoder.transform(df[['education']])
    df_imputed = pd.DataFrame(
        imputer.transform(df),
        columns=df.columns,
        index=df.index
    )

    df_imputed[NUMERIC_COLS] = scaler.transform(df_imputed[NUMERIC_COLS])

    return df_imputed