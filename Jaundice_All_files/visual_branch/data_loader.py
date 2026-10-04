import os
import pandas as pd
import numpy as np
from pathlib import Path

def load_clinical_data(csv_path):
    """
    Loads clinical metadata from a CSV or Excel file.
    Expected to have columns like: 'Infant_ID', 'Weight', 'Gestational_Age', 'Postnatal_Age', 'TSB', 'Risk_Class'
    """
    try:
        if csv_path.endswith('.csv'):
            df = pd.read_csv(csv_path)
        elif csv_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(csv_path)
        else:
            raise ValueError("Unsupported file format. Please provide a .csv or .xlsx file.")
            
        print(f"Loaded {len(df)} clinical records from {csv_path}.")
        return df
    except Exception as e:
        print(f"Error loading clinical data: {e}")
        return None


def normalize_clinical_features(df, feature_cols):
    """
    Normalizes the clinical features (e.g., Min-Max scaling or Z-score normalization).
    This is critical for neural networks to converge properly.
    """
    df_normalized = df.copy()
    for col in feature_cols:
        if col in df_normalized.columns:
            min_val = df_normalized[col].min()
            max_val = df_normalized[col].max()
            if pd.notna(min_val) and pd.notna(max_val) and max_val > min_val:
                df_normalized[col] = (df_normalized[col] - min_val) / (max_val - min_val)
    return df_normalized


def _detect_id_column(df):
    """Try to detect an Infant ID column from common candidates."""
    candidates = ['Infant_ID', 'InfantID', 'ID', 'Patient_ID', 'PatientID', 'infant_id']
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: try to find a column with small number of unique values/string-like
    for c in df.columns:
        if df[c].dtype == object:
            return c
    return None


def build_dataset(metadata_df, images_dir):
    """
    Maps each image in the processed_skin/ folder to its clinical metadata.
    Creates a unified list of samples.
    """
    dataset = []

    # Map expected columns to actual CSV columns
    column_map = {
        'Infant_ID': 'patient_id',
        'TSB': 'blood(mg/dL)',
        'Risk_Class': 'Treatment',
        'Weight': 'weight',
        'Gestational_Age': 'gestational_age',
        'Postnatal_Age': 'age(day)'
    }
    # Use mapped names for required columns
    id_col = column_map['Infant_ID']
    feature_cols = [column_map['Weight'], column_map['Gestational_Age'], column_map['Postnatal_Age']]
    target_tsb_col = column_map['TSB']
    target_risk_col = column_map['Risk_Class']

    # Basic validation: ensure id and TSB exist
    required_cols = [id_col, target_tsb_col, target_risk_col]
    missing_cols = [col for col in required_cols if col not in metadata_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in metadata: {missing_cols}")

    # If some features are missing, reduce feature_cols to available ones
    feature_cols = [c for c in feature_cols if c in metadata_df.columns]

    # Optional: Normalize features here or later in a pipeline
    if feature_cols:
        metadata_df = normalize_clinical_features(metadata_df, feature_cols)

    # Convert Risk_Class to integer if it's not numeric
    if metadata_df[target_risk_col].dtype == object or not np.issubdtype(metadata_df[target_risk_col].dtype, np.number):
        metadata_df[target_risk_col], uniques = pd.factorize(metadata_df[target_risk_col])
        print(f"Encoded Risk_Class with mapping: {dict(enumerate(uniques))}")

    # Iterate through the images in the directory
    valid_extensions = ['.npy', '.jpg', '.png']

    print(f"Scanning {images_dir} for processed files...")
    files_processed = 0
    matches_found = 0

    for filename in os.listdir(images_dir):
        if not any(filename.lower().endswith(ext) for ext in valid_extensions):
            continue

        files_processed += 1

        base_name = os.path.splitext(filename)[0]
        # Use image_idx for matching
        record = metadata_df[metadata_df['image_idx'] == filename]

        if not record.empty:
            record = record.iloc[0]
            image_path = os.path.join(images_dir, filename)

            # Extract Clinical Vector
            clinical_vector = []
            for col in feature_cols:
                try:
                    clinical_vector.append(float(record[col]))
                except Exception:
                    clinical_vector.append(0.0)
            clinical_vector = np.array(clinical_vector, dtype=np.float32)

            # Extract Targets
            try:
                tsb_value = float(record[target_tsb_col])
            except Exception:
                tsb_value = 0.0

            # Risk class now ensured to be integer via factorize above
            risk_class = int(record[target_risk_col])

            sample = {
                'image_path': image_path,
                'clinical_vector': clinical_vector,
                'tsb_value': tsb_value,
                'risk_class': risk_class,
                'infant_id': record[id_col]
            }
            dataset.append(sample)
            matches_found += 1
        else:
            # No clinical record found for this image
            pass

    print(f"\nSummary:")
    print(f"- Processed image files checked: {files_processed}")
    print(f"- Successful clinical metadata matches: {matches_found}")

    return dataset

def test_data_loader():
    """
    A small helper to test the data loader without a real dataset.
    It creates dummy data to show how it works.
    """
    print("--- Running Data Loader Test ---")
    
    # 1. Create a dummy CSV file
    dummy_csv_path = 'dummy_clinical_data.csv'
    dummy_data = {
        'Infant_ID': ['B001', 'B002', 'B003'],
        'Weight': [2.5, 3.1, 1.8],         # kg
        'Gestational_Age': [38, 40, 34],   # weeks
        'Postnatal_Age': [48, 24, 72],     # hours
        'TSB': [12.5, 8.2, 18.1],          # mg/dL
        'Risk_Class': [1, 0, 2]            # 0: Low, 1: Medium, 2: High
    }
    pd.DataFrame(dummy_data).to_csv(dummy_csv_path, index=False)
    
    # 2. Create a dummy processed_skin directory with fake sequence files
    dummy_dir = 'dummy_processed_skin'
    os.makedirs(dummy_dir, exist_ok=True)
    
    # Create matching and non-matching dummy files
    for f in ['B001_seq.npy', 'B002_seq.npy', 'B003_seq.npy', 'B009_seq.npy']:
        Path(os.path.join(dummy_dir, f)).touch()
        
    
    # 3. Run the pipeline
    print("\nExecuting Pipeline:")
    df = load_clinical_data(dummy_csv_path)
    if df is not None:
        dataset = build_dataset(df, dummy_dir)
        
        print("\nFirst Sample Output:")
        if dataset:
            first_sample = dataset[0]
            for key, value in first_sample.items():
                print(f"  {key}: {value} (Type: {type(value)})")
                
    # 4. Cleanup
    os.remove(dummy_csv_path)
    for f in os.listdir(dummy_dir):
        os.remove(os.path.join(dummy_dir, f))
    os.rmdir(dummy_dir)
    print("--- Test Complete ---")


if __name__ == "__main__":
    test_data_loader()
    pass
