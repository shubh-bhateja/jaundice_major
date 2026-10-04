import os
# Set environment variables before importing any modules that may load TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'           # suppress INFO/WARNING from TF
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['KMP_INIT_AT_FORK'] = 'FALSE'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from data_loader import load_clinical_data, build_dataset
from data_generator import MultimodalDataGenerator
from model_integration import build_multimodal_model

# Quick smoke test: verify dataset builds and generators produce expected shapes

def smoke_test(clinical_csv_path, images_dir):
    print("Running smoke test...")
    df = load_clinical_data(clinical_csv_path)
    if df is None:
        print("Failed to load clinical CSV.")
        return False
    dataset = build_dataset(df, images_dir)
    if len(dataset) == 0:
        print("No matched samples found. Ensure IDs in CSV match image filenames.")
        return False
    
    # Create a small generator
    gen = MultimodalDataGenerator(dataset=dataset[:8], batch_size=4, visual_seq_length=50000)
    X, y = gen[0]
    print("Visual input shape:", X['visual_input'].shape)
    print("Clinical input shape:", X['clinical_input'].shape)
    print("TSB output shape:", y['tsb_regression_output'].shape)
    print("Risk output shape:", y['risk_classification_output'].shape)
    return True

if __name__ == '__main__':
    # Default paths recommended by user
    CLINICAL_CSV = '/Users/shubhbhateja/College/minor_project/NeoJaundice/chd_jaundice_published_2.csv'
    IMAGES_DIR = '/Users/shubhbhateja/College/minor_project/images'
    smoke_test(CLINICAL_CSV, IMAGES_DIR)
