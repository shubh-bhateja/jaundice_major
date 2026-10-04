import os
import tensorflow as tf
from tensorflow.keras.optimizers import AdamW
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

# Import your custom modules
from data_loader import load_clinical_data, build_dataset
from data_generator import MultimodalDataGenerator
from model_integration import build_multimodal_model

def get_callbacks(model_save_path):
    """
    Defines Keras callbacks for training.
    """
    # 1. Early Stopping: Stop training when the validation loss stops improving.
    # 'val_loss' monitors the combined total loss (regression + classification).
    early_stopping = EarlyStopping(
        monitor='val_loss', 
        patience=10,            # Wait for 10 epochs without improvement
        restore_best_weights=True,
        verbose=1
    )
    
    # 2. Model Checkpoint: Save the best version of the multimodal model.
    # Saves automatically when the validation loss hits a new minimum.
    checkpoint = ModelCheckpoint(
        filepath=model_save_path,
        monitor='val_loss',
        save_best_only=True,
        mode='min',             # Lower loss is better
        verbose=1
    )
    
    # Optional: Reduce Learning Rate on Plateau
    # Helps the optimizer converge when it gets stuck near a minimum
    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,             # Reduce LR by factor of 0.2
        patience=5,             # Wait 5 epochs
        min_lr=1e-6,
        verbose=1
    )
    
    return [early_stopping, checkpoint, reduce_lr]

def train_pipeline(clinical_csv_path, processed_images_dir, model_save_path):
    """
    Main Training Pipeline.
    Executes Box 5 of the Methodology.
    """
    
    # ==========================================
    # 1. Load Data
    # ==========================================
    print("Loading Dataset...")
    metadata_df = load_clinical_data(clinical_csv_path)
    
    if metadata_df is None or metadata_df.empty:
         raise ValueError("Failed to load clinical metadata or file is empty.")
         
    # Build unified mapping list
    full_dataset = build_dataset(metadata_df, processed_images_dir)
    num_samples = len(full_dataset)
    
    if num_samples == 0:
        raise ValueError("No matching data found between clinical records and image files.")

    print(f"Total Combined Samples Found: {num_samples}")
    
    
    # ==========================================
    # 2. Split the Data (70% Train, 10% Val, 20% Test)
    # ==========================================
    print("\nSplitting Data...")
    
    # First split: 80% (Train + Val), 20% Test
    train_val_data, test_data = train_test_split(full_dataset, test_size=0.20, random_state=42)
    
    # Second split: 80% -> 70% Train, 10% Val
    # 0.125 * 0.80 = 0.10. Therefore, validation split is 12.5% of the remaining 80%
    train_data, val_data = train_test_split(train_val_data, test_size=0.125, random_state=42)
    
    print(f"  Training Samples:   {len(train_data)} (70%)")
    print(f"  Validation Samples: {len(val_data)} (10%)")
    print(f"  Test Samples:       {len(test_data)} (20%)")
    
    
    # ==========================================
    # 3. Create Data Generators
    # ==========================================
    # Define crucial model dimensions based on your specific architecture mapping 
    # Must match the `preprocess.py` output exactly
    VISUAL_SEQUENCE_LENGTH = 50000 
    
    # Using 3 clinical parameters: Weight, GA, PA
    CLINICAL_FEATURES_COUNT = 3     
    
    # E.g. 0: Low Risk, 1: Medium Risk, 2: High Risk
    NUM_RISK_CLASSES = 3            
    BATCH_SIZE = 16                 # Adjust based on memory
    
    print("\nInitializing Training Sequence Generators...")
    train_generator = MultimodalDataGenerator(
        dataset=train_data, batch_size=BATCH_SIZE, 
        num_classes=NUM_RISK_CLASSES, visual_seq_length=VISUAL_SEQUENCE_LENGTH, shuffle=True
    )
    
    val_generator = MultimodalDataGenerator(
        dataset=val_data, batch_size=BATCH_SIZE, 
        num_classes=NUM_RISK_CLASSES, visual_seq_length=VISUAL_SEQUENCE_LENGTH, shuffle=False
    )
    
    test_generator = MultimodalDataGenerator(
        dataset=test_data, batch_size=BATCH_SIZE, 
        num_classes=NUM_RISK_CLASSES, visual_seq_length=VISUAL_SEQUENCE_LENGTH, shuffle=False
    )
    
    
    # ==========================================
    # 4. Initialize the Mutlimodal Model & Optimizer
    # ==========================================
    print("\nBuilding Multimodal Dual-Head Architecture...")
    model = build_multimodal_model(
        visual_seq_length=VISUAL_SEQUENCE_LENGTH,
        clinical_features_count=CLINICAL_FEATURES_COUNT
    )
    
    # Initialize AdamW Optimizer
    # AdamW incorporates weight decay natively, improving generalization for computer vision.
    optimizer = AdamW(learning_rate=0.001, weight_decay=1e-4)

    # Compile the model with 1.0 Loss Weights for both heads
    model.compile(
        optimizer=optimizer,
        loss={
            'tsb_regression_output': 'mse',
            'risk_classification_output': 'categorical_crossentropy' 
        },
        loss_weights={
            'tsb_regression_output': 1.0,
            'risk_classification_output': 1.0
        },
        metrics={
            'tsb_regression_output': ['mae'], 
            'risk_classification_output': ['accuracy']
        }
    )
    
    # Print architecture overview
    model.summary()
    
    
    # ==========================================
    # 5. Execute Training Loop (Box 5)
    # ==========================================
    print("\nStarting Training Execution...")
    EPOCHS = 50
    
    callbacks = get_callbacks(model_save_path)
    
    history = model.fit(
        train_generator,
        validation_data=val_generator,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1
    )
    
    
    # ==========================================
    # 6. Final Evaluation
    # ==========================================
    print("\n--- Training Complete ---")
    print("Evaluating Best Saved Model on Held-out Test Set (20%)...")
    
    # Load the best weights saved by the ModelCheckpoint
    best_model = tf.keras.models.load_model(model_save_path)
    
    test_results = best_model.evaluate(test_generator, verbose=1)
    
    # Map metrics dict directly for clarity
    test_metrics = dict(zip(best_model.metrics_names, test_results))
    print("\nFinal Test Metrics:")
    for metric_name, value in test_metrics.items():
        print(f"  {metric_name}: {value:.4f}")
        
    print(f"\nModel successfully saved to {model_save_path}")
    return history, test_metrics


if __name__ == "__main__":
    # Use actual data paths
    CLINICAL_CSV_PATH = '/Users/shubhbhateja/College/minor_project/NeoJaundice/chd_jaundice_published_2.csv'
    PROCESSED_IMAGES_DIR = '/Users/shubhbhateja/College/minor_project/images'
    MODEL_SAVE_PATH = 'best_multimodal_neojaundice_model.h5'

    print("Train Configuration Ready.")

    # Start training pipeline
    history, metrics = train_pipeline(
        clinical_csv_path=CLINICAL_CSV_PATH,
        processed_images_dir=PROCESSED_IMAGES_DIR,
        model_save_path=MODEL_SAVE_PATH
    )
