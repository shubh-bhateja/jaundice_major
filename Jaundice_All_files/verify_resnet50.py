import os
import numpy as np
import matplotlib.pyplot as plt
from resnet50_classification import prepare_image_2d, build_resnet50_model, CSV_PATH, IMAGE_DIR
import pandas as pd

def verify_pipeline():
    print("Verifying Multimodal ResNet-50 Classification Pipeline...")
    
    # 1. Load CSV
    if not os.path.exists(CSV_PATH):
        print(f"Error: CSV not found at {CSV_PATH}")
        return
    
    df = pd.read_csv(CSV_PATH)
    sample_row = df.iloc[10] # Sample with some contrast
    sample_path = os.path.join(IMAGE_DIR, sample_row['image_idx'])
    
    # 2. Test Preprocessing
    print(f"Testing preprocessing on: {sample_path}")
    rgb_img, hsv_img = prepare_image_2d(sample_path)
    
    if rgb_img is not None and hsv_img is not None:
        print(f"Success: RGB shape = {rgb_img.shape}, HSV shape = {hsv_img.shape}")
        
        # Visualize
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(rgb_img)
        axes[0].set_title("Input RGB Skin Patch")
        axes[1].imshow(hsv_img)
        axes[1].set_title("Input HSV Skin Patch")
        
        save_path = "verification_plot.png"
        plt.savefig(save_path)
        print(f"Visualization saved to {save_path}")
        
    else:
        print("Error: Preprocessing failed to return images.")
        return

    # 3. Test Model Initialization
    print("Testing model initialization...")
    model = build_resnet50_model()
    print("Multimodal Model built successfully.")
    
    # 4. Dummy Inference
    print("Testing dummy inference...")
    dummy_rgb = np.expand_dims(rgb_img, axis=0)
    dummy_hsv = np.expand_dims(hsv_img, axis=0)
    
    prediction = model.predict({
        'rgb_input': dummy_rgb,
        'hsv_input': dummy_hsv
    })
    print(f"Dummy prediction: {prediction}")
    
    print("\nMULTIMODAL PIPELINE VERIFIED SUCCESSFULLY.")

if __name__ == "__main__":
    verify_pipeline()
