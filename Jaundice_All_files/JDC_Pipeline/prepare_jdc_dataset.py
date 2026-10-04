import os
import shutil
import pandas as pd
import math

EXCEL_PATH = '/Users/shubhbhateja/Downloads/Official_JDC/Updated_JDC_Dataset.xlsx'
JDC_IMAGES_BASE = '/Users/shubhbhateja/Downloads/Official_JDC/JDC-1,2&3 datasets updated file names'

DEST_BASE = '/Users/shubhbhateja/College/minor_project/JDC_Pipeline/jdc_dataset'
DEST_IMAGES = os.path.join(DEST_BASE, 'images')
DEST_CSV = os.path.join(DEST_BASE, 'chd_jaundice_jdc.csv')

os.makedirs(DEST_IMAGES, exist_ok=True)

df = pd.read_excel(EXCEL_PATH)

image_cols = [
    'Forehead-0', 'Forehead-1', 'Sternum-0', 'Sternum-1', 'Chest-0', 'Chest-1', 
    'Abdomen-0', 'Abdomen-1', 'Arms-0', 'Arms-1', 'Palm-0', 'Palm-1', 
    'Thigh-0', 'Thigh-1', 'Legs-0', 'Legs-1', 'Sole-0', 'Sole-1'
]

csv_data = []
not_found_count = 0
found_count = 0

for index, row in df.iterrows():
    hospital_id = row['Hospital ID']
    uuid = row['UUID']
    weight = row['Weight (g)']
    treatment_raw = row['Treatment']
    treatment = 1 if str(treatment_raw).lower() == 'phototherapy' else 0
    
    bilirubin_raw = row['Bilirubin Level (mg/dL)']
    try:
        blood = float(bilirubin_raw)
    except:
        blood = float('nan')
        
    gender_raw = row['Gender']
    gender = 'M' if str(gender_raw).lower() == 'male' else 'F'
    
    age_raw = row['Postnatal Age Blood Sample(days)']
    try:
        age_str = str(age_raw).split(' ')[0]
        age = float(age_str)
    except:
        age = float('nan')
        
    # Check each image col
    for col in image_cols:
        img_rel_path = row[col]
        if pd.isna(img_rel_path) or img_rel_path is None:
            continue
            
        img_rel_path = str(img_rel_path)
        
        # e.g. ./Dataset/JDC_1/202404914632_366ef1d4.../filename.jpeg
        parts = img_rel_path.split('/')
        if len(parts) >= 4:
            jdc_dir = parts[2].replace('_', '-') # JDC_1 -> JDC-1
            neonate_dir = parts[3]
            filename = parts[-1]
            
            src_path = os.path.join(JDC_IMAGES_BASE, jdc_dir, neonate_dir, filename)
            
            if os.path.exists(src_path):
                dest_path = os.path.join(DEST_IMAGES, filename)
                # Copy file if it doesn't exist
                if not os.path.exists(dest_path):
                    shutil.copy2(src_path, dest_path)
                
                csv_data.append({
                    'patient_id': hospital_id,
                    'image_idx': filename,
                    'gender': gender,
                    'age(day)': age,
                    'weight': weight,
                    'blood(mg/dL)': blood,
                    'Treatment': treatment
                })
                found_count += 1
            else:
                not_found_count += 1
                # print(f"Not found: {src_path}")

out_df = pd.DataFrame(csv_data)
out_df.to_csv(DEST_CSV, index=False)

print(f"Total images found and copied: {found_count}")
print(f"Total image paths in excel but not found on disk: {not_found_count}")
print(f"Saved CSV to {DEST_CSV}")
