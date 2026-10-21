import os
import re
import glob
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from sklearn.model_selection import StratifiedGroupKFold
import albumentations as A
from albumentations.pytorch import ToTensorV2

def generate_patient_stratified_splits(data_dir: str, output_dir: str = "./data/processed", seed: int = 42):
    all_paths = glob.glob(os.path.join(data_dir, "**/*.jpeg"), recursive=True) + \
                glob.glob(os.path.join(data_dir, "**/*.png"), recursive=True)
    
    records = []
    for path in all_paths:
        label_str = "PNEUMONIA" if "PNEUMONIA" in path else "NORMAL"
        label = 1 if label_str == "PNEUMONIA" else 0
        filename = os.path.basename(path)
        
        patient_match = re.match(r"(person\d+|NORMAL2-IM-\d+|IM-\d+)", filename)
        patient_id = patient_match.group(1) if patient_match else filename.split('_')[0]
        
        records.append({"path": path, "label": label, "patient_id": patient_id})
    
    df = pd.DataFrame(records)
    
    gkf_outer = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    train_idx, holdout_idx = next(gkf_outer.split(df, df['label'], groups=df['patient_id']))
    
    train_df = df.iloc[train_idx].reset_index(drop=True)
    holdout_df = df.iloc[holdout_idx].reset_index(drop=True)
    
    gkf_inner = StratifiedGroupKFold(n_splits=2, shuffle=True, random_state=seed)
    val_idx, test_idx = next(gkf_inner.split(holdout_df, holdout_df['label'], groups=holdout_df['patient_id']))
    
    val_df = holdout_df.iloc[val_idx].reset_index(drop=True)
    test_df = holdout_df.iloc[test_idx].reset_index(drop=True)
    
    os.makedirs(output_dir, exist_ok=True)
    train_df.to_csv(os.path.join(output_dir, "train.csv"), index=False)
    val_df.to_csv(os.path.join(output_dir, "val.csv"), index=False)
    test_df.to_csv(os.path.join(output_dir, "test.csv"), index=False)
    
    return train_df, val_df, test_df

class ChestXRayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = np.array(Image.open(row['path']).convert("RGB"))
        label = torch.tensor(row['label'], dtype=torch.float32)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        return image, label

def build_transforms(img_size: int = 224):
    train_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=10, border_mode=0, p=0.5),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=0, border_mode=0, p=0.3),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    eval_transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

    return train_transform, eval_transform