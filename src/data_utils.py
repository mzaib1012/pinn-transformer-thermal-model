import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class TransformerDataset:
    def __init__(self, csv_path):
        # 1. Load the tabular data
        self.df = pd.read_csv(csv_path)
        
        # 2. Extract inputs (Time, Height, Ambient Temp, Load Current) and Targets (Winding Temp)
        self.X_raw = self.df[["time", "height", "ambient_temp", "load_current"]].values
        self.y_raw = self.df[["winding_temp"]].values
        
        # 3. Scale features for stable neural network training
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.X_scaled = self.scaler_X.fit_transform(self.X_raw)
        self.y_scaled = self.scaler_y.fit_transform(self.y_raw)
        
    def get_torch_data(self):
        # Convert scaled data into PyTorch float tensors
        X_tensor = torch.tensor(self.X_scaled, dtype=torch.float32)
        y_tensor = torch.tensor(self.y_scaled, dtype=torch.float32)
        return X_tensor, y_tensor

    def denormalize_predictions(self, pred_scaled):
        # Helper to convert model outputs back to actual Celsius degrees
        if isinstance(pred_scaled, torch.Tensor):
            pred_scaled = pred_scaled.detach().cpu().numpy()
        return self.scaler_y.inverse_transform(pred_scaled)
