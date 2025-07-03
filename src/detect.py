# RECONSTRUCTION ERROR & ANOMALY DETECTION
import torch
import numpy as np

def compute_reconstruction_errors(model, X_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    errors = []
    with torch.no_grad():
        for seq in X_data:
            seq_tensor = torch.tensor(seq[np.newaxis, :, :], dtype=torch.float32).to(device)
            reconstructed = model(seq_tensor)
            mse = torch.mean((reconstructed - seq_tensor) ** 2).item()
            errors.append(mse)
    return np.array(errors)
