           # Main script to train or detect anomalies

from src.data_preparation import load_preprocess_data, create_sequences
from src.train import train_model
from src.detect import compute_reconstruction_errors
import numpy as np
import matplotlib.pyplot as plt

# Load & preprocess
data_scaled, labels, scaler = load_preprocess_data("C:/Users/Lenovo/Desktop/financial-anomaly-lstm-autoencoder/data/raw/creditcard.csv")
X, y = create_sequences(data_scaled, labels, seq_length=10)

# Only train on normal transactions
X_train = X[y == 0]

# Train model
model = train_model(X_train, num_epochs=5)

# Compute reconstruction errors on all data
errors = compute_reconstruction_errors(model, X)

# Plot errors vs true frauds
plt.figure(figsize=(10,4))
plt.plot(errors, label="Reconstruction error")
plt.scatter(np.where(y==1), errors[y==1], color='r', label="Frauds")
plt.legend()
plt.show()

# Save to numpy arrays
np.save("outputs/errors.npy", errors)
np.save("outputs/sequence_labels.npy", y)
