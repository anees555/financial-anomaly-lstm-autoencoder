#  Unsupervised Anomaly Detection on Financial Transactions using LSTM Autoencoder (PyTorch)

I have implemented an **unsupervised anomaly detection system** on financial transaction data using an **LSTM Autoencoder** built in PyTorch. It reconstructs sequences of normal transactions and flags those with high reconstruction error as potential frauds.

---

##  Highlights
✅ Unsupervised LSTM Autoencoder on time series data  
✅ Trained only on normal transactions  
✅ Detects anomalies by reconstruction error  
✅ ROC, F1, precision-recall metrics  
✅ Jupyter notebooks for experiments  
✅ Clean modular code (PyTorch + Numpy + Matplotlib)

---

##  Project Workflow

---

## 🚀 How it works
1. **Data Preparation**
   - Load the `creditcard.csv` dataset.
   - Normalize features using `MinMaxScaler`.
   - Create sequences (windows of 10 transactions).

2. **Model Training**
   - LSTM Autoencoder trained **only on normal transactions**.
   - Learns to reconstruct typical transaction sequences.

3. **Anomaly Detection**
   - Compute reconstruction errors on all data.
   - Transactions with errors above 95th percentile flagged as anomalies.

4. **Evaluation & Visualization**
   - Confusion matrix, precision, recall, F1.
   - ROC curve and AUC.
   - Plots of reconstruction errors highlighting frauds.

---

## 📊 Example results

| Metric      | Value   |
|-------------|---------|
| Precision   | 0.92    |
| Recall      | 0.83    |
| F1 Score    | 0.87    |
| ROC AUC     | 0.97    |

*(Sample metrics from experiment run)*

---

## 🔍 Example plots

### Reconstruction Error with Fraud Locations
![Reconstruction Error Plot](outputs/plots/reconstruction_errors.png)

### ROC Curve
![ROC Curve](outputs/plots/roc_curve.png)

---

