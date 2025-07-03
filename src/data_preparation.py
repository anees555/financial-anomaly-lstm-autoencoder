import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def load_preprocess_data(data_path):
    data_path = "C:/Users/Lenovo/Desktop/financial-anomaly-lstm-autoencoder/data/raw/creditcard.csv"

    finance_data = pd.read_csv(data_path)

    # create Time_diff instead of Time
    finance_data['Time_diff'] = finance_data['Time'].diff().fillna(0)
    finance_data = finance_data.drop(columns=['Time'])

    # get the Time_diff in first column
    cols = ['Time_diff'] + [col for col in finance_data.columns if col not in ['Time_diff', 'Class']]
    finance_data = finance_data[cols + ['Class']]

    # Normalizing all the features except Class
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(finance_data.drop(columns=['Class']))
    labels = finance_data['Class'].values

    return scaler, data_scaled, labels

def create_sequences(data, labels, seq_length=10):
    sequences = []
    sequence_labels = []
    
    for i in range(len(data) - seq_length):
        sequences.append(data[i:i+seq_length])
        sequence_labels.append(labels[i+seq_length])
    
    return np.array(sequences), np.array(sequence_labels)


