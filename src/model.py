# LSTM AUTOENCODER MODEL

import torch
import torch.nn as nn

class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features, embedding_dim=64):
        super(LSTMAutoencoder, self).__init__()
        self.n_features = n_features
        self.embedding_dim = embedding_dim
        
        self.encoder = nn.LSTM(
            input_size=n_features,
            hidden_size=embedding_dim,
            num_layers=1,
            batch_first=True
        )
        
        self.decoder = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=n_features,
            num_layers=1,
            batch_first=True
        )
        
    def forward(self, x):
        # x: [batch_size, seq_length, n_features]
        _, (hidden, _) = self.encoder(x)
        
        # Repeat hidden across sequence length
        repeated_hidden = hidden.repeat(x.size(1), 1, 1).permute(1,0,2)
        
        decoded_output, _ = self.decoder(repeated_hidden)
        return decoded_output

