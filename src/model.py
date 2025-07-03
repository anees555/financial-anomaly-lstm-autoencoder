import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from src.model import LSTMAutoencoder

def train_model(X_train, num_epochs=20, batch_size=64, learning_rate=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Prepare dataset
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_train_tensor), batch_size=batch_size, shuffle=True)
    
    # Initialize model
    model = LSTMAutoencoder(n_features=X_train.shape[2]).to(device)
    criterion = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch in train_loader:
            seqs = batch[0].to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = criterion(outputs, seqs)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")
        
    return model
