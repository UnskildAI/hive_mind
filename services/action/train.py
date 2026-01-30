import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from services.action.policy import ActionExpert
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("action_train")

class MockActionDataset(Dataset):
    def __init__(self, length=100, input_dim=1295, action_dim=7, horizon=10):
        self.length = length
        self.input_dim = input_dim
        self.output_dim = action_dim * horizon

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return flattened input and flattened action labels
        return torch.randn(self.input_dim), torch.randn(self.output_dim)

def train(epochs=1, batch_size=32, save_path="action_policy.pth"):
    logger.info(f"Starting training for {epochs} epochs...")
    
    expert = ActionExpert()
    model = expert.model
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.MSELoss()
    
    # input_dim from expert instance
    dataset = MockActionDataset(input_dim=expert.input_dim, action_dim=expert.action_dim, horizon=expert.horizon)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for i, (inputs, targets) in enumerate(dataloader):
            inputs = inputs.to(expert.device)
            targets = targets.to(expert.device)
            
            optimizer.zero_grad()
            outputs = model.net(inputs) # Calling net directly for training on flat inputs
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        logger.info(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(dataloader):.4f}")
        
    torch.save(model.state_dict(), save_path)
    logger.info(f"Model saved to {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()
    
    train(args.epochs, args.batch_size)
