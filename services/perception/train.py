import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
from services.perception.model import PerceptionModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("perception_train")

class MockDataset(Dataset):
    def __init__(self, length=100):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return random image and dummy label
        return torch.randn(3, 224, 224), torch.randn(16, 64) # match token output

def train(epochs=1, batch_size=4, save_path="perception_model.pth"):
    logger.info(f"Starting training for {epochs} epochs...")
    
    # Initialize model
    perception = PerceptionModel()
    model = perception.model
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    dataset = MockDataset()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for i, (images, targets) in enumerate(dataloader):
            images = images.to(perception.device)
            targets = targets.to(perception.device)
            
            optimizer.zero_grad()
            outputs = model(images)
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
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()
    
    train(args.epochs, args.batch_size)
