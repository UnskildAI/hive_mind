import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import argparse
import random
from services.task.model import TaskModel
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("task_train")

class MockTaskDataset(Dataset):
    def __init__(self, length=100, perception_dim=1024, text_dim=128):
        self.length = length
        self.perception_dim = perception_dim
        self.text_dim = text_dim

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # Return random perception, random text embedding, target
        p = torch.randn(self.perception_dim)
        t = torch.randn(self.text_dim) # Mock text embedding pre-computed
        target = torch.randn(256) # output dim
        return p, t, target

def train(epochs=1, batch_size=4, save_path="task_model.pth"):
    logger.info(f"Starting training for {epochs} epochs...")
    
    task_service = TaskModel()
    model = task_service.model
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.MSELoss()
    
    dataset = MockTaskDataset(perception_dim=task_service.perception_dim, text_dim=task_service.text_dim)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(epochs):
        total_loss = 0
        for i, (p, t, targets) in enumerate(dataloader):
            p = p.to(task_service.device)
            t = t.to(task_service.device)
            targets = targets.to(task_service.device)
            
            optimizer.zero_grad()
            # The model forward expects perception_flat, text_emb
            outputs = model(p, t)
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
