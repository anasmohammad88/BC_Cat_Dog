import torch
import torch.nn as nn 
import torch.optim as optim 
import torchmetrics as met
from torchvision import transforms
from cnn.model import CatDogClassification
from cnn.utility import load_data, save_model
from torch.utils.tensorboard import SummaryWriter

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CatDogClassification().to(device)
    writer = SummaryWriter(log_dir="logs/cat_dog")
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor()
    ])
    
    eval_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
        
    train_data = load_data("data/DogsVsCats/train", transform=train_transform, batch_size=128)
    eval_data = load_data("data/DogsVsCats/test", transform=eval_transform, batch_size=128)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(params=model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=10)
    
    train_accuracy = met.Accuracy(task='binary').to(device)
    eval_accuracy = met.Accuracy(task='binary').to(device)
    best_eval_accuracy = 0.0
    
    for epoch in range(50):
        model.train()
        train_loss = 0.0
        train_accuracy.reset()
        
        for image, label in train_data:
            image = image.to(device)
            label = label.to(device)
            optimizer.zero_grad()
            
            logits = model(image).squeeze(1)
            loss = criterion(logits, label.float())
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * image.size(0)
            train_accuracy.update(torch.sigmoid(logits), label.int())
            
        train_loss /= len(train_data.dataset)        
        train_accu = train_accuracy.compute().item()
        
        
        model.eval()
        eval_loss = 0.0
        eval_accuracy.reset()
        
        with torch.no_grad():
            for image, label in eval_data:
                image = image.to(device)
                label = label.to(device)
                
                logits = model(image).squeeze(1)
                loss = criterion(logits, label.float())
                
                eval_loss += loss.item()* image.size(0)
                eval_accuracy.update(torch.sigmoid(logits), label.int())
        
        eval_loss /= len(eval_data.dataset)
        eval_accu = eval_accuracy.compute().item()
        
        scheduler.step(eval_accu)
        
        writer.add_scalars("Loss", {"Train": train_loss,"Eval": eval_loss}, epoch)
        writer.add_scalars("Accuracy", {"Train": train_accu,"Eval": eval_accu}, epoch)
        
        print(f"Epoch: {epoch}, Train Loss: {train_loss:.4f}, Eval Loss: {eval_loss:.4f}, Train Accuracy: {train_accu:.4f}, Eval Accuracy: {eval_accu:.4f}, LR: {optimizer.param_groups[0]["lr"]:.6f}")
        
        if eval_accu > best_eval_accuracy:
            best_eval_accuracy = eval_accu
            save_model(model, "BC_cat_dog")
            
    writer.close()
    print(f"Best Evaluation Accuracy: {best_eval_accuracy:.4f}")

            
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    train(parser.parse_args())