import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from ast_model import ASTClassification
from config import ASTConfig
from loader import AudioDataset

def train_epoch(model, dataloader, optimizer):
    model.train()
    total_loss = 0.0

    for inputs, labels in dataloader:
        optimizer.zero_grad()
        outputs = model(outputs, labels)
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    return average_loss

def test_model(model, dataloader):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs, targets)
            total_loss += outputs[0].item()

            _, predicted = torch.max(outputs[1], 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = correct_predictions / total_samples
    average_loss = total_loss / len(dataloader)
    f1 = f1_score(all_targets, all_preds)
    return accuracy, average_loss, f1

def save_model_weights(model, epoch, language, ff = 0):
    torch.save(model.state_dict(), f'{language}_{ff}_ast_weights_epoch_{epoch}.pth') 

num_epochs = 10
learning_rate = 0.001

language = "Hindi"

config = ASTConfig()
model = ASTClassification(config)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

train_dataset = AudioDataset(f"data/files/SC_audio_{language}/", f"data/annotations/{language}_train.csv")
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)  

test_dataset = AudioDataset(f"data/files/SC_audio_{language}/", f"data/annotations/{language}_test.csv")
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=True)  


for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer)
    test_accuracy, test_loss, test_f1 = test_model(model, test_dataloader)
    train_accuracy, train_loss, train_f1 = test_model(model, train_dataloader)
    print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, ' +  
          f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}'+
          f'Train Accuracy: {test_accuracy:.4f}, Train F1: {test_f1:.4f}')
    
    
    save_model_weights(model, epoch, language, ff=0)
