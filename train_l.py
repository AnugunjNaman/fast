import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from loader import AudioDataset
from tqdm import tqdm
from fast_model import MobileViT as MVT

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

language = "Hindi"

num_epochs = 30
learning_rate = 0.001

batch_size = 16

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float()[:, :, :, :-1])
        loss = criterion(outputs.view(-1, 2), labels.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    average_loss = total_loss / len(dataloader)
    return average_loss

def test_model(model, dataloader, criterion):
    model.eval()
    correct_predictions = 0
    total_samples = 0
    total_loss = 0.0
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs.float()[:, :, :, :-1])
            loss = criterion(outputs.view(-1, 2), targets.view(-1))
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == targets).sum().item()
            total_samples += targets.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    accuracy = correct_predictions / total_samples
    average_loss = total_loss / len(dataloader)
    f1 = f1_score(all_targets, all_preds)
    return accuracy, average_loss, f1

def save_model_weights(model, language, ff = 0):
    torch.save(model.state_dict(), f'{language}_{ff}_ast_weights_epoch.pth') 

def load_weights(language, ff = 0):
    return torch.load(f'{language}_{ff}_ast_weights_epoch.pth', map_location=device)

model = MVT(image_size = (128, 1536), dims = [96, 120, 144], channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384], num_classes=2)
model = model.to(device)
print(model)

optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()
train_dataset = AudioDataset(f"data/files/SC_audio_{language}/", f"data/annotations/{language}_train.csv", hop_size=500)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)  

test_dataset = AudioDataset(f"data/files/SC_audio_{language}/", f"data/annotations/{language}_test.csv", hop_size=500)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

print("Batch Size: ", batch_size)

acc = 0.0
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
    test_accuracy, test_loss, test_f1 = test_model(model, test_dataloader, criterion)
    train_accuracy, train_loss, train_f1 = test_model(model, train_dataloader, criterion)
    if test_accuracy > acc:
        acc = test_accuracy
        save_model_weights(model, language, ff=1)
    print(f'Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
    print(f'Epoch {epoch+1}/{num_epochs} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}')

del model
checkpoint = load_weights(language, ff=1)
model = MVT(image_size = (128, 1536), dims = [96, 120, 144], channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384], num_classes=2)
model.load_state_dict(checkpoint)
test_accuracy, test_loss, test_f1 = test_model(model, test_dataloader, criterion)
print("Final Acc/F1: ", test_accuracy, test_f1)
