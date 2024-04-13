import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from loader import AudioDataset
from tqdm import tqdm
from ast_model import ASTClassification
from config import ASTConfig
import argparse

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

parser = argparse.ArgumentParser(description='Audio classification training and testing script')

# Add arguments
parser.add_argument('--language', type=str, default='Hindi', help='Language for audio classification')
parser.add_argument('--num_epochs', type=int, default=30, help='Number of epochs for training')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for optimizer')
parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training and testing')
parser.add_argument('--gpu_id', type=int, default=1, help='Node number GPU if available')

# Parse the arguments
args = parser.parse_args()

# Display the arguments
print("Arguments:")
print("Language:", args.language)
print("Number of Epochs:", args.num_epochs)
print("Learning Rate:", args.learning_rate)
print("Batch Size:", args.batch_size)
print("GPU Enabled On:", args.gpu_id)


device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0.0

    for inputs, labels in tqdm(dataloader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs.float())
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
            outputs = model(inputs.float())
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
    torch.save(model.state_dict(), f'saves/ast/{language}_{ff}_ast_weights_epoch.pth') 

def load_weights(language, ff = 0):
    return torch.load(f'saves/ast/{language}_{ff}_ast_weights_epoch.pth', map_location=device)

def main():
    config = ASTConfig()
    print("Configs of Model: ")
    print(config)
    
    model = ASTClassification(config=config)
    model = model.to(device)
    print(model)

    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    train_dataset = AudioDataset(f"data/files/SC_audio_{args.language}/", f"data/annotations/{args.language}_train.csv")
    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)  

    test_dataset = AudioDataset(f"data/files/SC_audio_{args.language}/", f"data/annotations/{args.language}_test.csv")
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    acc = 0.0
    for epoch in range(args.num_epochs):
        train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
        test_accuracy, test_loss, test_f1 = test_model(model, test_dataloader, criterion)
        train_accuracy, train_loss, train_f1 = test_model(model, train_dataloader, criterion)
        if test_accuracy > acc:
            acc = test_accuracy
            save_model_weights(model, args.language, ff=1)
        print(f'Epoch {epoch+1}/{args.num_epochs} Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
        print(f'Epoch {epoch+1}/{args.num_epochs} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}')

    del model
    checkpoint = load_weights(args.language, ff=1)
    model = ASTClassification(config=config)
    model.load_state_dict(checkpoint)
    test_accuracy, test_loss, test_f1 = test_model(model, test_dataloader, criterion)
    print("Final Acc/F1: ", test_accuracy, test_f1)

if __name__=="__main__":
    main()
