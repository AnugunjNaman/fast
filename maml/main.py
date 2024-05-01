import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # See parent dir modules

import torch
from utils import set_random_seeds, replace_model_params, get_device, clone_model
import matplotlib.pyplot as plt
from maml import maml
from tasks import sample_adimatask, ADIMATask
import numpy as np
import functorch
from config import LastConfig, FastConfig
from last_model import LastAST
from fast_model import FastAST
from torch.utils.data import DataLoader
from loader import AudioDataset
from config_maml import MAMLConfigADIMA
from pathlib import Path
from sklearn.metrics import f1_score
from tqdm import tqdm
import csv
import argparse
import warnings
warnings.filterwarnings('ignore', category=UserWarning)


parser = argparse.ArgumentParser(description='MAML training/testing script.')
# Add arguments
parser.add_argument('--model', type=str, default='LAST', help='LAST (Lipschitz augmented) or FAST model.',
                    choices=['LAST', 'FAST'])
parser.add_argument('--num_epochs', type=int, default=400, help='Number of epochs for training MAML.')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for everywhere that loads data - set low until it fits on VRAM.')
parser.add_argument('--n_tasks', type=int, default=3, help='Number of tasks to sample for each MAML training loop.')
parser.add_argument('--meta_lr', type=float, default=1e-4, help='"beta" - Learning rate for meta-learning training loop.')
parser.add_argument('--inner_lr', type=float, default=1e-4, help='"alpha" - Learning rate for "inner" gradient descent.')
parser.add_argument('--test_language', type=str, default='All', help='Language for testing.', 
                    choices=['All', 'Hindi', 'Bengali', 'Punjabi', 'Haryanvi', 'Kannada', 
                                    'Odia', 'Bhojpuri', 'Gujarati', 'Tamil', 'Malayalam'])
# Parse the arguments
args = parser.parse_args()


################################################################################

# User decisions
MODEL = args.model
MAML_CONFIG = MAMLConfigADIMA(num_epochs=args.num_epochs, batch_size=args.batch_size, n_tasks=args.n_tasks,
                              meta_lr=args.meta_lr, inner_lr=args.inner_lr)
# Automatic decisions
print("Model:", MODEL)
print("MAML Config:", MAML_CONFIG)
DEVICE = get_device()
set_random_seeds(MAML_CONFIG.random_seed)
MAML_CONFIG_STRING = (
    f'_seed{MAML_CONFIG.random_seed}'
    f'_beta{MAML_CONFIG.meta_lr}'
    f'_epochs{MAML_CONFIG.num_epochs}'
    f'_innerupdates{MAML_CONFIG.inner_training_steps}'
    f'_K{MAML_CONFIG.batch_size}'
    f'_alpha{MAML_CONFIG.inner_lr}'
    f'_tasksamples{MAML_CONFIG.n_tasks}'
)
MAML_CONFIG_PARAMS_SAVE_PATH = Path(
    f'saves/{MODEL}/META_weights'
    + MAML_CONFIG_STRING +
    f'.pth'
)   # NOTE: This will be really hard to edit later...


################################################################################

# Load model
if MODEL == 'LAST':
    model = LastAST(config=LastConfig())
elif MODEL == 'FAST':
    model = FastAST(config=FastConfig())
else:
    raise ValueError("Wrong MODEL")
model.to(DEVICE)
# print(model)

# Deconstruct torch model into functional part and parameters
# NOTE: I'm aware functorch is obsolete
# TODO: buffers are not quite working...
if MODEL == 'LAST':
    fmodel, meta_params = functorch.make_functional(model)
    buffers = None
elif MODEL == 'FAST':
    fmodel, meta_params, buffers = functorch.make_functional_with_buffers(model)
else:
    raise ValueError("Wrong MODEL")

# DON'T RUN MAML if config's saved parameters already exists!
if not MAML_CONFIG_PARAMS_SAVE_PATH.exists():
    # Run MAML, save results!
    meta_optimizer = torch.optim.Adam(meta_params, lr=MAML_CONFIG.meta_lr)
    training_loss = maml(meta_params, fmodel, meta_optimizer, 
                         MAML_CONFIG.num_epochs, 
                         MAML_CONFIG.inner_training_steps, 
                         MAML_CONFIG.batch_size, 
                         MAML_CONFIG.inner_lr, 
                         sample_adimatask, 
                         MAML_CONFIG.n_tasks,
                         device=DEVICE,
                         buffers=buffers)
    print(f"Finished MAML training ({MAML_CONFIG.num_epochs} epochs)!")
    # AH YUP buffers actually becomes modified here... need to save it?

    # Save weights!
    replace_model_params(model, meta_params)   # Stick them back in... NOTE: also need to stick buffers back in!!!
    torch.save(model.state_dict(), MAML_CONFIG_PARAMS_SAVE_PATH)
    print(f"MAML weights saved to '{MAML_CONFIG_PARAMS_SAVE_PATH}'")
    
    # Save figure: MAML training loss
    plt.title(f"Training Curve: ADIMA MAML (N=2-Way, K={MAML_CONFIG.batch_size}-Shot) on {MODEL}")
    plt.plot(training_loss, c='lightgreen', linestyle='--', 
             linewidth=2.2, label=f'MAML Training Loss ({MAML_CONFIG.num_epochs} epochs)')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(f'ADIMA_MAML_{MODEL}_trainingcurve'+MAML_CONFIG_STRING+'.png')
    print(f"MAML training curve saved as PNG!")
    # plt.show()

    # Save training loss numbers (over the epochs)
    with open(f'ADIMA_MAML_{MODEL}_trainingcurve'+MAML_CONFIG_STRING+'.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for tl in training_loss:
            writer.writerow([tl])   # Write each loss number as a row
    print(f"MAML training losses saved as CSV!")
else:
    # Just load the saved weights (parameters)
    checkpoint = torch.load(MAML_CONFIG_PARAMS_SAVE_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint)
    # model.to(device)    # Is this necessary?
    print(f"MAML weights found and loaded from '{MAML_CONFIG_PARAMS_SAVE_PATH}'")


################################################################################

# At this stage, model is primed by MAML and ready for K-shot testing!
# i.e. 0-shot would be testing immediately, 1-shot would be training on 1 batch then testing, etc.
# A reasonable K-shot might be K=4...

def test_model(model, dataloader, criterion, device, model_train=False):
    if model_train:
        model.train()
    else:
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

# User decisions
if args.test_language == 'All':
    if buffers is not None:
        # languages = ['Hindi', 'Bengali', 'Punjabi', 'Hindi']    # FAST buffers need warmup...
        languages = ['Hindi', 'Bengali', 'Punjabi', 'Haryanvi', 'Kannada', 
                     'Odia', 'Bhojpuri', 'Gujarati', 'Tamil', 'Malayalam']
    else:
        # languages = ['Hindi', 'Bengali', 'Punjabi']     # LAST doesn't
        languages = ['Hindi', 'Bengali', 'Punjabi', 'Haryanvi', 'Kannada', 
                     'Odia', 'Bhojpuri', 'Gujarati', 'Tamil', 'Malayalam']
        # languages = ['Bengali', 'Hindi']
else:
    languages = [args.test_language]
for language in languages:
    train_dataset = AudioDataset(f"../data/files/SC_audio_{language}/", 
                                 f"../data/annotations/{language}_train.csv", hop_size=500)
    test_dataset = AudioDataset(f"../data/files/SC_audio_{language}/", 
                                f"../data/annotations/{language}_test.csv", hop_size=500)
    train_dataloader = DataLoader(train_dataset, batch_size=MAML_CONFIG.batch_size, 
                                  shuffle=True, num_workers=1, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=MAML_CONFIG.batch_size, 
                                 shuffle=True, num_workers=1, pin_memory=True)
    # K_features, K_targets = next(iter(dataloader))
    criterion = torch.nn.CrossEntropyLoss()

    print(f"\n{language}: Running 0-shot evaluations...")
    train_accuracy, train_loss, train_f1 = test_model(model, train_dataloader, criterion, DEVICE, buffers is not None)
    print(f"Training Accuracy: {train_accuracy}")
    print(f"Training Loss: {train_loss}")
    print(f"Training F1: {train_f1}")
    test_accuracy, test_loss, test_f1 = test_model(model, test_dataloader, criterion, DEVICE, buffers is not None)
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Loss: {test_loss}")
    print(f"Test F1: {test_f1}")

####

# K-Shot Testing: 1 epoch of 4 positive, 4 negative (4-shot) training on a particular language, then go again?
def train_epoch(model, dataloader, optimizer, criterion, device):
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

for language in languages:
    set_random_seeds(MAML_CONFIG.random_seed)   # NOTE: Maintain reproducibility
    atask = ADIMATask(language)
    shot_features, shot_targets, shot_criterion = atask.sample(args.batch_size, test_set=False)     # NOTE: This is affected by randomness
    cloned_model = clone_model(model)
    shot_optimizer = torch.optim.SGD(cloned_model.parameters(), lr=args.inner_lr)
    shot_avg_loss = train_epoch(cloned_model, [(shot_features, shot_targets)], shot_optimizer, shot_criterion, DEVICE)
    print(f"\n{language}: Running 4-shot evaluations...")
    shot_train_accuracy, shot_train_loss, shot_train_f1 = test_model(cloned_model, train_dataloader, criterion, DEVICE, buffers is not None)
    print(f"Training Accuracy: {shot_train_accuracy}")
    print(f"Training Loss: {shot_train_loss}")
    print(f"Training F1: {shot_train_f1}")
    shot_test_accuracy, shot_test_loss, shot_test_f1 = test_model(cloned_model, test_dataloader, criterion, DEVICE, buffers is not None)
    print(f"Test Accuracy: {shot_test_accuracy}")
    print(f"Test Loss: {shot_test_loss}")
    print(f"Test F1: {shot_test_f1}")
