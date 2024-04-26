import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # See parent dir modules

from models import MLP, mlp
import torch
from utils import set_random_seeds, clone_detach_params, replace_model_params, get_device
import matplotlib.pyplot as plt
from maml import perform_k_training_steps, maml, accuracy_on_mtask
from tasks import sample_task, sample_mnisttask, sample_adimatask
import numpy as np
import functorch
from config import LastConfig, FastConfig
from last_model import LastAST
from fast_model import FastAST
# from train_last import save_model_weights, train_epoch, test_model, load_weights
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

def main_regression():
    set_random_seeds(42)
    if torch.cuda.is_available():
        print("CUDA found!")
        device = torch.device('cuda')
    else:
        print("Warning: using CPU...")
        device = torch.device('cpu')

    params = [torch.rand(40, 1, device=device).uniform_(-np.sqrt(6. / 41), np.sqrt(6. / 41)).requires_grad_(),
            torch.zeros(40, device=device).requires_grad_(),
            torch.rand(40, 40, device=device).uniform_(-np.sqrt(6. / 80), np.sqrt(6. / 80)).requires_grad_(),
            torch.zeros(40, device=device).requires_grad_(),
            torch.rand(1, 40, device=device).uniform_(-np.sqrt(6. / 41), np.sqrt(6. / 41)).requires_grad_(),
            torch.zeros(1, device=device).requires_grad_()]
    set_random_seeds(42)

    meta_optimizer = torch.optim.Adam(params, lr=1e-3)
    training_loss = maml(params, mlp, meta_optimizer, 1, 5_000, 10, 1e-3, sample_task, device=device, nb_tasks=10)

    plt.title('MAML, K=10')
    x = torch.linspace(-5, 5, 50).to(device)
    y = mlp(params, x[..., None])
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='lightgreen', linestyle='--', linewidth=2.2,
                label='pre-update')
    # New task
    task = sample_task()    # You gotta be joking - this is system dependent
    ground_truth_y = task.a * torch.sin(x + task.b)
    plt.plot(x.data.cpu().numpy(), ground_truth_y.data.cpu().numpy(), c='red', label='ground truth')
    # Fine-tuning, 10 gradient steps
    new_params = [p.clone() for p in params]
    perform_k_training_steps(new_params, mlp, task, 10, 10, 1e-3, device=device)
    # After 10 gradient steps
    y = mlp(new_params, x[..., None])
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='darkgreen', linestyle='--', linewidth=2.2,
                label='10 grad step')
    plt.legend()
    plt.xlim(-5, 5)
    plt.ylim(-5, 5)
    plt.savefig('code100_5K_seed42.png')
    plt.show()

################################################################################

def main_mnist():
    set_random_seeds(42)
    if torch.cuda.is_available():
        print("CUDA found!")
        device = torch.device('cuda')
    else:
        print("Warning: using CPU...")
        device = torch.device('cpu')

    mnist_model = MLP(784, 10, [300, 100], criterion_type='CrossEntropyLoss')
    mnist_model.to(device)
    fmodel, params = functorch.make_functional(mnist_model)
    og_params = clone_detach_params(params)     # LMAO these achieve 4%
    print(og_params[-1])
    set_random_seeds(42)

    meta_optimizer = torch.optim.Adam(params, lr=1e-3)
    N_EPOCHS = 100
    training_loss = maml(params, fmodel, meta_optimizer, 1, N_EPOCHS, 10, 1e-3, sample_mnisttask, device=device, nb_tasks=10)
    print(f"Finished MAML training {N_EPOCHS}!")
    plt.title('MNIST MAML Test: N=2-Way, K=10-Shot')
    plt.plot(training_loss, c='lightgreen', linestyle='--', linewidth=2.2, label=f'MAML Training Loss ({N_EPOCHS} epochs)')
    plt.legend()
    plt.savefig(f'mnist_trainingloss_epochs{N_EPOCHS}_seed42.png')
    plt.show()

    # TODO: How to show MAML-ready state?
    # 0) Choose a task.
    # 1) Try pre-update on the chosen task - percent accuracy.
    # 2) Then first contrast is params trained on the chosen task - percent accuracy as we train more epochs.
    # 3) Then second contrast is params trained from scratch on chosen task - percent accuracy as we train epochs.
    #   - Hopefully this takes way more epochs to reach the accuracy.
    # 4) Plot the 3 sets of accuracies - 1) flat line, 2) rise, 3) rise from nothing

    # 0) New task
    mtask = sample_mnisttask()
    # 1) Pre-update MAML-ready model accuracy evaluation
    maml_params = clone_detach_params(params)   # LMAO these somehow achieve 0%, like literally 0
    print(maml_params[-1])
    pre_acc_train, pre_acc_test = accuracy_on_mtask(fmodel, params, mtask, device)
    print(f"\nMAML-Ready Model: Accuracy (Training Set): {pre_acc_train}, Accuracy (Test Set): {pre_acc_test}\n")
    features = mtask.ab_features.to(device)
    predicted_logits = fmodel(params, features)
    predicted_ab_targets = predicted_logits.argmax(dim=1)   # Summarizes each row to index of biggest value
    acc_train = (predicted_ab_targets == mtask.ab_targets.to(device)).float().mean().item()
    print(features)
    print(predicted_logits)
    print(predicted_ab_targets)
    print(predicted_ab_targets.unique())    # LMAO only predicts 7 - the last thing it optimized for was 7 apparently
    print(mtask.ab_targets)
    print(mtask.ab_targets.unique())    # Only 0 and 1, etc.; I messed up framing these tasks
    print(acc_train)    # Hence 0% accuracy
    pre_acc_train, pre_acc_test = accuracy_on_mtask(fmodel, og_params, mtask, device)
    print(f"\nMAML-Ready Model: Accuracy (Training Set): {pre_acc_train}, Accuracy (Test Set): {pre_acc_test}\n")
    features = mtask.ab_features.to(device)
    predicted_logits = fmodel(og_params, features)
    predicted_ab_targets = predicted_logits.argmax(dim=1)   # Summarizes each row to index of biggest value
    acc_train = (predicted_ab_targets == mtask.ab_targets.to(device)).float().mean().item()
    print(features)
    print(predicted_logits)
    print(predicted_ab_targets)
    print(predicted_ab_targets.unique())
    print(mtask.ab_targets)
    print(mtask.ab_targets.unique())
    print(acc_train)
    pre_acc_train, pre_acc_test = accuracy_on_mtask(fmodel, maml_params, mtask, device)
    print(f"\nMAML-Ready Model: Accuracy (Training Set): {pre_acc_train}, Accuracy (Test Set): {pre_acc_test}\n")
    # 2) MAML-ready params trained on chosen task
    new_params = [p.clone() for p in params]
    loss_list, acc_train_list, acc_test_list = \
        perform_k_training_steps(new_params, fmodel, mtask, 10, 50, 1e-3, device=device, verbose=True)
    print(new_params[-1])
    # 3) Scratch params trained on chosen task
    scratch_mnist_model = MLP(784, 10, [300, 100], criterion_type='CrossEntropyLoss')
    scratch_mnist_model.to(device)
    scratch_fmodel, scratch_params = functorch.make_functional(scratch_mnist_model)
    scratch_loss_list, scratch_acc_train_list, scratch_acc_test_list = \
        perform_k_training_steps(scratch_params, scratch_fmodel, mtask, 10, 50, 1e-3, device=device, verbose=True)

    # 4) Plot
    plt.title('MNIST MAML Test: N=2-Way, K=10-Shot')
    # 1)
    plt.hlines(pre_acc_train, 0, 50, colors='lightgreen', linestyle='--', linewidth=2.2, label='pre-update train acc')
    plt.hlines(pre_acc_test, 0, 50, colors='green', linestyle='--', linewidth=2.2, label='pre-update test acc')
    # 2)
    plt.plot(acc_train_list, c='cyan', linestyle='-', linewidth=2.2, label='MAML train 50 acc')
    plt.plot(acc_test_list, c='blue', linestyle='-', linewidth=2.2, label='MAML test 50 acc')
    # 3)
    plt.plot(scratch_acc_train_list, c='pink', linestyle='-', linewidth=2.2, label='Scratch train 50 acc')
    plt.plot(scratch_acc_test_list, c='red', linestyle='-', linewidth=2.2, label='Scratch test 50 acc')
    plt.legend()
    # plt.xlim(-5, 5)
    # plt.ylim(-5, 5)
    plt.savefig('mnist_accuracy_seed42.png')
    plt.show()

################################################################################

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
                    choices=['All', 'Hindi', 'Bengali', 'Punjabi'])
# Parse the arguments
args = parser.parse_args()

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

####

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

####

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
        languages = ['Hindi', 'Bengali', 'Punjabi', 'Hindi']    # FAST buffers need warmup...
    else:
        languages = ['Hindi', 'Bengali', 'Punjabi']     # LAST doesn't
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

# TODO: 4-Shot training on each language? Then go again?


# TODO: How to show MAML-ready state?
# 0) Choose a task.
# 1) Try pre-update on the chosen task - percent accuracy.
# 2) Then first contrast is params trained on the chosen task - percent accuracy as we train more epochs.
# 3) Then second contrast is params trained from scratch on chosen task - percent accuracy as we train epochs.
#   - Hopefully this takes way more epochs to reach the accuracy.
# 4) Plot the 3 sets of accuracies - 1) flat line, 2) rise, 3) rise from nothing

# del last_model
# checkpoint = torch.load(f'saves/last/META_biggest_3_ast_weights_epoch.pth', map_location=device)
# config = LastConfig()
# last_model = LastAST(config=config)
# last_model.load_state_dict(checkpoint)
# last_model.to(device)

# # 0) New task
# # atask = sample_adimatask()
# # 1) Pre-update MAML-ready model accuracy evaluation
# # maml_params = clone_detach_params(params)   # LMAO these somehow achieve 0%, like literally 0
# # print(params[-1])
# language = 'Punjabi'
# test_dataset = AudioDataset(f"../data/files/SC_audio_{language}/", 
#                             f"../data/annotations/{language}_test.csv", hop_size=500)
# # train_dataset = AudioDataset(f"../data/files/SC_audio_{language}/", 
# #                              f"../data/annotations/{language}_train.csv", hop_size=500)
# test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
# # train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=1, pin_memory=True)
# # K_features, K_targets = next(iter(dataloader))
# # test_model(last_model, dataloader, criterion)
# criterion = torch.nn.CrossEntropyLoss()

# from sklearn.metrics import f1_score
# from tqdm import tqdm
# def test_model(model, dataloader, criterion, device):
#     model.eval()
#     correct_predictions = 0
#     total_samples = 0
#     total_loss = 0.0
#     all_preds, all_targets = [], []
    
#     with torch.no_grad():
#         for inputs, targets in tqdm(dataloader):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs.float()[:, :, :, :-1])
#             loss = criterion(outputs.view(-1, 2), targets.view(-1))
#             total_loss += loss.item()

#             _, predicted = torch.max(outputs, 1)
#             correct_predictions += (predicted == targets).sum().item()
#             total_samples += targets.size(0)

#             all_preds.extend(predicted.cpu().numpy())
#             all_targets.extend(targets.cpu().numpy())
    
#     accuracy = correct_predictions / total_samples
#     average_loss = total_loss / len(dataloader)
#     f1 = f1_score(all_targets, all_preds)
#     return accuracy, average_loss, f1

# test_accuracy, test_loss, test_f1 = test_model(last_model, test_dataloader, criterion, device)
# print(f"Test Accuracy: {test_accuracy}")
# print(f"Test Loss: {test_loss}")
# print(f"Test F1: {test_f1}")

# for epoch in range(args.num_epochs):
#         train_loss = train_epoch(model, train_dataloader, optimizer, criterion)
#         test_accuracy, test_loss, test_f1 = test_model(model, test_dataloader, criterion)
#         train_accuracy, train_loss, train_f1 = test_model(model, train_dataloader, criterion)
#         if test_accuracy > acc:
#             acc = test_accuracy
#             save_model_weights(model, args.language, ff=1)
        
#         train_losses.append(train_loss)
#         test_losses.append(test_loss)
#         print(f'Epoch {epoch+1}/{args.num_epochs} Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Train F1: {train_f1:.4f}')
#         print(f'Epoch {epoch+1}/{args.num_epochs} Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}, Test F1: {test_f1:.4f}')

# pre_acc_train, pre_acc_test = accuracy_on_mtask(fmodel, params, atask, device)
# print(f"\nMAML-Ready Model: Accuracy (Training Set): {pre_acc_train}, Accuracy (Test Set): {pre_acc_test}\n")
# features = mtask.ab_features.to(device)
# predicted_logits = fmodel(params, features)
# predicted_ab_targets = predicted_logits.argmax(dim=1)   # Summarizes each row to index of biggest value
# acc_train = (predicted_ab_targets == mtask.ab_targets.to(device)).float().mean().item()
# print(features)
# print(predicted_logits)
# print(predicted_ab_targets)
# print(predicted_ab_targets.unique())    # LMAO only predicts 7 - the last thing it optimized for was 7 apparently
# print(mtask.ab_targets)
# print(mtask.ab_targets.unique())    # Only 0 and 1, etc.; I messed up framing these tasks
# print(acc_train)    # Hence 0% accuracy
# pre_acc_train, pre_acc_test = accuracy_on_mtask(fmodel, og_params, mtask, device)
# print(f"\nMAML-Ready Model: Accuracy (Training Set): {pre_acc_train}, Accuracy (Test Set): {pre_acc_test}\n")
# features = mtask.ab_features.to(device)
# predicted_logits = fmodel(og_params, features)
# predicted_ab_targets = predicted_logits.argmax(dim=1)   # Summarizes each row to index of biggest value
# acc_train = (predicted_ab_targets == mtask.ab_targets.to(device)).float().mean().item()
# print(features)
# print(predicted_logits)
# print(predicted_ab_targets)
# print(predicted_ab_targets.unique())
# print(mtask.ab_targets)
# print(mtask.ab_targets.unique())
# print(acc_train)
# pre_acc_train, pre_acc_test = accuracy_on_mtask(fmodel, maml_params, mtask, device)
# print(f"\nMAML-Ready Model: Accuracy (Training Set): {pre_acc_train}, Accuracy (Test Set): {pre_acc_test}\n")
# # 2) MAML-ready params trained on chosen task
# new_params = [p.clone() for p in params]
# loss_list, acc_train_list, acc_test_list = \
#     perform_k_training_steps(new_params, fmodel, mtask, 10, 50, 1e-3, device=device, verbose=True)
# print(new_params[-1])
# # 3) Scratch params trained on chosen task
# scratch_mnist_model = MLP(784, 10, [300, 100], criterion_type='CrossEntropyLoss')
# scratch_mnist_model.to(device)
# scratch_fmodel, scratch_params = functorch.make_functional(scratch_mnist_model)
# scratch_loss_list, scratch_acc_train_list, scratch_acc_test_list = \
#     perform_k_training_steps(scratch_params, scratch_fmodel, mtask, 10, 50, 1e-3, device=device, verbose=True)

# # 4) Plot
# plt.title('MNIST MAML Test: N=2-Way, K=10-Shot')
# # 1)
# plt.hlines(pre_acc_train, 0, 50, colors='lightgreen', linestyle='--', linewidth=2.2, label='pre-update train acc')
# plt.hlines(pre_acc_test, 0, 50, colors='green', linestyle='--', linewidth=2.2, label='pre-update test acc')
# # 2)
# plt.plot(acc_train_list, c='cyan', linestyle='-', linewidth=2.2, label='MAML train 50 acc')
# plt.plot(acc_test_list, c='blue', linestyle='-', linewidth=2.2, label='MAML test 50 acc')
# # 3)
# plt.plot(scratch_acc_train_list, c='pink', linestyle='-', linewidth=2.2, label='Scratch train 50 acc')
# plt.plot(scratch_acc_test_list, c='red', linestyle='-', linewidth=2.2, label='Scratch test 50 acc')
# plt.legend()
# # plt.xlim(-5, 5)
# # plt.ylim(-5, 5)
# plt.savefig('mnist_accuracy_seed42.png')
# plt.show()
