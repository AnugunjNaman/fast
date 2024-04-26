import torch
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import copy

def set_random_seeds(random_seed=0):
    random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

def get_device(cuda_number=None, verbose=True):
    if torch.cuda.is_available():
        if verbose:
            print(f"CUDA ({torch.cuda.device_count()} device(s)) found! Using GPU...")
        if cuda_number:
            device = torch.device(f'cuda:{cuda_number}')    # Specified GPU
            if verbose:
                print(f"Got GPU 'cuda:{cuda_number}'...")
        else:
            device = torch.device('cuda')   # Default CUDA device
    else:
        print("Warning: Using CPU...")
        device = torch.device('cpu')
    return device

def clone_params(params_list):
    # Get copy of the parameters that can be worked independently but which still treat the OG parameters as leaves
    return [param.clone() for param in params_list]

def clone_detach_params(params_list):
    # Get "deep" copy of the parameters that is completely independent, yet still tracks gradients
    # return [torch.tensor(param, requires_grad=True) for param in params_list]
    return [param.clone().detach() for param in params_list]

def clone_detach_regrad_params(params_list):
    # Get "deep" copy of the parameters that is completely independent, yet still tracks gradients
    # return [torch.tensor(param, requires_grad=True) for param in params_list]
    return [param.clone().detach().requires_grad_(True) for param in params_list]

def manual_zero_grad(params_list):
    for param in params_list:
        param.grad = None

def clone_params_retain_grad(params_list):
    # Get copy of the parameters that can be worked independently but which still treat the OG parameters as leaves BUT ALSO can be used with .backward()
    cloned_params_list = [param.clone() for param in params_list]
    for cloned_param in cloned_params_list:
        cloned_param.retain_grad()
    return cloned_params_list

def clone_model(model):
    # model2.load_state_dict(model1.state_dict())
    cloned = copy.deepcopy(model)
    return cloned

def replace_model_params(model, new_params):
    state_dict = model.state_dict()
    param_keys = [k for k in state_dict.keys() if "weight" in k or "bias" in k]     # This is so jank
    for key, new_value in zip(param_keys, new_params):
        state_dict[key] = new_value     # Overwrite old param; old buffers left intact
    model.load_state_dict(state_dict)

def save_training_curve(train_loss, train_acc, test_acc, 
                        model_name, model_shape, optim_name, 
                        lr, wd, n_epochs):
    """ Plot for each epoch: training loss, training accuracy, test accuracy """
    # x-axis epochs, y-axis left loss, y-axis right train accuracy/test accuracy
    plt.clf()
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()   # 2 y-axes share the x-axis
    x_axis = range(1, len(train_loss)+1)
    ax1.plot(x_axis, train_loss, '-', color='r', linewidth=2, label='Training Loss')
    ax2.plot(x_axis, train_acc, '--', color='C0', linewidth=2, label='Training Accuracy')
    ax2.plot(x_axis, test_acc, '-', color='C0', linewidth=2, label='Test Accuracy')
    ax1.set_xlabel("Epoch #")
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Accuracy")
    ax1.legend(loc='center left')
    ax2.legend(loc='center right')
    ax1.set_title(f"Training Curve: '{optim_name}' Optimizer; LR: {lr}, WD: {wd}")
    out_name = (f"training_curve_{optim_name}_{model_name}_{model_shape}"
                f"_LR{str(lr).replace('.', 'p')}_WD{str(wd).replace('.', 'p')}"
                f"_epochs{str(n_epochs)}.png")
    plt.savefig(out_name)

def save_learning_curve(train_sizes, train_accs, test_accs,
                        model_name, model_shape, optim_name,
                        lr, wd, n_epochs):
    """ Plot training set sizes vs. final accuracies """
    # Plot train and test accuracies
    plt.clf()
    fig, ax = plt.subplots()
    ax.plot(train_sizes, train_accs, '--', color='C0', linewidth=2, label='Training Accuracy')
    ax.plot(train_sizes, test_accs, '-', color='C0', linewidth=2, label='Test Accuracy')
    ax.set_xlabel("Training Set Size")
    ax.set_ylabel("Accuracy")
    ax.legend(loc='lower right')
    ax.set_title(f"Learning Curve: '{optim_name}' Optimizer; LR: {lr}, WD: {wd}")
    out_name = (f"learning_curve_{optim_name}_{model_name}_{model_shape}"
                f"_LR{str(lr).replace('.', 'p')}_WD{str(wd).replace('.', 'p')}"
                f"_epochs{str(n_epochs)}.png")
    plt.savefig(out_name)
