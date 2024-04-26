import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # See parent dir modules

import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from loader import AudioDataset
from torch.utils.data import DataLoader

class Task:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def sample(self, K):
        x = torch.rand((K, 1)) * 10 - 5  # Sample x in [-5, 5]
        y = self.a * torch.sin(x + self.b)
        loss_fct = nn.MSELoss()
        return x, y, loss_fct

def sample_task():
    a = torch.rand(1).item() * 4.9 + .1  # Sample the amplitude in [0.1, 5.0]
    b = torch.rand(1).item() * np.pi  # Sample the phase in [0, pi]
    return Task(a, b)

################################################################################

# From CS587 HW1 Code Skeleton
def load_mnist_data(data_path='./data', split='train', n_training_examples=-1):
    assert (split in ['train', 'val'])
    if split == 'train':
        train_data = datasets.MNIST(data_path, train=True, download=True, transform=transforms.ToTensor())
        if n_training_examples == -1:
            features, targets = train_data.data, train_data.targets
        else:
            features, targets = train_data.data[:n_training_examples], train_data.targets[:n_training_examples]
    # MNIST does not have a validation set... so we use the test set.
    if split in ['val']:
        test_data = datasets.MNIST(data_path, train=False, download=True, transform=transforms.ToTensor())
        features, targets = test_data.data, test_data.targets
    return features.reshape(targets.shape[0], -1).float(), targets
# mnist_x_train, mnist_y_train = load_mnist_data()
# mnist_x_test, mnist_y_test = load_mnist_data(split='val')

# 2-way, 10-shot
# Classify between 0 and 1, 1 and 2, etc.
# For task of 0 vs. 1, receive 10 (feature, target) examples of 0, 10 examples of 1
class MNISTTask:
    def __init__(self, a, b):
        self.a = a
        self.b = b
        # The task will be to identify whether the feature is a or b
        mnist_x_train, mnist_y_train = load_mnist_data()
        a_idx = (mnist_y_train == a)
        b_idx = (mnist_y_train == b)
        self.a_features = mnist_x_train[a_idx]
        self.a_targets = mnist_y_train[a_idx]   # Tensor of all a
        self.b_features = mnist_x_train[b_idx]
        self.b_targets = mnist_y_train[b_idx]
        self.ab_features = torch.concat([self.a_features, self.b_features])
        self.ab_targets = torch.concat([self.a_targets, self.b_targets])
        mnist_x_test, mnist_y_test = load_mnist_data(split='val')
        # self.ab_features_test = torch.concat([mnist_x_test[mnist_y_test == a], mnist_x_test[mnist_y_test == b]])
        self.ab_features_test = mnist_x_test[(mnist_y_test == a).logical_or(mnist_y_test == b)]
        # self.ab_targets_test = torch.concat([mnist_y_test[mnist_y_test == a], mnist_y_test[mnist_y_test == b]])
        self.ab_targets_test = mnist_y_test[(mnist_y_test == a).logical_or(mnist_y_test == b)]
    def sample(self, K):
        random_idx_a = torch.randperm(self.a_features.shape[0])[:K]
        x_a = self.a_features[random_idx_a]
        y_a = self.a_targets[random_idx_a]  # Smaller Tensor of all a
        random_idx_b = torch.randperm(self.b_features.shape[0])[:K]
        x_b = self.b_features[random_idx_b]
        y_b = self.b_targets[random_idx_b]
        # Reminder: CrossEntropyLoss input should be UNNORMALIZED logits for each class (don't need pos or sum to 1)
        # Input size C for unbatched, (batch, C) for batched
        # Target size batch (yes, confusing!); each spot has a class INDEX [0, C) corresponding to whole row of size C
        loss_func = nn.CrossEntropyLoss()
        # Finally, mix in the features and targets
        features_ab = torch.concat([x_a, x_b])
        targets_ab = torch.concat([y_a, y_b])
        random_idx = torch.randperm(features_ab.shape[0])
        return features_ab[random_idx], targets_ab[random_idx], loss_func

def sample_mnisttask():
    a = np.random.randint(0, 10)    # Sample a, uniformly 0 through 9
    if a == 9:
        b = 0
    else:
        b = a + 1   # Select b as "next" number for simplicity; just 10 different tasks
    return MNISTTask(a, b)

################################################################################

# Need some way to randomly sample K feature-target pairs for each class from either training or test
class ADIMATask:
    def __init__(self, language) -> None:
        self.language = language
        self.test_dataset = AudioDataset(f"../data/files/SC_audio_{language}/", 
                                         f"../data/annotations/{language}_test.csv", hop_size=500)
        self.train_dataset = AudioDataset(f"../data/files/SC_audio_{language}/", 
                                          f"../data/annotations/{language}_train.csv", hop_size=500)
    def sample(self, K, test_set=False, verbose=False):
        if test_set:
            dataloader = DataLoader(self.test_dataset, batch_size=2*K, shuffle=True, num_workers=1, pin_memory=True)
        else:
            dataloader = DataLoader(self.train_dataset, batch_size=2*K, shuffle=True, num_workers=1, pin_memory=True)
        # NOTE: Ensure there's K of each class
        # K_features, K_targets = next(iter(dataloader))    # Sampling 2K works but doesn't ensure K pos and K neg - imbalance adds noise!
        positives_acc = 0   # Need this to get to K
        negatives_acc = 0   # Need this to get to K
        repeats = 0
        K_features_list = []
        K_targets_list = []
        while True:
            # Get batch
            K_features, K_targets = next(iter(dataloader))
            positives = K_targets.sum()
            negatives = 2*K - positives
            # See if we have enough positives and negatives now
            K_features_list.append(K_features)
            K_targets_list.append(K_targets)
            positives_acc += positives
            negatives_acc += negatives
            if positives_acc >= K and negatives_acc >= K:
                break
            repeats += 1
        if verbose:
            print(f"Repeated batch collections: {repeats}")
        repeated_features_concat = torch.concat(K_features_list)
        repeated_targets_concat = torch.concat(K_targets_list)
        pos_locs = (repeated_targets_concat == 1)
        K_features_pos = repeated_features_concat[pos_locs][:K]
        K_features_neg = repeated_features_concat[~pos_locs][:K]
        K_targets_pos = repeated_targets_concat[pos_locs][:K]
        K_targets_neg = repeated_targets_concat[~pos_locs][:K]
        K_features = torch.concat([K_features_pos, K_features_neg])
        K_targets = torch.concat([K_targets_pos, K_targets_neg])
        random_idx = torch.randperm(K_targets.shape[0])
        return K_features[random_idx], K_targets[random_idx], nn.CrossEntropyLoss()

# Return either Hindi, Bengali, or Punjabi task
def sample_adimatask():
    try:
        sample_adimatask.ORDER_GLOBAL   # Static variable
    except AttributeError:
        sample_adimatask.ORDER_GLOBAL = 0   # Start with Hindi
    lang_selector = sample_adimatask.ORDER_GLOBAL
    # lang_selector = np.random.randint(0, 3)
    if lang_selector == 0:
        language = 'Hindi'
    elif lang_selector == 1:
        language = 'Bengali'
    elif lang_selector == 2:
        language = 'Punjabi'
    else:
        raise ValueError("lang_selector shouldn't be anything else")
    sample_adimatask.ORDER_GLOBAL = (sample_adimatask.ORDER_GLOBAL+1)%3     # "Next" language in cycle on next sample
    return ADIMATask(language)
