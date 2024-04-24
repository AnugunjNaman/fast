import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms

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
