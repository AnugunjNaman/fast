import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.autograd import grad
import copy


def mlp(x, params):
    h = torch.relu(torch.nn.functional.linear(x, params[0], bias=params[1]))
    h = torch.relu(torch.nn.functional.linear(h, params[2], bias=params[3]))
    return torch.nn.functional.linear(h, params[4], bias=params[5])


class MLP(nn.Module):
    def __init__(self, in_dims, out_dims, hidden_dims=None, 
                 dimension_progression=None) -> None:
        super().__init__()  # torch stuff
        if dimension_progression is None:
            dimension_progression = MLP.dim_prog(in_dims, out_dims, hidden_dims)
        assert len(dimension_progression) >= 2
        # There will always be an output layer
        output_layer = nn.Linear(dimension_progression[-2], out_dims, bias=True)
        # Create hidden layers (if any)
        # NOTE: If no hidden layers, this is linear regression and technically not an MLP
        exclude_out_dim = dimension_progression[:-1]    # Last element is [-2]
        hidden_layers = []
        if len(exclude_out_dim) >= 1:
            # There are hidden layers!
            for layer_in_dim, layer_out_dim in zip(exclude_out_dim[:-1], exclude_out_dim[1:]):
                hidden_layers.append(nn.Linear(layer_in_dim, layer_out_dim, bias=True))
                hidden_layers.append(nn.ReLU())     # Add activation with each hidden layer
        # Final combined sequence of layers, for use
        self.sequential = nn.Sequential(*hidden_layers, output_layer)
    
    def dim_prog(in_dims, out_dims, hidden_dims=None, features=None):
        if features is not None:
            in_dims = features.shape[-1]
        if hidden_dims is None:
            hidden_dims = []
        return [in_dims] + hidden_dims + [out_dims]
    
    def forward(self, input):
        # input: (batch, in_dims)
        return self.sequential(input)


def perform_k_training_steps(params, task, batch_size, inner_training_steps, alpha, device):
    for epoch in range(inner_training_steps):
        x_batch, y_batch, loss_fct = task.sample(batch_size)
        loss = loss_fct(mlp(x_batch.to(device), params), y_batch.to(device))

        for p in params:  # Zero grad
            p.grad = None
        gradients = grad(loss, params)
        for p, g in zip(params, gradients):  # Grad step
            p.data -= alpha * g


def perform_k_training_steps_model(model, task, batch_size, inner_training_steps, alpha, device):
    model.train()
    optimizer = nn.optim.Adam(model.parameters(), lr=alpha)
    for _ in range(inner_training_steps):
        optimizer.zero_grad()
        x_batch, y_batch, loss_fct = task.sample(batch_size)
        loss = loss_fct(model(x_batch.to(device)), y_batch.to(device))
        loss.backward()
        optimizer.step()


class Task:
    def __init__(self, a, b):
        self.a = a
        self.b = b
    def sample(self, K):
        x = torch.rand((K, 1)) * 10 - 5  # Sample x in [-5, 5]; ooooh this is for choosing x values of observation points; so like a batch of 10 observations at 10 different x of what the sin y is corresponding to the a and b of this task
        y = self.a * torch.sin(x + self.b)  # So the hidden information to learn is still a and b, i.e. the amplitude and phase
        loss_fct = nn.MSELoss()
        return x, y, loss_fct


from torchvision import datasets, transforms
def load_mnist_data(data_path='./data', split='train', n_training_examples=-1):
  assert (split in ['train', 'val'])
  if split == 'train':
    train_data = datasets.MNIST(
        data_path, train=True, download=True, transform=transforms.ToTensor())
    if n_training_examples == -1:
      features, targets = train_data.data, train_data.targets
    else:
      features, targets = train_data.data[:n_training_examples], train_data.targets[:n_training_examples]
  # MNIST does not have a validation set... so we use the test set.
  if split in ['val']:
    test_data = datasets.MNIST(
        data_path, train=False, download=True, transform=transforms.ToTensor())
    features, targets = test_data.data, test_data.targets
  return features.reshape(targets.shape[0], -1).float(), targets
# mnist_x_train, mnist_y_train = load_mnist_data()
# mnist_x_test, mnist_y_test = load_mnist_data(split='val')
class MNISTTask:
    def __init__(self, a, b):
        # The task will be to identify whether the number is 1
        mnist_x_train, mnist_y_train = load_mnist_data()
        a_idx = (mnist_y_train == a).logical_or(mnist_y_train == b)
        self.a_or_b_features = mnist_x_train[a_idx]
        self.a_or_b_targets = mnist_y_train[a_idx]
    def sample(self, K):
        get_idxs = torch.randperm(self.a_or_b_features.shape[0])[:K]
        x = self.a_or_b_features[get_idxs]
        y = self.a_or_b_targets[get_idxs]
        # x = torch.rand((K, 1)) * 10 - 5  # Sample x in [-5, 5]; ooooh this is for choosing x values of observation points; so like a batch of 10 observations at 10 different x of what the sin y is corresponding to the a and b of this task
        # y = self.a * torch.sin(x + self.b)  # So the hidden information to learn is still a and b, i.e. the amplitude and phase
        loss_fct = nn.MSELoss()
        return x, y, loss_fct


def create_task_MNIST():
    a = np.random.randint(0, 10)    # Sample the first possibility
    if a == 9:
        b = 0
    else:
        b = a + 1
    return MNISTTask(a, b)

# @torch.no_grad()
def create_task():
    a = torch.rand(1).item() * 4.9 + .1  # Sample the amplitude in [0.1, 5.0]
    b = torch.rand(1).item() * np.pi  # Sample the phase in [0, pi]
    return Task(a, b)


def clone_params(params_list):
    # Get copy of the parameters that can be worked independently but which still treat the OG parameters as leaves
    return [param.clone() for param in params_list]

def clone_params_retain_grad(params_list):
    # Get copy of the parameters that can be worked independently but which still treat the OG parameters as leaves BUT ALSO can be used with .backward()
    cloned_params_list = [param.clone() for param in params_list]
    for cloned_param in cloned_params_list:
        cloned_param.retain_grad()
    return cloned_params_list

def clone_detach_regrad_params(params_list):
    # Get "deep" copy of the parameters that is completely independent, yet still tracks gradients
    # return [torch.tensor(param, requires_grad=True) for param in params_list]
    return [param.clone().detach().requires_grad_(True) for param in params_list]

def clone_model(model):
    # model2.load_state_dict(model1.state_dict())
    cloned = copy.deepcopy(model)
    return cloned

def manual_zero_grad(params_list):
    for param in params_list:
        param.grad = None


################################################################################

def perform_MAML_on_thetas(thetas, alpha, beta, K_inner_train, K_outer_test, n_epochs, my_device):
    # thetas is the set of initialized parameters we'll be working on   # NOTE: Algo 2 Line 1
    training_loss_dict = {}     # Collect training loss for convenience of visuals later
    for i_epoch in tqdm(range(n_epochs)):   # NOTE: Algo 2 Line 2

        # Sample batch of tasks (from "distribution of tasks")
        tasks = [create_task() for _ in range(10)]  # NOTE: Algo 2 Line 3

        # Collect for each of the tasks:
        # 1) A modified set of parameters, trained just one step toward that task
        # 2) An additionally sampled test set for use in the meta-training
        thetas_prime = {}   # e.g. 10 of these, indexed by i for each task
        D_prime = {}
        for i, task in enumerate(tasks):    # NOTE: Algo 2 Line 4
            
            # Inner, per-task training
            task_thetas = clone_params(thetas)  # Don't touch original params
            for _ in range(1):
                task_x_train, task_y_train, loss_fct = task.sample(K_inner_train)   # NOTE: Algo 2 Line 5
                task_loss = loss_fct(
                    mlp(task_x_train.to(my_device), task_thetas),
                    task_y_train.to(my_device)
                )   # NOTE: Algo 2 Line 6
                for task_theta in task_thetas:  # Zero grad
                    task_theta.grad = None
                # assert task_thetas[0].grad is None  # Sanity check
                # Gradient descent training step for each theta (parameter)
                # Do "manually" to avoid automatically propagating .grad to OG leaf
                task_gradients = grad(task_loss, task_thetas)   # NOTE: Algo 2 Line 6
                for task_theta, task_gradient in zip(task_thetas, task_gradients):  # Grad step
                    task_theta.data -= alpha*task_gradient  # NOTE: Algo 2 Line 7
            # task_thetas are now modified!
            # Collect parameters
            thetas_prime[i] = task_thetas

            # Collect additionally sampled "test" data points for the meta-update; NOTE: Algo 2 Line 8
            task_x_test, task_y_test, loss_fct = task.sample(K_outer_test)
            D_prime[i] = (task_x_test, task_y_test, loss_fct)
        
        # NOTE: Algo 2 Line 9

        # Meta-update step  # NOTE: Algo 2 Line 10
        meta_loss_dict = {}
        for i in range(10):
            task_x_test, task_y_test, loss_fct = D_prime[i]
            loss = loss_fct(
                mlp(task_x_test.to(my_device), thetas_prime[i]), 
                task_y_test.to(my_device)
            )   # The loss of 1-step trained parameters on the test data from that same task
            meta_loss_dict[i] = loss
        meta_loss = sum(meta_loss_dict.values())
        for theta in thetas:  # Zero grad
            theta.grad = None
        # assert thetas[0].grad is None  # Sanity check
        # Gradient descent training step for each theta (parameter)
        # Do "manually" to avoid automatically propagating .grad to OG leaf
        gradients = grad(meta_loss, thetas)    # TODO: Will this work???
        for theta, gradient in zip(thetas, gradients):
            theta.data -= beta*gradient
        # thetas are now modified!
        
        training_loss_dict[i_epoch] = meta_loss
    
    # NOTE: Algo 2 Line 11
    return training_loss_dict


################################################################################

if __name__ == "__main__":
    # Set experiment constants
    if torch.cuda.is_available():
        print("CUDA found!")
        MY_DEVICE = torch.device('cuda')
    else:
        print("Warning: using CPU...")
        MY_DEVICE = torch.device('cpu')
    ALPHA = 1e-3    # Inner learning rate
    BETA = 1e-3     # Meta-update learning rate
    K_INNER = 10  # Inner/meta batch size
    K_OUTER = 25
    N_EPOCHS = 70_000
    # N_INNER_EPOCHS = 1    # Just take 1 step for each task in the inner
    ############################################################################
    # Init params (xavier_uniform) for 3 linear layers (weights and biases, 2 apiece)
    THETAS = [torch.rand(40, 1, device=MY_DEVICE).uniform_(-np.sqrt(6/41), np.sqrt(6/41)).requires_grad_(),
              torch.zeros(40, device=MY_DEVICE).requires_grad_(),
              torch.rand(40, 40, device=MY_DEVICE).uniform_(-np.sqrt(6/80), np.sqrt(6/80)).requires_grad_(),
              torch.zeros(40, device=MY_DEVICE).requires_grad_(),
              torch.rand(1, 40, device=MY_DEVICE).uniform_(-np.sqrt(6/41), np.sqrt(6/41)).requires_grad_(),
              torch.zeros(1, device=MY_DEVICE).requires_grad_()]
    # Replace with
    # THETAS = list(model.parameters())
    mnist_model = MLP(784, 10, [300, 100])
    ############################################################################

    # Run experiment!
    thetas = clone_detach_regrad_params(THETAS)     # Preserve original THETAS so we can work with it
    alpha = ALPHA
    beta = BETA
    K_inner_train = K_INNER
    K_outer_test = K_OUTER
    n_epochs = N_EPOCHS
    my_device = MY_DEVICE
    ret_losses = perform_MAML_on_thetas(thetas, alpha, beta, K_inner_train, K_outer_test, n_epochs, my_device)
    # At this point, thetas is MAML-ready

    ############################################################################
    # Plots

    x = torch.linspace(-5, 5, 50).to(my_device)

    # MAML-ready model
    plt.title('MAML, K=10')
    y = mlp(x[..., None], thetas)   # This is the predicted y using the params that are "ready to learn"
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='lightgreen', linestyle='--', linewidth=2.2, label='pre-update')

    # New task to try
    FINAL_TASK = create_task()
    ground_truth_y = FINAL_TASK.a * torch.sin(x + FINAL_TASK.b)
    plt.plot(x.data.cpu().numpy(), ground_truth_y.data.cpu().numpy(), c='red', label='ground truth')

    # Fine-tuning MAML-ready model, 10 gradient steps of size 20 batch (200 shot?)
    fine_tuned_thetas = clone_detach_regrad_params(thetas)  # Preserve thetas
    perform_k_training_steps(fine_tuned_thetas, FINAL_TASK, 20, 10, 1e-3, device=my_device)
    y = mlp(x[..., None], fine_tuned_thetas)
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='darkgreen', linestyle='--', linewidth=2.2, label='10 grad step')

    # After 10,000 gradient steps
    og_params_dupe = clone_detach_regrad_params(THETAS)  # Preserve thetas
    perform_k_training_steps(og_params_dupe, FINAL_TASK, 20, 10_000, 1e-3, device=my_device)
    y = mlp(x[..., None], og_params_dupe)
    plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='cyan', linestyle='--', linewidth=1.2, label='10,000 grad step from OG')

    plt.legend()
    plt.savefig(f'maml_ultimate_{N_EPOCHS}.png')
    plt.show()
