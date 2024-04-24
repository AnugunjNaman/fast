from models import MLP, mlp
import torch
from utils import set_random_seeds
import matplotlib.pyplot as plt
from maml import perform_k_training_steps, maml
from tasks import sample_task, sample_mnisttask
import numpy as np
import functorch

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
set_random_seeds(42)

meta_optimizer = torch.optim.Adam(params, lr=1e-3)
N_EPOCHS = 200
training_loss = maml(params, fmodel, meta_optimizer, 1, N_EPOCHS, 10, 1e-3, sample_mnisttask, device=device, nb_tasks=10)
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
# 1) TODO accuracy evaluation
# 2) ... still need to do accuracy evaluation
new_params = [p.clone() for p in params]
perform_k_training_steps(new_params, fmodel, mtask, 10, 1000, 1e-3, device=device, verbose=True)
# 3) ... still need to do accuracy evaluation
scratch_mnist_model = MLP(784, 10, [300, 100], criterion_type='CrossEntropyLoss')
scratch_mnist_model.to(device)
scratch_fmodel, scratch_params = functorch.make_functional(scratch_mnist_model)
perform_k_training_steps(scratch_params, scratch_fmodel, mtask, 10, 1000, 1e-3, device=device, verbose=True)

# 4) TODO
# plt.title('MNIST MAML Test: N=2-Way, K=10-Shot')
# # x = torch.linspace(-5, 5, 50).to(device)
# # y = mlp(x[..., None], params)
# plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='lightgreen', linestyle='--', linewidth=2.2,
#             label='pre-update')

# ground_truth_y = task.a * torch.sin(x + task.b)
# plt.plot(x.data.cpu().numpy(), ground_truth_y.data.cpu().numpy(), c='red', label='ground truth')
# # Fine-tuning, 10 gradient steps
# new_params = [p.clone() for p in params]
# perform_k_training_steps(new_params, mlp, task, 10, 10, 1e-3, device=device)    # Build in custom printing?
# # After 10 gradient steps
# y = mlp(x[..., None], new_params)
# plt.plot(x.data.cpu().numpy(), y.data.cpu().numpy(), c='darkgreen', linestyle='--', linewidth=2.2,
#             label='10 grad step')
# plt.legend()
# plt.xlim(-5, 5)
# plt.ylim(-5, 5)
# plt.savefig('code100_5K_seed42_backwardtweak_aligned_alt_MLP_temp.png')
# plt.show()
