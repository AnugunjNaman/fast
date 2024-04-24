import torch
import numpy as np
from tqdm import tqdm
from torch.autograd import grad

def perform_k_training_steps(params, fmodel, task, batch_size, inner_training_steps, alpha, device='cpu', verbose=False):
    for epoch in range(inner_training_steps):
        x_batch, target, loss_fct = task.sample(batch_size)
        loss = loss_fct(fmodel(params, x_batch.to(device)), target.to(device))

        for p in params:  # Zero grad
            p.grad = None
        gradients = grad(loss, params)
        for p, g in zip(params, gradients):  # Grad step
            p.data -= alpha * g
        
        if verbose:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
    
    if verbose:
        print("Finished training!")

    return params

# def perform_k_training_steps(params, fmodel, task, batch_size, inner_training_steps, alpha, device='cpu', verbose=False):
#     # NOTE: Assumes params coming in is DETACHED, i.e. they are leaves!
#     # Directly replacing the other function won't work because these cloned params aren't leaves, so .step() should fail
#     for param in params:
#         param.retain_grad()
#     optimizer = torch.optim.SGD(params, lr=alpha)   # Want to step traditionally
#     for epoch in range(inner_training_steps):
#         optimizer.zero_grad()   # Wipe all params' gradients (in case my code messes up elsewhere)
#         x_batch, target, loss_fct = task.sample(batch_size)
#         loss = loss_fct(fmodel(params, x_batch.to(device)), target.to(device))
#         loss.backward()     # Due to retain_grad() on these clones, the original params get these grads too
#         if verbose:
#             print(f"Epoch {epoch}, Loss: {loss.item()}")
#         optimizer.step()
#     if verbose:
#         print("Finished training!")
#     optimizer.zero_grad()   # Wipe these clones' .grads (in case my code assumes this later)

def maml(params, fmodel, meta_optimizer, inner_training_steps, nb_epochs, batch_size_K, alpha, task_sampler, nb_tasks=10, device='cpu'):
    """
    Algorithm from https://arxiv.org/pdf/1703.03400v3.pdf (MAML for Few-Shot Supervised Learning)
    """
    training_loss = []
    for epoch in tqdm(range(nb_epochs)):  # Line 2 in the pseudocode

        theta_i_prime = []
        D_i_prime = []

        # Sample batch of tasks
        tasks = [task_sampler() for _ in range(nb_tasks)]  # Line 3 in the pseudocode
        for task in tasks:
            cloned_params = [p.clone() for p in params]
            perform_k_training_steps(cloned_params, fmodel, task, batch_size_K,
                                     inner_training_steps, alpha, device=device)
            theta_i_prime.append(cloned_params)
            # Sample data points Di' for the meta-update (line 8 in the pseudocode)
            x, y, loss_fct = task.sample(25)
            D_i_prime.append((x, y, loss_fct))

        # Meta update
        meta_optimizer.zero_grad()
        batch_training_loss = []
        losses = []
        for i in range(nb_tasks):
            x, y, loss_fct = D_i_prime[i]
            f_theta_prime = theta_i_prime[i]
            # Compute \nabla_theta L(f_theta_i_prime) for task ti
            loss = loss_fct(fmodel(f_theta_prime, x.to(device)), y.to(device))
            # loss.backward()
            losses.append(loss)
            batch_training_loss.append(loss.item())
        losses_sum = sum(losses)    # Sum the list of losses!!!
        losses_sum.backward()   # All together, propagate to leaf params' .grads

        meta_optimizer.step()  # Line 10 in the pseudocode
        training_loss.append(np.mean(batch_training_loss))
    return training_loss

################################################################################

# def perform_k_training_steps_class(params, fmodel, task, batch_size, inner_training_steps, alpha, device='cpu'):
#     # NOTE: Assumes params coming in is DETACHED, i.e. they are leaves!
#     # Directly replacing the other function won't work because these cloned params aren't leaves, so .step() should fail
#     for param in params:
#         param.retain_grad()
#     optimizer = torch.optim.SGD(params, lr=alpha)   # Want to step traditionally
#     for epoch in range(inner_training_steps):
#         optimizer.zero_grad()   # Wipe all params' gradients (in case my code messes up elsewhere)
#         (x_a, y_a), (x_b, y_b), loss_func = task.sample(batch_size)
#         loss = loss_func(fmodel(params, x_batch.to(device)), target.to(device))
#         loss.backward()     # Due to retain_grad() on these clones, the original params get these grads too
#         optimizer.step()
#     optimizer.zero_grad()   # Wipe these clones' .grads (in case my code assumes this later)

# def maml_class(params, fmodel, meta_optimizer, inner_training_steps, nb_epochs, batch_size_K, alpha, task_sampler, nb_tasks=10, device='cpu'):
#     """
#     Algorithm from https://arxiv.org/pdf/1703.03400v3.pdf (MAML for Few-Shot Supervised Learning)
#     """
#     training_loss = []
#     for epoch in tqdm(range(nb_epochs)):  # Line 2 in the pseudocode

#         theta_i_prime = []
#         D_i_prime = []

#         # Sample batch of tasks
#         tasks = [task_sampler() for _ in range(nb_tasks)]  # Line 3 in the pseudocode
#         for task in tasks:
#             cloned_params = [p.clone() for p in params]
#             perform_k_training_steps(cloned_params, fmodel, task, batch_size_K,
#                                      inner_training_steps, alpha, device=device)
#             theta_i_prime.append(cloned_params)
#             # Sample data points Di' for the meta-update (line 8 in the pseudocode)
#             x, y, loss_fct = task.sample(25)
#             D_i_prime.append((x, y, loss_fct))

#         # Meta update
#         meta_optimizer.zero_grad()
#         batch_training_loss = []
#         losses = []
#         for i in range(nb_tasks):
#             x, y, loss_fct = D_i_prime[i]
#             f_theta_prime = theta_i_prime[i]
#             # Compute \nabla_theta L(f_theta_i_prime) for task ti
#             loss = loss_fct(fmodel(f_theta_prime, x.to(device)), y.to(device))
#             # loss.backward()
#             losses.append(loss)
#             batch_training_loss.append(loss.item())
#         losses_sum = sum(losses)    # Sum the list of losses!!!
#         losses_sum.backward()   # All together, propagate to leaf params' .grads

#         meta_optimizer.step()  # Line 10 in the pseudocode
#         training_loss.append(np.mean(batch_training_loss))
#     return training_loss
