import torch
import torch.nn as nn

def mlp(params, x):
    h = torch.relu(torch.nn.functional.linear(x, params[0], bias=params[1]))
    h = torch.relu(torch.nn.functional.linear(h, params[2], bias=params[3]))
    return torch.nn.functional.linear(h, params[4], bias=params[5])

class MLP(nn.Module):
    def __init__(self, in_dims, out_dims, hidden_dims=None, 
                 dimension_progression=None,
                 criterion_type='MSE',
                 optimizer_type='Adam', optimizer_lr=1e-3) -> None:
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

        # Set loss function
        if criterion_type == 'MSE':
            self.criterion = nn.MSELoss()
        elif criterion_type == 'CrossEntropyLoss':
            self.criterion = nn.CrossEntropyLoss()
        else:
            raise ValueError("No such criterion_type")

        # Set optimizer
        if optimizer_type == 'Adam':
            self.optimizer = torch.optim.Adam(self.parameters(), lr=optimizer_lr)
        elif optimizer_type == 'SGD':
            self.optimizer = torch.optim.SGD(self.parameters(), lr=optimizer_lr)
        else:
            raise ValueError("No such optimizer_type")
    
    def dim_prog(in_dims, out_dims, hidden_dims=None, features=None):
        if features is not None:
            in_dims = features.shape[-1]
        if hidden_dims is None:
            hidden_dims = []
        return [in_dims] + hidden_dims + [out_dims]
    
    def forward(self, input):
        # input: (batch, in_dims)
        return self.sequential(input)
