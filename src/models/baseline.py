import torch.nn as nn

from src.utils.weight_init import layer_init


class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], activation_fn=nn.ReLU, weight_init='xavier'):
        super(LinearNet, self).__init__()
        self.hidden_layers = nn.ModuleList([])
        for hidden_dim in hidden_dims:
            # ensure that the activation function is instantiated inside nn.Module
            self.hidden_layers.append(nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                activation_fn()
            ))
            input_dim = hidden_dim
        self.hidden_layers.append(nn.Linear(input_dim, output_dim))
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                layer_init(module, weight_init)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x
