import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], activation_fn=nn.ReLU):
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

    def forward(self, x):
        for layer in self.hidden_layers:
            x = layer(x)
        return x
