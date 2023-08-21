import torch.nn as nn


class LinearNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims=[], activation=nn.ReLU()):
        super(LinearNet, self).__init__()
        self.layers = nn.ModuleList([])
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(input_dim, hidden_dim))
            self.layers.append(activation)
            input_dim = hidden_dim
        self.layers.append(nn.Linear(input_dim, output_dim))
        
    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
