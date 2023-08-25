import torch
import torch.nn as nn

from src.utils.weight_init import layer_init


class ConstructiveCascadeNetwork(nn.Module):
    def __init__(self, input_dim, output_dim, initial_hidden_dim, activation_fn=nn.ReLU, weight_init='xavier'):
        super(ConstructiveCascadeNetwork, self).__init__()
        self.input_dim = input_dim
        self.activation_fn = activation_fn
        self.total_hidden_dim = initial_hidden_dim
        self.weight_init = weight_init

        # Initial hidden layer, as in the paper, should be a single fully connected layer
        self.initial_hidden_layer = nn.Sequential(
            nn.Linear(self.input_dim, initial_hidden_dim),
            self.activation_fn()
        )

        # Cascade layers
        self.cascade_layers = nn.ModuleList([])

        # Initial output layer
        self.output_layers = nn.ModuleList([nn.Linear(initial_hidden_dim, output_dim)])
        
        # Initialize weights
        for module in self.modules():
            if isinstance(module, nn.Linear):
                layer_init(module, self.weight_init)

    def forward(self, x):
        # List to store outputs of all cascade layers and the input
        outputs = [x]
        initial_hidden_output = self.initial_hidden_layer(x)
        outputs.append(initial_hidden_output)

        # pass the current outputs to the next cascade layer
        # each time concatenate the outputs of all previous cascade layers
        for cascade_layer in self.cascade_layers:
            outputs.append(cascade_layer(torch.cat(outputs, dim=-1)))

        # Need to keep both the Batch size channel and the feature channel after the summation
        # Break down the FC layer for controlling different learning rate
        output = None
        for hidden_states, output_layer in zip(outputs[1:], self.output_layers):
            if output is None:
                output = output_layer(hidden_states)
            else:
                output += output_layer(hidden_states)

        return output

    def add_cascade_layer(self, new_cascade_dim, dropout_rate=0.0):
        # Move new created weights to the correct device
        curr_device = next(self.parameters()).device

        # New cascade layer will be connected to the input, initial hidden layer, and all previous cascade layers
        new_cascade_input_dim = self.input_dim + self.total_hidden_dim
        new_cascade_layer = nn.Sequential(
            layer_init(nn.Linear(new_cascade_input_dim, new_cascade_dim), self.weight_init),
            self.activation_fn(),
            nn.Dropout(dropout_rate)
        ).to(curr_device)
        self.cascade_layers.append(new_cascade_layer)
        self.total_hidden_dim += new_cascade_dim

        # Create new output layer
        new_output_layer = layer_init(
            nn.Linear(new_cascade_dim, self.output_layers[0].out_features), self.weight_init).to(curr_device)
        self.output_layers.append(new_output_layer)
