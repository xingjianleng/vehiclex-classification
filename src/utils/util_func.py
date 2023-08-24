def update_optimizer(optimizer, model, learning_rates):
    """
    Update the optimizer to assign different learning rates for the latest cascade layer, the new weights in the output layer, and all other weights.
    
    optimizer: The existing optimizer.
    model: The cascade correlation network model.
    learning_rates: Dictionary containing learning rates for the three groups: {'L1': value, 'L2': value, 'L3': value}
    """
    
    # Identify the names of the weights of the latest cascade layer
    latest_cascade_names = set()
    # Identify the names of the new weights in the output layer associated with the latest cascade layer
    new_output_names = set()

    for name, param in model.named_parameters():
        if "cascade_layers." + str(len(model.cascade_layers) - 1) in name:
            latest_cascade_names.add(name)
        elif "output_layers." + str(len(model.output_layers) - 1) in name:
            new_output_names.add(name)
    
    # Group the parameters based on their names
    group1_weights, group2_weights, group3_weights = [], [], []
    
    for name, param in model.named_parameters():
        if name in latest_cascade_names:
            group1_weights.append(param)
        elif name in new_output_names:
            group2_weights.append(param)
        else:
            group3_weights.append(param)
    
    # Extract existing hyperparameters from the optimizer
    existing_hyperparams = optimizer.param_groups[0].copy()
    del existing_hyperparams['params']
    del existing_hyperparams['lr']

    groups_weights = [group1_weights, group2_weights, group3_weights]
    for param_group, lr in zip(groups_weights, learning_rates.values()):    
        # Append new parameter groups for the optimizer
        optimizer.param_groups.append({
            **existing_hyperparams,
            'params': param_group, 
            'lr': lr
        })

    return optimizer
