import torch


def compute_td_loss(states, actions, rewards, next_states, is_done,
                    agent, target_network, device, gamma=0.99,
                    check_shapes=False):
    """ Compute td loss using torch operations only. Use the formulae above. """
    states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]
    # for some torch reason should not make actions a tensor
    actions = torch.tensor(actions, device=device, dtype=torch.long)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float) # shape: [batch_size, *state_shape]
    is_done = torch.tensor(is_done.astype('float32'), device=device, dtype=torch.float)  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)
    assert predicted_qvalues.requires_grad, "qvalues must be a torch tensor with grad"
    
    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    # compute q-values for all actions in next states
    # and use it to compute V*(next_states) 
    with torch.no_grad():
        predicted_next_qvalues = target_network(next_states)
        next_state_values = predicted_next_qvalues.max(1)[0] * is_not_done

    assert next_state_values.dim(
    ) == 1 and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * next_state_values
    
    assert target_qvalues_for_actions.requires_grad == False, "do not send gradients to target!"

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions) ** 2)

    if check_shapes:
        assert predicted_next_qvalues.data.dim(
        ) == 2, "make sure you predicted q-values for all actions in next state"
        assert next_state_values.data.dim(
        ) == 1, "make sure you computed V(s') as maximum over just the actions axis and not all axes"
        assert target_qvalues_for_actions.data.dim(
        ) == 1, "there's something wrong with target q-values, they must be a vector"

    return loss


def compute_td_loss_double_q(states, actions, rewards, next_states, is_done,
                             agent, target_network, device, gamma=0.99):
    """ Compute td loss using torch operations only. Use DDQN loss. """
    states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]
    # for some torch reason should not make actions a tensor
    actions = torch.tensor(actions, device=device, dtype=torch.long)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float) # shape: [batch_size, *state_shape]
    is_done = torch.tensor(is_done.astype('float32'), device=device, dtype=torch.float)  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)
    assert predicted_qvalues.requires_grad, "qvalues must be a torch tensor with grad"
    
    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    # compute q-values for all actions in next states
    # and use it to compute V*(next_states) 
    with torch.no_grad():
        predicted_next_qvalues_target = target_network(next_states)
        predicted_next_qvalues_agent = agent(next_states)
        greedy_actions = torch.argmax(predicted_next_qvalues_agent, dim=1)
        next_state_values = predicted_next_qvalues_target[range(len(greedy_actions)), greedy_actions] * is_not_done

    assert next_state_values.dim(
    ) == 1 and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * next_state_values
    
    assert target_qvalues_for_actions.requires_grad == False, "do not send gradients to target!"

    # mean squared error loss to minimize
    loss = torch.mean((predicted_qvalues_for_actions -
                       target_qvalues_for_actions) ** 2)

    return loss


def compute_td_loss_double_q_PER(states, actions, rewards, next_states, is_done, weights,
                                 agent, target_network, device, gamma=0.99):
    """ Compute td loss using torch operations only. Use DDQN loss. """
    states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]
    # for some torch reason should not make actions a tensor
    actions = torch.tensor(actions, device=device, dtype=torch.long)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float) # shape: [batch_size, *state_shape]
    weights = torch.tensor(weights, device=device, dtype=torch.float) # shape: [batch_size]
    is_done = torch.tensor(is_done.astype('float32'), device=device, dtype=torch.float)  # shape: [batch_size]
    is_not_done = 1 - is_done

    # get q-values for all actions in current states
    predicted_qvalues = agent(states)
    assert predicted_qvalues.requires_grad, "qvalues must be a torch tensor with grad"
    
    # select q-values for chosen actions
    predicted_qvalues_for_actions = predicted_qvalues[range(len(actions)), actions]

    # compute q-values for all actions in next states
    # and use it to compute V*(next_states) 
    with torch.no_grad():
        predicted_next_qvalues_target = target_network(next_states)
        predicted_next_qvalues_agent = agent(next_states)
        greedy_actions = torch.argmax(predicted_next_qvalues_agent, dim=1)
        next_state_values = predicted_next_qvalues_target[range(len(greedy_actions)), greedy_actions] * is_not_done

    assert next_state_values.dim(
    ) == 1 and next_state_values.shape[0] == states.shape[0], "must predict one value per state"

    # compute "target q-values" for loss - it's what's inside square parentheses in the above formula.
    # at the last state use the simplified formula: Q(s,a) = r(s,a) since s' doesn't exist
    # you can multiply next state values by is_not_done to achieve this.
    target_qvalues_for_actions = rewards + gamma * next_state_values
    
    assert target_qvalues_for_actions.requires_grad == False, "do not send gradients to target!"
    
    errors = torch.abs(predicted_qvalues_for_actions - target_qvalues_for_actions).detach().cpu().numpy()

    # mean squared error loss to minimize
    loss = torch.mean(weights * (predicted_qvalues_for_actions -
                       target_qvalues_for_actions) ** 2)

    return loss, errors


def huber(x, delta=1):
    return torch.where(x.abs() < delta, 0.5 * x ** 2, delta * (x.abs() - 0.5 * delta))


def compute_td_loss_qr_double_q(states, actions, rewards, next_states, is_done,
                             agent, target_network, device, gamma=0.99):
    """ Compute td loss using torch operations only. Use DDQN loss. """
    states = torch.tensor(states, device=device, dtype=torch.float)    # shape: [batch_size, *state_shape]
    # for some torch reason should not make actions a tensor
    actions = torch.tensor(actions, device=device, dtype=torch.long)    # shape: [batch_size]
    rewards = torch.tensor(rewards, device=device, dtype=torch.float)  # shape: [batch_size]
    next_states = torch.tensor(next_states, device=device, dtype=torch.float) # shape: [batch_size, *state_shape]
    is_done = torch.tensor(is_done.astype('float32'), device=device, dtype=torch.float)  # shape: [batch_size]
    is_not_done = 1 - is_done

    tau = ((2 * torch.arange(agent.n_quant) + 1) / (2.0 * agent.n_quant)).view(1, -1).to(device)
    
    predicted_quantiles_for_actions = agent(states)[range(len(actions)), actions]
    with torch.no_grad():
        predicted_next_quantiles_target = target_network(next_states)
        next_actions = torch.argmax(agent(states).mean(dim=2), dim=1)
        next_state_quantiles = predicted_next_quantiles_target[range(len(next_actions)), next_actions]
        
    target_quantiles_for_actions = rewards.unsqueeze(-1) + gamma * is_not_done.unsqueeze(-1) * next_state_quantiles
    
    diff = target_quantiles_for_actions.t().unsqueeze(-1) - predicted_quantiles_for_actions 
    loss = huber(diff) * (tau - (diff.detach() < 0).float()).abs()
    loss = loss.mean()

    return loss