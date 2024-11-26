import torch
import torch.nn as nn
import torch.nn.functional as F

# Actor Network
class CustomActorNet(nn.Module):
    def __init__(self, observation_spec, action_spec, fc_layers=(400,300)):
        super(CustomActorNet, self).__init__()
        self._input_shape = observation_spec["observation"].shape[0]
        self._output_shape = action_spec.shape[0]
        self._action_low = action_spec.low
        self._action_high = action_spec.high
        self._fc_layers = fc_layers
        self._layers = []

        input_shape = self._input_shape
        for layer in self._fc_layers:
            self._layers.append(nn.Linear(input_shape, layer))
            input_shape = layer
        self._final_layer = nn.Linear(input_shape, self._output_shape)

    def forward(self, observation):
        x = observation
        for layer in self._layers:
            x = F.relu(layer(x))
        x = torch.tanh(self._final_layer(x))

        output = self._action_low + (self._action_high - self._action_low) * 0.5 * (x + 1)
        return output


# Critic Network
class CustomCriticNet(nn.Module):
    def __init__(self, observation_spec, action_spec, obs_fc_layers=(400,), joint_fc_layers=(300,)):
        super(CustomCriticNet, self).__init__()
        self._input_shape = observation_spec["observation"].shape[0]
        self._joint_input_shape = obs_fc_layers[-1] + action_spec.shape[0]
        self._output_shape = 1
        self._obs_fc_layers = obs_fc_layers
        self._joint_fc_layers = joint_fc_layers
        self._obs_layers = []
        self._joint_layers = []

        input_shape = self._input_shape
        for layer in self._obs_fc_layers:
            self._obs_layers.append(nn.Linear(input_shape, layer))
            input_shape = layer
        input_shape = self._joint_input_shape
        for layer in self._joint_fc_layers:
            self._joint_layers.append(nn.Linear(input_shape, layer))
            input_shape = layer
        self._final_layer = nn.Linear(input_shape, self._output_shape)

    def forward(self, observation, action):
        x = observation
        for layer in self._obs_layers:
            x = F.relu(layer(x))
        x = torch.cat([x, action], dim=1)
        for layer in self._joint_layers:
            x = F.relu(layer(x))
        output = self._final_layer(x)
        return output
