import numpy as np
import torch
from torch import nn
from utils import conv2d_size_out

class DQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        n_frames, current_w, current_h = state_shape
        self.network = nn.Sequential()
        self.network.add_module('conv_1', nn.Conv2d(in_channels = n_frames, out_channels = 32,
                                                    kernel_size = 8, stride = 4))
        self.network.add_module('relu_1', nn.ReLU())
        current_w, current_h = conv2d_size_out(current_w, 8, 4), conv2d_size_out(current_h, 8, 4)
        self.network.add_module('conv_2', nn.Conv2d(in_channels = 32, out_channels = 64,
                                                    kernel_size = 4, stride = 2))
        self.network.add_module('relu_2', nn.ReLU())
        current_w, current_h = conv2d_size_out(current_w, 4, 2), conv2d_size_out(current_h, 4, 2)
        self.network.add_module('conv_3', nn.Conv2d(in_channels = 64, out_channels = 64,
                                                    kernel_size = 3, stride = 1))
        self.network.add_module('relu_3', nn.ReLU())
        current_w, current_h = conv2d_size_out(current_w, 3, 1), conv2d_size_out(current_h, 3, 1)
        
        self.linear = nn.Sequential()
        self.linear.add_module('linear_1', nn.Linear(64 * current_h * current_w, 256))
        self.linear.add_module('relu_linear', nn.ReLU())
        self.linear.add_module('linear_2', nn.Linear(256, n_actions))
        
    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        batch_size, _, _, _ = state_t.shape
        im_features = self.network(state_t)
        qvalues = self.linear(im_features.view(batch_size, -1))

        assert len(
            qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == self.n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)
    

class DuelingDQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.state_shape = state_shape
        
        n_frames, current_w, current_h = state_shape
        self.network = nn.Sequential()
        self.network.add_module('conv_1', nn.Conv2d(in_channels = n_frames, out_channels = 32,
                                                    kernel_size = 8, stride = 4))
        self.network.add_module('relu_1', nn.ReLU())
        current_w, current_h = conv2d_size_out(current_w, 8, 4), conv2d_size_out(current_h, 8, 4)
        self.network.add_module('conv_2', nn.Conv2d(in_channels = 32, out_channels = 64,
                                                    kernel_size = 4, stride = 2))
        self.network.add_module('relu_2', nn.ReLU())
        current_w, current_h = conv2d_size_out(current_w, 4, 2), conv2d_size_out(current_h, 4, 2)
        self.network.add_module('conv_3', nn.Conv2d(in_channels = 64, out_channels = 64,
                                                    kernel_size = 3, stride = 1))
        self.network.add_module('relu_3', nn.ReLU())
        current_w, current_h = conv2d_size_out(current_w, 3, 1), conv2d_size_out(current_h, 3, 1)
        
        self.v = nn.Sequential()
        self.v.add_module('linear_1', nn.Linear(64 * current_h * current_w, 512))
        self.v.add_module('relu_linear', nn.ReLU())
        self.v.add_module('linear_2', nn.Linear(512, 1))
        
        self.adv = nn.Sequential()
        self.adv.add_module('linear_1', nn.Linear(64 * current_h * current_w, 512))
        self.adv.add_module('relu_linear', nn.ReLU())
        self.adv.add_module('linear_2', nn.Linear(512, self.n_actions))
        
    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        batch_size, _, _, _ = state_t.shape
        im_features = self.network(state_t)
        
        v = self.v(im_features.view(batch_size, -1))
        adv = self.adv(im_features.view(batch_size, -1))
        qvalues = v - adv.mean(dim=1, keepdim=True) + adv

        assert len(
            qvalues.shape) == 2 and qvalues.shape[0] == state_t.shape[0] and qvalues.shape[1] == self.n_actions

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        model_device = next(self.parameters()).device
        states = torch.tensor(states, device=model_device, dtype=torch.float)
        qvalues = self.forward(states)
        return qvalues.data.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)


class QRDQNAgent(nn.Module):
    def __init__(self, state_shape, n_actions, n_quant=51, epsilon=0):

        super().__init__()
        self.epsilon = epsilon
        self.n_actions = n_actions
        self.n_quant = n_quant
        self.state_shape = state_shape
        
        n_frames, current_w, current_h = state_shape
        self.network = nn.Sequential()
        self.network.add_module('conv_1', nn.Conv2d(in_channels = n_frames, out_channels = 32,
                                                    kernel_size = 8, stride = 4))
        self.network.add_module('relu_1', nn.ReLU())
        current_w, current_h = conv2d_size_out(current_w, 8, 4), conv2d_size_out(current_h, 8, 4)
        self.network.add_module('conv_2', nn.Conv2d(in_channels = 32, out_channels = 64,
                                                    kernel_size = 4, stride = 2))
        self.network.add_module('relu_2', nn.ReLU())
        current_w, current_h = conv2d_size_out(current_w, 4, 2), conv2d_size_out(current_h, 4, 2)
        self.network.add_module('conv_3', nn.Conv2d(in_channels = 64, out_channels = 64,
                                                    kernel_size = 3, stride = 1))
        self.network.add_module('relu_3', nn.ReLU())
        current_w, current_h = conv2d_size_out(current_w, 3, 1), conv2d_size_out(current_h, 3, 1)
        
        self.linear = nn.Sequential()
        self.linear.add_module('linear_1', nn.Linear(64 * current_h * current_w, 256))
        self.linear.add_module('relu_linear', nn.ReLU())
        self.linear.add_module('linear_2', nn.Linear(256, n_actions * n_quant))
        
    def forward(self, state_t):
        """
        takes agent's observation (tensor), returns qvalues (tensor)
        :param state_t: a batch of 4-frame buffers, shape = [batch_size, 4, h, w]
        """
        # Use your network to compute qvalues for given state
        batch_size, _, _, _ = state_t.shape
        im_features = self.network(state_t)
        qvalues = self.linear(im_features.view(batch_size, -1)).view(batch_size, self.n_actions, self.n_quant)

        return qvalues

    def get_qvalues(self, states):
        """
        like forward, but works on numpy arrays, not tensors
        """
        with torch.no_grad():
            model_device = next(self.parameters()).device
            states = torch.tensor(states, device=model_device, dtype=torch.float)
            qvalues = self.forward(states).mean(dim=2)
        return qvalues.cpu().numpy()

    def sample_actions(self, qvalues):
        """pick actions given qvalues. Uses epsilon-greedy exploration strategy. """
        epsilon = self.epsilon
        batch_size, n_actions = qvalues.shape

        random_actions = np.random.choice(n_actions, size=batch_size)
        best_actions = qvalues.argmax(axis=-1)

        should_explore = np.random.choice(
            [0, 1], batch_size, p=[1-epsilon, epsilon])
        return np.where(should_explore, random_actions, best_actions)