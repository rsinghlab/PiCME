import math

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GRUD(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        x_mean=0,
        bias=True,
        batch_first=False,
        dropout=0.1,
        is_packed=False,
    ):
        super(GRUD, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.is_packed = is_packed
        self.batch_first = batch_first

        # self.zeros used for Decay Rate equation (gamma = exp(-max(0, ...)))
        self.zeros = torch.nn.Parameter(torch.zeros(input_size), requires_grad=False)

        if x_mean == 0:
            self.x_mean = torch.nn.Parameter(torch.zeros(input_size))
        else:
            self.x_mean = torch.nn.Parameter(torch.tensor(x_mean))
        self.bias = bias
        self.dropout = dropout

        # Initialize Decay Rate Ws and Bs: Gamma --> 1 for Inputs, 1 for Hidden State
        self.w_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
        self.w_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        self.b_dg_x = torch.nn.Parameter(torch.Tensor(input_size))
        self.b_dg_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        # Intialize Update Gate Parameters: Z
        self.w_xz = torch.nn.Parameter(torch.Tensor(input_size))
        self.u_hz = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.v_mz = torch.nn.Parameter(torch.Tensor(input_size))
        self.b_z = torch.nn.Parameter(torch.Tensor(hidden_size))

        # Initialize Reset Gate Parameters: R
        self.w_xr = torch.nn.Parameter(torch.Tensor(input_size))
        self.u_hr = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.v_mr = torch.nn.Parameter(torch.Tensor(input_size))
        self.b_r = torch.nn.Parameter(torch.Tensor(hidden_size))

        # Initialize ~h Gate Parameters
        self.w_xh = torch.nn.Parameter(torch.Tensor(input_size))
        self.u_hh = torch.nn.Parameter(torch.Tensor(hidden_size))
        self.v_mh = torch.nn.Parameter(torch.Tensor(input_size))
        self.b_h = torch.nn.Parameter(torch.Tensor(hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.hidden_size)
        for weight in self.parameters():
            torch.nn.init.uniform_(weight, -stdv, stdv)

    def forward(self, X, Mask, Delta):
        device = X.get_device()

        if self.is_packed:
            lengths = X.batch_sizes
            batch_size = lengths.size()[0]
            X = nn.utils.rnn.pad_packed_sequence(X, batch_first=False)
            Mask = nn.utils.rnn.pad_packed_sequence(Mask, batch_first=False)
            Delta = nn.utils.rnn.pad_packed_sequence(Delta, batch_first=False)

        num_steps = X.size()[0]  # first dim is longest length from pad packed
        batch_size = X.size()[1]
        hidden_states = torch.zeros(batch_size, num_steps, self.hidden_size)
        h = torch.nn.Parameter(torch.zeros(self.hidden_size)).to(device)

        for step in range(num_steps):
            x = torch.squeeze(X[step : step + 1, :, :])
            m = torch.squeeze(Mask[step : step + 1, :, :])
            d = torch.squeeze(Delta[step : step + 1, :, :])

            gamma_x = torch.exp(-torch.max(self.zeros, (self.w_dg_x * d + self.b_dg_x)))
            gamma_h = torch.exp(-torch.max(self.zeros, (self.w_dg_h * d + self.b_dg_h)))

            # Apply decay to x --> no missing values, thus x_mean = 0.
            x = m * x + (1 - m) * (gamma_x * x + (1 - gamma_x) * self.x_mean)

            # Apply decay to h_t
            h = gamma_h * h

            # Apply GRU-D Cell Math and Calculate Next h
            z = torch.sigmoid(self.w_xz * x + self.u_hz * h + self.b_z)
            r = torch.sigmoid(self.w_xr * x + self.u_hr * h + self.b_r)
            h_tilde = torch.tanh(
                (self.w_xh * x + self.u_hh * (r * h) + self.v_mh * m + self.b_h)
            )

            h = (1 - z) * h + z * h_tilde

            if self.dropout > 0:
                dropout = torch.nn.Dropout(p=self.dropout)
                h = dropout(h)

            hidden_states[:, step, :] = h

        # Check for NaN in hidden_states
        if torch.isnan(hidden_states).any():
            print("NaN detected in hidden_states")

        hidden_states = hidden_states.to(device)
        return hidden_states, hidden_states[:, -1, :]
