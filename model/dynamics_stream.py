import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsTimeStep(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, dt_min=0.001, dt_max=0.5):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)

        self.tau = nn.Parameter(torch.empty(hidden_size).uniform_(0.5, 2.0))
        
        self.dt_min = dt_min
        self.dt_max = dt_max
        self.tolerance = 1e-3
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_in.weight, gain=1.0)
        nn.init.zeros_(self.W_in.bias)
        nn.init.orthogonal_(self.W_rec.weight, gain=1.0)
        with torch.no_grad():
            self.tau.data.clamp_(0.1, 10.0)

    def compute_ode_derivative(self, x, h):

        tau_clamped = torch.clamp(self.tau, min=0.1, max=10.0)

        total_input = self.W_in(x) + self.W_rec(h)
        target_state = torch.tanh(total_input)

        dhdt = (target_state - h) / tau_clamped
        return dhdt

    def forward(self, x, h):
        k1 = self.compute_ode_derivative(x, h)

        change_magnitude = torch.norm(k1, dim=-1, keepdim=True)
        adaptive_dt = self.tolerance / (change_magnitude + 1e-8)
        dt = torch.clamp(adaptive_dt, self.dt_min, self.dt_max).mean()

        h_mid = h + (dt / 2) * k1

        k2 = self.compute_ode_derivative(x, h_mid)

        h_new = h + dt * k2

        h_new = torch.clamp(h_new, -10.0, 10.0)

        return h_new
