import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsTimeStep(nn.Module):
    def __init__(self, input_size, hidden_size, dropout=0.1, **kwargs):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size

        self.W_in = nn.Linear(input_size, hidden_size)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)

        self.tau = nn.Parameter(torch.ones(hidden_size))
        self.dt = nn.Parameter(torch.tensor(0.1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.constant_(self.tau, 1.0)
        nn.init.constant_(self.dt, 0.1)

    def forward(self, x, h):
        batch_size = x.size(0)

        h_hat = torch.tanh(self.W_in(x) + self.W_rec(h))

        ratio = self.dt / self.tau
        g = torch.sigmoid(ratio).unsqueeze(0).expand(batch_size, -1)

        h_new = h + g * (h_hat - h)
        h_new = torch.clamp(h_new, -10.0, 10.0)

        return h_new

    def get_tau(self):
        return self.tau.detach().cpu().numpy()

    def get_dt(self):
        return self.dt.item()
