import torch
import torch.nn as nn
import torch.nn.functional as F


class DynamicsTimeStep(nn.Module):
    def __init__(self, input_size, hidden_size, dt=0.1, nonlinearity='tanh'):
        super(DynamicsTimeStep, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.dt = dt

        self.W_in = nn.Linear(input_size, hidden_size, bias=True)
        self.W_rec = nn.Linear(hidden_size, hidden_size, bias=False)
        self.tau = nn.Parameter(torch.empty(hidden_size).uniform_(0.5, 2.0))

        if nonlinearity == 'tanh':
            self.activation = torch.tanh
        elif nonlinearity == 'relu':
            self.activation = F.relu
        elif nonlinearity == 'sigmoid':
            self.activation = torch.sigmoid
        elif nonlinearity == 'swish':
            self.activation = F.silu
        else:
            raise ValueError(f"Unsupported nonlinearity: {nonlinearity}")

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.W_in.weight, gain=1.0)
        nn.init.zeros_(self.W_in.bias)
        nn.init.orthogonal_(self.W_rec.weight, gain=1.0)
        with torch.no_grad():
            self.tau.data.clamp_(0.1, 10.0)

    def compute_ode_derivative(self, x, h):
        tau_clamped = torch.clamp(self.tau, min=0.1, max=10.0)
        input_contrib = self.W_in(x)
        recurrent_contrib = self.W_rec(h)
        total_input = input_contrib + recurrent_contrib
        target_state = self.activation(total_input)
        dhdt = (target_state - h) / tau_clamped
        return dhdt

    def forward(self, x, h):
        dhdt = self.compute_ode_derivative(x, h)
        h_new = h + self.dt * dhdt
        return h_new


class DynamicsStream(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dt=0.1, nonlinearity='tanh'):
        super(DynamicsStream, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dynamics_layers = nn.ModuleList()

        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.dynamics_layers.append(
                DynamicsTimeStep(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    dt=dt,
                    nonlinearity=nonlinearity
                )
            )

        self.output_layer = nn.Linear(hidden_size, output_size)

        self.use_residual = (input_size == output_size)
        if not self.use_residual and input_size != output_size:
            self.residual_proj = nn.Linear(input_size, output_size)
        else:
            self.residual_proj = None

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        hidden_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]

        outputs = []

        for t in range(seq_len):
            current_input = x[:, t, :]

            for layer_idx in range(self.num_layers):
                if layer_idx == 0:
                    hidden_states[layer_idx] = self.dynamics_layers[layer_idx](
                        current_input, hidden_states[layer_idx]
                    )
                else:
                    hidden_states[layer_idx] = self.dynamics_layers[layer_idx](
                        hidden_states[layer_idx - 1], hidden_states[layer_idx]
                    )

            output_t = self.output_layer(hidden_states[-1])
            outputs.append(output_t)

        output_sequence = torch.stack(outputs, dim=1)

        if self.use_residual:
            output_sequence = output_sequence + x
        elif self.residual_proj is not None:
            residual = self.residual_proj(x)
            output_sequence = output_sequence + residual

        return output_sequence


class AdvancedDynamicsTimeStep(DynamicsTimeStep):
    def __init__(self, input_size, hidden_size, dt=0.1, nonlinearity='tanh',
                 adaptive_dt=False, integration_method='euler'):
        super(AdvancedDynamicsTimeStep, self).__init__(
            input_size, hidden_size, dt, nonlinearity
        )

        self.adaptive_dt = adaptive_dt
        self.integration_method = integration_method

        if adaptive_dt:
            self.dt_min = 0.001
            self.dt_max = 0.5
            self.tolerance = 1e-3

    def adaptive_step_size(self, h, dhdt):
        if not self.adaptive_dt:
            return self.dt

        change_magnitude = torch.norm(dhdt, dim=-1, keepdim=True)
        adaptive_dt = self.tolerance / (change_magnitude + 1e-8)
        adaptive_dt = torch.clamp(adaptive_dt, self.dt_min, self.dt_max)

        return adaptive_dt.mean().item()

    def runge_kutta_2(self, x, h, dt):
        k1 = self.compute_ode_derivative(x, h)
        h_mid = h + (dt / 2) * k1
        k2 = self.compute_ode_derivative(x, h_mid)
        h_new = h + dt * k2
        return h_new

    def runge_kutta_4(self, x, h, dt):
        k1 = self.compute_ode_derivative(x, h)
        h2 = h + (dt / 2) * k1
        k2 = self.compute_ode_derivative(x, h2)
        h3 = h + (dt / 2) * k2
        k3 = self.compute_ode_derivative(x, h3)
        h4 = h + dt * k3
        k4 = self.compute_ode_derivative(x, h4)
        h_new = h + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        return h_new

    def forward(self, x, h):
        dhdt = self.compute_ode_derivative(x, h)

        if self.adaptive_dt:
            dt = self.adaptive_step_size(h, dhdt)
        else:
            dt = self.dt

        if self.integration_method == 'euler':
            h_new = h + dt * dhdt
        elif self.integration_method == 'rk2':
            h_new = self.runge_kutta_2(x, h, dt)
        elif self.integration_method == 'rk4':
            h_new = self.runge_kutta_4(x, h, dt)
        else:
            raise ValueError(f"Unknown integration method: {self.integration_method}")

        h_new = torch.clamp(h_new, -10.0, 10.0)

        return h_new


class ImprovedDynamicsTimeStep(AdvancedDynamicsTimeStep):
    def __init__(self, input_size, hidden_size, dropout=0.1):
        super(ImprovedDynamicsTimeStep, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            dt=0.1,
            nonlinearity='tanh',
            adaptive_dt=True,
            integration_method='rk2'
        )


class ImprovedDynamicsStream(DynamicsStream):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.1):
        super(DynamicsStream, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.dynamics_layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.dynamics_layers.append(
                AdvancedDynamicsTimeStep(
                    input_size=layer_input_size,
                    hidden_size=hidden_size,
                    dt=0.1,
                    nonlinearity='tanh',
                    adaptive_dt=True,
                    integration_method='rk2'
                )
            )

        self.output_layer = nn.Linear(hidden_size, output_size)

        self.use_residual = (input_size == output_size)
        if not self.use_residual and input_size != output_size:
            self.residual_proj = nn.Linear(input_size, output_size)
        else:
            self.residual_proj = None

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        hidden_states = [
            torch.zeros(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)
            for _ in range(self.num_layers)
        ]

        outputs = []

        for t in range(seq_len):
            current_input = x[:, t, :]

            for layer_idx in range(self.num_layers):
                if layer_idx == 0:
                    hidden_states[layer_idx] = self.dynamics_layers[layer_idx](
                        current_input, hidden_states[layer_idx]
                    )
                else:
                    hidden_states[layer_idx] = self.dynamics_layers[layer_idx](
                        hidden_states[layer_idx - 1], hidden_states[layer_idx]
                    )

            output_t = self.output_layer(hidden_states[-1])
            outputs.append(output_t)

        output_sequence = torch.stack(outputs, dim=1)

        if self.use_residual:
            output_sequence = output_sequence + x
        elif self.residual_proj is not None:
            residual = self.residual_proj(x)
            output_sequence = output_sequence + residual

        return output_sequence

