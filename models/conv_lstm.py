import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """ConvLSTM cell with peephole connections, following Shi et al. (2015)."""

    def __init__(self, in_ch, hidden_ch, kernel_size=3, bias=True):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if kernel_size[0] != kernel_size[1]:
            raise ValueError("ConvLSTMCell expects a square kernel size")

        padding = kernel_size[0] // 2
        self.hidden_ch = hidden_ch

        self.conv_x = nn.Conv2d(
            in_ch,
            4 * hidden_ch,
            kernel_size,
            padding=padding,
            bias=bias,
        )
        self.conv_h = nn.Conv2d(
            hidden_ch,
            4 * hidden_ch,
            kernel_size,
            padding=padding,
            bias=False,
        )

        self.w_ci = None
        self.w_cf = None
        self.w_co = None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.conv_x.weight)
        nn.init.orthogonal_(self.conv_h.weight)

        if self.conv_x.bias is not None:
            nn.init.constant_(self.conv_x.bias, 0)
            self.conv_x.bias.data[self.hidden_ch : 2 * self.hidden_ch].fill_(1.0)

    def _init_peepholes(self, height, width, device, dtype):
        shape = (1, self.hidden_ch, height, width)
        self.w_ci = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
        self.w_cf = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))
        self.w_co = nn.Parameter(torch.zeros(shape, device=device, dtype=dtype))

    def initialize_peepholes(self, height, width, device, dtype):
        if self.w_ci is None:
            self._init_peepholes(height, width, device, dtype)
            return

        if self.w_ci.shape[-2:] != (height, width):
            raise ValueError(
                "ConvLSTMCell peepholes were initialized for "
                f"{self.w_ci.shape[-2:]} but received {(height, width)}"
            )

    def init_state(self, x):
        batch_size, _, height, width = x.shape
        if self.w_ci is None or self.w_ci.shape[-2:] != (height, width):
            self._init_peepholes(height, width, x.device, x.dtype)

        h = torch.zeros(
            batch_size, self.hidden_ch, height, width, device=x.device, dtype=x.dtype
        )
        c = torch.zeros_like(h)
        return h, c

    def forward(self, x, state):
        h_prev, c_prev = state
        if self.w_ci is None or self.w_ci.shape[-2:] != x.shape[-2:]:
            self._init_peepholes(x.shape[-2], x.shape[-1], x.device, x.dtype)

        gates = self.conv_x(x) + self.conv_h(h_prev)
        i, f, g, o = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i + self.w_ci * c_prev)
        f = torch.sigmoid(f + self.w_cf * c_prev)
        g = torch.tanh(g)

        c_next = f * c_prev + i * g
        o = torch.sigmoid(o + self.w_co * c_next)
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


class ConvLSTMForecaster(nn.Module):
    """Encoding-forecasting ConvLSTM kept compatible with the current training loop."""

    def __init__(self, in_ch=1, hidden_ch=None, kernel_size=(3, 3), num_layers=3):
        super().__init__()

        if hidden_ch is None:
            hidden_ch = [64, 64, 64]
        if len(hidden_ch) < num_layers:
            raise ValueError("hidden_ch must provide at least num_layers entries")

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)

        self.in_ch = in_ch
        self.num_layers = num_layers
        self.hidden_ch = list(hidden_ch[:num_layers])

        self.encoder_cells = self._build_stack(in_ch, self.hidden_ch, kernel_size)
        self.forecaster_cells = self._build_stack(in_ch, self.hidden_ch, kernel_size)

        total_hidden = sum(self.hidden_ch)
        self.decoder = nn.Sequential(
            nn.Conv2d(total_hidden, in_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    @staticmethod
    def _build_stack(in_ch, hidden_ch, kernel_size):
        cells = []
        for layer_idx, hidden_dim in enumerate(hidden_ch):
            layer_input = in_ch if layer_idx == 0 else hidden_ch[layer_idx - 1]
            cells.append(ConvLSTMCell(layer_input, hidden_dim, kernel_size, bias=True))
        return nn.ModuleList(cells)

    def _run_stack(self, cur_input, states, cell_stack):
        next_states = []
        hidden_outputs = []

        for layer_idx, cell in enumerate(cell_stack):
            h_next, c_next = cell(cur_input, states[layer_idx])
            next_states.append((h_next, c_next))
            hidden_outputs.append(h_next)
            cur_input = h_next

        return next_states, hidden_outputs

    def initialize_for_input(self, x):
        _, _, _, height, width = x.shape
        for cell in list(self.encoder_cells) + list(self.forecaster_cells):
            cell.initialize_peepholes(height, width, x.device, x.dtype)

    def forward(self, x, t_out, teacher_forcing=0.0, y=None):
        _, t_in, _, _, _ = x.shape

        encoder_states = []
        cur_input = x[:, 0]
        for layer_idx, cell in enumerate(self.encoder_cells):
            state_input = cur_input if layer_idx == 0 else encoder_states[-1][0]
            encoder_states.append(cell.init_state(state_input))

        for t in range(t_in):
            cur_input = x[:, t]
            encoder_states, _ = self._run_stack(cur_input, encoder_states, self.encoder_cells)

        forecast_states = []
        last_hidden = x[:, 0]
        for layer_idx, cell in enumerate(self.forecaster_cells):
            hidden_shape_source = last_hidden if layer_idx == 0 else forecast_states[-1][0]
            _ = cell.init_state(hidden_shape_source)
            h_enc, c_enc = encoder_states[layer_idx]
            forecast_states.append((h_enc, c_enc))

        preds = []
        prev_pred = x[:, -1]

        for t in range(t_out):
            if self.training and y is not None and torch.rand((), device=x.device) < teacher_forcing:
                cur_input = y[:, t]
            else:
                cur_input = prev_pred

            forecast_states, hidden_outputs = self._run_stack(
                cur_input, forecast_states, self.forecaster_cells
            )
            pred = self.decoder(torch.cat(hidden_outputs, dim=1))
            preds.append(pred)
            prev_pred = pred

        return torch.stack(preds, dim=1)
