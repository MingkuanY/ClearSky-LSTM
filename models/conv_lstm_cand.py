import torch
import torch.nn as nn


class ConvLSTMCellCand(nn.Module):
    """ConvLSTM cell with peephole connections, closer to Shi et al. (2015)."""

    def __init__(self, in_ch, hidden_ch, kernel_size=3, bias=True, peephole=True):
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = (kernel_size[0] // 2, kernel_size[1] // 2)

        self.hidden_ch = hidden_ch
        self.peephole = peephole
        self.conv_x = nn.Conv2d(in_ch, 4 * hidden_ch, kernel_size, padding=padding, bias=bias)
        self.conv_h = nn.Conv2d(hidden_ch, 4 * hidden_ch, kernel_size, padding=padding, bias=False)

        nn.init.orthogonal_(self.conv_x.weight)
        nn.init.orthogonal_(self.conv_h.weight)
        if self.conv_x.bias is not None:
            nn.init.constant_(self.conv_x.bias, 0)
            self.conv_x.bias.data[hidden_ch:2 * hidden_ch].fill_(1.0)

        if peephole:
            self.W_ci = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))
            self.W_cf = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))
            self.W_co = nn.Parameter(torch.zeros(1, hidden_ch, 1, 1))
        else:
            self.register_parameter("W_ci", None)
            self.register_parameter("W_cf", None)
            self.register_parameter("W_co", None)

    def init_state(self, batch_size, height, width, device, dtype):
        h = torch.zeros(batch_size, self.hidden_ch, height, width, device=device, dtype=dtype)
        c = torch.zeros(batch_size, self.hidden_ch, height, width, device=device, dtype=dtype)
        return h, c

    def forward(self, x, state):
        h_prev, c_prev = state

        x_gates = self.conv_x(x)
        h_gates = self.conv_h(h_prev)

        x_i, x_f, x_o, x_g = torch.chunk(x_gates, 4, dim=1)
        h_i, h_f, h_o, h_g = torch.chunk(h_gates, 4, dim=1)

        if self.peephole:
            i = torch.sigmoid(x_i + h_i + self.W_ci * c_prev)
            f = torch.sigmoid(x_f + h_f + self.W_cf * c_prev)
        else:
            i = torch.sigmoid(x_i + h_i)
            f = torch.sigmoid(x_f + h_f)

        g = torch.tanh(x_g + h_g)
        c = f * c_prev + i * g

        if self.peephole:
            o = torch.sigmoid(x_o + h_o + self.W_co * c)
        else:
            o = torch.sigmoid(x_o + h_o)

        h = o * torch.tanh(c)
        return h, c


class ConvLSTMForecasterCand(nn.Module):
    """
    Encoding-forecasting ConvLSTM closer to Shi et al. (2015).

    Differences from the baseline implementation:
    - separate encoder and forecasting stacks
    - optional peephole connections in each cell
    - forecasting stack starts from copied encoder states
    - all forecasting hidden states are concatenated and projected with a 1x1 head
    - forecasting is driven by zero inputs rather than autoregressive frame feedback
    """

    def __init__(
        self,
        in_ch=1,
        hidden_ch=(64, 64, 64),
        kernel_size=(3, 3),
        num_layers=3,
        bias=True,
        peephole=True,
    ):
        super().__init__()

        if len(hidden_ch) < num_layers:
            raise ValueError("len(hidden_ch) must be at least num_layers")

        self.in_ch = in_ch
        self.hidden_ch = list(hidden_ch[:num_layers])
        self.num_layers = num_layers

        encoder_cells = []
        forecaster_cells = []
        for i in range(num_layers):
            cur_input_dim = in_ch if i == 0 else self.hidden_ch[i - 1]
            encoder_cells.append(
                ConvLSTMCellCand(cur_input_dim, self.hidden_ch[i], kernel_size, bias=bias, peephole=peephole)
            )
            forecaster_cells.append(
                ConvLSTMCellCand(cur_input_dim, self.hidden_ch[i], kernel_size, bias=bias, peephole=peephole)
            )

        self.encoder_cells = nn.ModuleList(encoder_cells)
        self.forecaster_cells = nn.ModuleList(forecaster_cells)
        self.head = nn.Sequential(
            nn.Conv2d(sum(self.hidden_ch), in_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def _init_states(self, batch_size, height, width, device, dtype, cells):
        return [cell.init_state(batch_size, height, width, device, dtype) for cell in cells]

    def forward(self, x, t_out, teacher_forcing=0.0, y=None):
        del teacher_forcing, y

        batch_size, t_in, _, height, width = x.shape
        device = x.device
        dtype = x.dtype

        encoder_states = self._init_states(batch_size, height, width, device, dtype, self.encoder_cells)

        for t in range(t_in):
            cur_input = x[:, t]
            for layer_idx, cell in enumerate(self.encoder_cells):
                h, c = cell(cur_input, encoder_states[layer_idx])
                encoder_states[layer_idx] = (h, c)
                cur_input = h

        forecast_states = [(h.clone(), c.clone()) for (h, c) in encoder_states]
        preds = []
        zero_input = torch.zeros(batch_size, self.in_ch, height, width, device=device, dtype=dtype)

        for _ in range(t_out):
            cur_input = zero_input
            forecast_hiddens = []

            for layer_idx, cell in enumerate(self.forecaster_cells):
                h, c = cell(cur_input, forecast_states[layer_idx])
                forecast_states[layer_idx] = (h, c)
                forecast_hiddens.append(h)
                cur_input = h

            pred = self.head(torch.cat(forecast_hiddens, dim=1))
            preds.append(pred)

        return torch.stack(preds, dim=1)
