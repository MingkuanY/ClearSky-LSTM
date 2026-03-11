import torch
import torch.nn as nn
import torch.nn.functional as F

""" LSTM cell unit """
class ConvLSTMCell(nn.Module):

    def __init__(self, in_ch, hidden_ch, kernel_size=3, bias=True):
        """ Create a new LSTM cell """
        super().__init__()

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        padding = kernel_size[0] // 2
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=padding, bias=bias)
        nn.init.orthogonal_(self.conv.weight)
        if self.conv.bias is not None:
            nn.init.constant_(self.conv.bias, 0)
            # Set forget gate bias to 1.0 to help gradient flow early on
            self.conv.bias.data[hidden_ch:2*hidden_ch].fill_(1.0)

    def init_state(self, x):
        """ Define initial state of LSTM cell 
        - x: x is a tensor with shape [B, Cin, H, W] 
            - B is batch size, C_in is no. input channels, HxW is HeightxWidth of inpt image
        """
        B, C_in, H, W = x.shape

        h = torch.zeros(B, self.hidden_ch, H, W, device=x.device, dtype=x.dtype)
        c = torch.zeros(B, self.hidden_ch, H, W, device=x.device, dtype=x.dtype)

        return h, c

    def forward(self, x, state):
        """ Forward pass through LSTM:
        - x: is a tensor with shape [B, Cin, H, W] 
        - state: [h, c] where h is hidden state, c is cell state
        """
        h, c = state  # each: [B, Ch, H, W]
        cat = torch.cat([x, h], dim=1)
        gates = self.conv(cat)
        i, f, o, g = torch.chunk(gates, 4, dim=1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        o = torch.sigmoid(o)
        g = torch.tanh(g)

        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next
    

""" Baseline ConvLSTM network """
import torch
import torch.nn as nn

class ConvLSTMForecaster(nn.Module):
    def __init__(self, in_ch=1, hidden_ch=[64, 64, 64], kernel_size=(3, 3), num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_ch = hidden_ch
        
        # 1. Multi-layer Cell List
        cell_list = []
        for i in range(num_layers):
            cur_input_dim = in_ch if i == 0 else hidden_ch[i-1]
            cell_list.append(ConvLSTMCell(cur_input_dim, hidden_ch[i], kernel_size, bias=True))
        self.cell_list = nn.ModuleList(cell_list)

        # 2. Chunky Decoder (3 layers instead of 1) - should help translate high-dimensional hidden states back to diverse radar maps
        self.decoder = nn.Sequential(
            nn.Conv2d(hidden_ch[-1], hidden_ch[-1] // 2, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden_ch[-1] // 2, in_ch, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x, t_out, teacher_forcing=0.0, y=None):
        B, T_in, C, H, W = x.shape
        device = x.device
        
        # Initialize hidden states for all layers
        hidden_states = []
        for i in range(self.num_layers):
            h = torch.zeros(B, self.hidden_ch[i], H, W, device=device)
            c = torch.zeros(B, self.hidden_ch[i], H, W, device=device)
            hidden_states.append((h, c))

        # --- ENCODER ---
        # Process all T_in frames through the stack
        for t in range(T_in):
            cur_input = x[:, t]
            for i in range(self.num_layers):
                h, c = self.cell_list[i](cur_input, hidden_states[i])
                hidden_states[i] = (h, c)
                cur_input = h # Next layer's input is current layer's hidden state

        # --- FORECASTER ---
        preds = []
        last_frame = x[:, -1] # most recent obs
        prev_pred = self.decoder(hidden_states[-1][0])

        for t in range(t_out):
            # teacher forcing (can turn on via CLI if we want)
            if self.training and torch.rand(1) < teacher_forcing and y is not None:
                cur_input = y[:, t]
            else:
                cur_input = prev_pred
            
            # Pass through stack
            for i in range(self.num_layers):
                h, c = self.cell_list[i](cur_input, hidden_states[i])
                hidden_states[i] = (h, c)
                cur_input = h
            
            # Decode and apply residual connection (pred = delta + last obs)
            delta = self.decoder(hidden_states[-1][0])
            prev_pred = delta 
            preds.append(prev_pred)

        return torch.stack(preds, dim=1)