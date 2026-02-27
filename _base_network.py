import torch
import torch.nn as nn
import torch.nn.functional as F

""" LSTM cell unit """
class ConvLSTMCell(nn.Module):

    def __init__(self, in_ch, hidden_ch, kernel_size=3, bias=True):
        """ Create a new LSTM cell """
        super().__init__()
        padding = kernel_size // 2
        self.hidden_ch = hidden_ch
        self.conv = nn.Conv2d(in_ch + hidden_ch, 4 * hidden_ch, kernel_size, padding=padding, bias=bias)

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
class ConvLSTMForecaster(nn.Module):

    def __init__(self, in_ch=1, hidden_ch=64, num_layers=2, out_ch=1):
        """ Initialize a new ConvLSTM class """
        super().__init__()

        self.num_layers = num_layers
        self.cells = nn.ModuleList()

        for i in range(num_layers):
            self.cells.append(ConvLSTMCell(in_ch if i == 0 else hidden_ch, hidden_ch))

        self.to_frame = nn.Conv2d(hidden_ch, out_ch, kernel_size=1)

    def forward(self, x, t_out, teacher_forcing=None, y=None):
        """
        x: [B, T_in, C, H, W]
        y: optional ground truth future frames [B, T_out, C, H, W] if teacher forcing
        teacher_forcing: float in [0,1] or None
        """
        B, T_in, C, H, W = x.shape
        device = x.device

        # init states
        states = []
        x0 = x[:, 0]
        for cell in self.cells:
            states.append(cell.init_state(x0))

        # encode past
        for t in range(T_in):
            inp = x[:, t]
            for l, cell in enumerate(self.cells):
                h, c = states[l]
                h, c = cell(inp, (h, c))
                states[l] = (h, c)
                inp = h  

        # forecast
        preds = []
        prev = self.to_frame(states[-1][0])  # initial guess based on final hidden
        for t in range(t_out):
            if teacher_forcing is not None and y is not None:
                use_gt = (torch.rand(1, device=device) < teacher_forcing).item()
                inp_frame = y[:, t] if use_gt else prev
            else:
                inp_frame = prev

            inp = inp_frame
            for l, cell in enumerate(self.cells):
                h, c = states[l]
                h, c = cell(inp, (h, c))
                states[l] = (h, c)
                inp = h

            prev = self.to_frame(states[-1][0])
            preds.append(prev)

        return torch.stack(preds, dim=1)  # [B, T_out, C, H, W]
    
