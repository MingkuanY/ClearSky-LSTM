"""
ClearSky-LSTM: a ConvLSTM–UNet hybrid for precipitation nowcasting.

- each input frame is encoded by a shared SmaAt-UNet encoder
- a multi-layer ConvLSTM operates on the sequence of bottleneck features to
model temporal evolution
- for each forecast step, the ConvLSTM hidden state is decoded by a shared SmaAt-UNet decoder using skip connections averaged across the input window

Can interface like this:
    model = ClearSkyLSTM()
    pred = model(x, t_out=6)
    
    # x: [B, T_in, 1, H, W]
    # pred: [B, T_out, 1, H, W], in [0,1]
"""

import torch
import torch.nn as nn

from .smaat_unet import CBAM, DoubleConv, SmaAtUNetDecoder, SmaAtUNetEncoder
from .conv_lstm import ConvLSTMCell


class ClearSkyLSTM(nn.Module):
    """
    base - base channel width, defaults to 64, can set to 32 to reduce memory usage
    """

    def __init__(self, in_ch: int = 1, base: int = 64, lstm_layers: int = 1):
        super().__init__()

        self.bottleneck_ch = base * 8

        self.enc1 = SmaAtUNetEncoder(in_ch,      base)
        self.enc2 = SmaAtUNetEncoder(base,       base * 2)
        self.enc3 = SmaAtUNetEncoder(base * 2,   base * 4)
        self.enc4 = SmaAtUNetEncoder(base * 4,   base * 8)

        self.bottleneck_conv = DoubleConv(base * 8, base * 8)
        self.bottleneck_cbam = CBAM(base * 8)

        self.lstm_cells = nn.ModuleList([
            ConvLSTMCell(
                in_ch=base * 8,
                hidden_ch=base * 8,
            )
            for _ in range(lstm_layers)
        ])

        self.dec4 = SmaAtUNetDecoder(base * 8, base * 8, base * 4)
        self.dec3 = SmaAtUNetDecoder(base * 4, base * 4, base * 2)
        self.dec2 = SmaAtUNetDecoder(base * 2, base * 2, base)
        self.dec1 = SmaAtUNetDecoder(base,     base,     base)

        self.out_conv = nn.Sequential(
            nn.Conv2d(base, in_ch, kernel_size=1),
            nn.Sigmoid(),
        )

    def _encode(self, frame: torch.Tensor):
        x, skip1 = self.enc1(frame)
        x, skip2 = self.enc2(x)
        x, skip3 = self.enc3(x)
        x, skip4 = self.enc4(x)
        z = self.bottleneck_conv(x)
        z = self.bottleneck_cbam(z)
        return z, (skip1, skip2, skip3, skip4)

    def _decode(
        self,
        h: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        skip1, skip2, skip3, skip4 = skips
        x = self.dec4(h, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        return self.out_conv(x)

    def forward(
        self,
        x: torch.Tensor,
        t_out: int,
        teacher_forcing: float = 0.0,
        y: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, T_in, C, H, W = x.shape
        device = x.device
        dtype  = x.dtype

        z0, skips0 = self._encode(x[:, 0])
        lstm_states = [cell.init_state(z0) for cell in self.lstm_cells]

        skips_sum: list[torch.Tensor] = [s.clone() for s in skips0]

        inp = z0
        for layer, cell in enumerate(self.lstm_cells):
            h, c = cell(inp, lstm_states[layer])
            lstm_states[layer] = (h, c)
            inp = h

        for t in range(1, T_in):
            z, skips_t = self._encode(x[:, t])

            for i, s in enumerate(skips_t):
                skips_sum[i] = skips_sum[i] + s

            inp = z
            for layer, cell in enumerate(self.lstm_cells):
                h, c = cell(inp, lstm_states[layer])
                lstm_states[layer] = (h, c)
                inp = h

        skips_avg = tuple(s / T_in for s in skips_sum)

        zero_z = torch.zeros(B, self.bottleneck_ch,
                             H // 16, W // 16,
                             device=device, dtype=dtype)
        preds: list[torch.Tensor] = []

        for t in range(t_out):
            if (teacher_forcing > 0.0
                    and self.training
                    and y is not None
                    and torch.rand(1, device=device).item() < teacher_forcing):
                lstm_inp, _ = self._encode(y[:, t])
            else:
                lstm_inp = zero_z

            inp = lstm_inp
            for layer, cell in enumerate(self.lstm_cells):
                h, c = cell(inp, lstm_states[layer])
                lstm_states[layer] = (h, c)
                inp = h

            pred = self._decode(lstm_states[-1][0], skips_avg)
            preds.append(pred)

        return torch.stack(preds, dim=1)
