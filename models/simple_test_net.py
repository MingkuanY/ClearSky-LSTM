import torch
import torch.nn as nn

from .conv_lstm import ConvLSTMCell
from .smaat_unet import CBAM, DoubleConv, SmaAtUNetDecoder, SmaAtUNetEncoder


class SimpleTestNet(nn.Module):
    """
    U-Net-style encoder-decoder with a ConvLSTM bottleneck.

    Each input frame is encoded by a shared encoder. The encoded bottleneck
    sequence is processed by stacked ConvLSTM cells, and the resulting hidden
    states are decoded into future frames using shared decoder weights.
    """

    def __init__(self, in_ch: int = 1, base: int = 32, lstm_layers: int = 1):
        super().__init__()

        self.in_ch = in_ch
        self.base = base
        self.bottleneck_ch = base * 8

        self.enc1 = SmaAtUNetEncoder(in_ch, base)
        self.enc2 = SmaAtUNetEncoder(base, base * 2)
        self.enc3 = SmaAtUNetEncoder(base * 2, base * 4)
        self.enc4 = SmaAtUNetEncoder(base * 4, base * 8)

        self.bottleneck_conv = DoubleConv(base * 8, base * 8)
        self.bottleneck_cbam = CBAM(base * 8)

        self.lstm_cells = nn.ModuleList(
            [
                ConvLSTMCell(
                    in_ch=base * 8,
                    hidden_ch=base * 8,
                )
                for _ in range(lstm_layers)
            ]
        )

        self.dec4 = SmaAtUNetDecoder(base * 8, base * 8, base * 4)
        self.dec3 = SmaAtUNetDecoder(base * 4, base * 4, base * 2)
        self.dec2 = SmaAtUNetDecoder(base * 2, base * 2, base)
        self.dec1 = SmaAtUNetDecoder(base, base, base)

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
        bottleneck: torch.Tensor,
        skips: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        skip1, skip2, skip3, skip4 = skips
        x = self.dec4(bottleneck, skip4)
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
        del teacher_forcing, y

        batch_size, t_in, _, height, width = x.shape
        device = x.device
        dtype = x.dtype

        z0, skips0 = self._encode(x[:, 0])
        lstm_states = [cell.init_state(z0) for cell in self.lstm_cells]
        skip_sums = [skip.clone() for skip in skips0]

        inp = z0
        for layer_idx, cell in enumerate(self.lstm_cells):
            h, c = cell(inp, lstm_states[layer_idx])
            lstm_states[layer_idx] = (h, c)
            inp = h

        for t in range(1, t_in):
            z_t, skips_t = self._encode(x[:, t])
            for i, skip in enumerate(skips_t):
                skip_sums[i] = skip_sums[i] + skip

            inp = z_t
            for layer_idx, cell in enumerate(self.lstm_cells):
                h, c = cell(inp, lstm_states[layer_idx])
                lstm_states[layer_idx] = (h, c)
                inp = h

        shared_skips = tuple(skip_sum / t_in for skip_sum in skip_sums)
        zero_bottleneck = torch.zeros(
            batch_size,
            self.bottleneck_ch,
            height // 16,
            width // 16,
            device=device,
            dtype=dtype,
        )

        preds = []
        for _ in range(t_out):
            inp = zero_bottleneck
            for layer_idx, cell in enumerate(self.lstm_cells):
                h, c = cell(inp, lstm_states[layer_idx])
                lstm_states[layer_idx] = (h, c)
                inp = h

            preds.append(self._decode(lstm_states[-1][0], shared_skips))

        return torch.stack(preds, dim=1)
