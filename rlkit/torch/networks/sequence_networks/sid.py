"""
Network that forms encoding based on statistics, integral, and derivative at each step.

Author: Ian Char
Date: December 12, 2022
"""
import torch

import rlkit.torch.pytorch_util as ptu
from rlkit.torch.networks.mlp import Mlp
from rlkit.torch.core import PyTorchModule


class SIDNet(PyTorchModule):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        statistic_size: int,
        lookback_len: int,
        encoder_width: int,
        encoder_depth: int,
        decoder_width: int,
        decoder_depth: int,
    ):
        """Constructor.

        Args:
            input_size: Size of the input.
            output_size: Size of the output.
            statistic_size: Size of the statistic.
            lookback_len: The lookback to consider for the integral.
            encoder_width: Width of the hidden units in the encoder.
            encoder_depth: Number of hidden units in the encoder.
            decoder_width: Width of the hidden units in the decoder.
            decoder_depth: Number of hidden units in the decoder.
        """
        self.lookback_len = lookback_len
        self.encoder = Mlp(
            input_size=input_size,
            output_size=statistic_size,
            hidden_sizes=[encoder_width for _ in range(encoder_depth)],
        )
        self.decoder = Mlp(
            input_size=statistic_size * 3,
            output_size=output_size,
            hidden_sizes=[decoder_width for _ in range(decoder_depth)],
        )

    def forward(self, net_in, **kwargs):
        """Forward pass.

        Args:
            net_in: This should have shape (batch_size, L, input_dim)

        Returns: Output with shape (batch_size, L, output_dim)
        """
        stats = self.encoder(net_in)
        padded_stats = torch.cat([
            ptu.zeros(stats.shape[0], self.lookback_len - 1, stats.shape[-1]),
            stats,
        ], dim=1)
        integral_stats = torch.cat([
            torch.mean(padded_stats[:, i-self.lookback_len:i], dim=1, keepdim=True)
            for i in range(self.lookback_len, padded_stats.shape[1])
        ], dim=1)
        diff_stats = torch.cat([
            padded_stats[:, [i]] - padded_stats[:, [i - 1]]
            for i in range(self.lookback_len, padded_stats.shape[1])
        ])
        sid_out = torch.cat([
            stats,
            integral_stats,
            diff_stats,
        ], dim=-1)
        return self.decoder(sid_out)


class FlattenSidNet(SIDNet):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """
    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=-1)
        return super().forward(flat_inputs, **kwargs)
