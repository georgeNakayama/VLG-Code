import torch
import torch.nn as nn
from typing import Optional, Literal, List

class BaseEncoding(nn.Module):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        include_input: bool = True,
    ) -> None:
        super().__init__()
        self.include_input = include_input
        self.in_dim = in_dim


    def get_out_dim(self) -> int:
        pass

    def forward(
        self,
        in_tensor: torch.Tensor,
    ) -> torch.Tensor:
        pass

class SinusoidalEncoding(BaseEncoding):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        num_frequencies: int,
        min_freq_exp: float,
        max_freq_exp: float,
        include_input: bool = True,
    ) -> None:
        super().__init__(in_dim, include_input)
        self.num_frequencies = num_frequencies
        self.min_freq = min_freq_exp
        self.max_freq = max_freq_exp


    def get_out_dim(self) -> int:
        if self.in_dim is None:
            raise ValueError("Input dimension has not been set")
        if self.num_frequencies == 0:
            return self.in_dim
        out_dim = self.in_dim * self.num_frequencies * 2
        if self.include_input:
            out_dim += self.in_dim
        return out_dim

    def forward(
        self,
        in_tensor: torch.Tensor,
    ) -> torch.Tensor:
        """Calculates NeRF encoding. If covariances are provided the encodings will be integrated as proposed
            in mip-NeRF.

        Args:
            in_tensor: For best performance, the input tensor should be between 0 and 1.
            covs: Covariances of input points.
        Returns:
            Output values will be between -1 and 1
        """
        if self.num_frequencies == 0:
            return in_tensor if self.include_input else torch.zeros_like(in_tensor)
        dtype = in_tensor.dtype
        scaled_in_tensor = 2 * torch.pi * in_tensor  # scale to [0, 2pi]
        freqs = 2 ** torch.linspace(self.min_freq, self.max_freq, self.num_frequencies, device=in_tensor.device)
        scaled_inputs = scaled_in_tensor[..., None] * freqs  # [..., "input_dim", "num_scales"]
        scaled_inputs = scaled_inputs.view(*scaled_inputs.shape[:-2], -1)  # [..., "input_dim" * "num_scales"]

        encoded_inputs = torch.sin(torch.cat([scaled_inputs, scaled_inputs + torch.pi / 2.0], dim=-1)).to(dtype)
        if self.include_input:
            encoded_inputs = torch.cat([encoded_inputs, in_tensor], dim=-1)
        return encoded_inputs
    
class DiscreteEncoding(BaseEncoding):
    """Multi-scale sinusoidal encodings. Support ``integrated positional encodings`` if covariances are provided.
    Each axis is encoded with frequencies ranging from 2^min_freq_exp to 2^max_freq_exp.

    Args:
        in_dim: Input dimension of tensor
        num_frequencies: Number of encoded frequencies per axis
        min_freq_exp: Minimum frequency exponent
        max_freq_exp: Maximum frequency exponent
        include_input: Append the input coordinate to the encoding
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        bin_num: int,
        max_bounds: List[float],
        min_bounds: List[float],
    ) -> None:
        super().__init__(in_dim, False)
        self.bin_num = bin_num
        self.bounds = min_bounds + max_bounds
        assert len(self.bounds) == in_dim * 2
        self.out_dim = out_dim
        self.embedding_proj = nn.Embedding(bin_num*in_dim, out_dim)

    def get_out_dim(self) -> int:
        return self.out_dim
    def forward(self, in_tensor: torch.Tensor):
        assert in_tensor.shape[-1] == self.in_dim
        bounds = torch.tensor(self.bounds).to(in_tensor)
        ids = (in_tensor - bounds[:self.in_dim]) / (bounds[self.in_dim:] - bounds[:self.in_dim]) * (self.bin_num - 1)
        ids = ids.long()
        ids = ids.clamp(0, self.bin_num-1) + torch.arange(self.in_dim).to(ids) * self.bin_num
        embedding = self.embedding_proj(ids).mean(-2)
        return embedding