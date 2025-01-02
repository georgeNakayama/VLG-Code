# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""

import copy
from typing import Optional, List, Literal

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from .transformer import TransformerDecoder, TransformerDecoderLayer

class EdgeTransformer(TransformerDecoder):
    def __init__(
        self, 
        d_model: int, 
        nhead: int, 
        num_layers: int,
        dim_feedforward: int=2048, 
        dropout: float=0.1, 
        activation: Literal["relu", "gelu", "glu"]='relu', 
        normalize_before: bool=False,
        return_intermediate: bool=True
        ):  
        edge_decoder_layer = TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout, activation, normalize_before)
        decoder_norm = nn.LayerNorm(d_model)
        super().__init__(edge_decoder_layer, num_layers, decoder_norm, return_intermediate=return_intermediate)