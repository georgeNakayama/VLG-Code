import numpy as np
import os 
import torch 
from dataclasses import dataclass, field
from typing import Literal, List
from argparse import Namespace
import logging 
import torch.nn.functional as F
log = logging.getLogger(__name__)
@dataclass 
class TextDataWrapperConfig:
    batch_size: int
    sequence_length: int
    token_dir: str
    eot_token: int = 50256
    panel_start: int = 50257
    panel_end: int = 50258

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(
        self, 
        token_dir: str, 
        batch_size: int, 
        sequence_length: int, 
        process_rank: int, 
        num_processes: int, 
        split: Literal['train', 'val'],
        stop_token: int = 50256,
    ):
        self.B = batch_size
        self.T = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        self.tokens =  torch.tensor(np.load(token_dir), dtype=torch.long)
        self.stop_token = stop_token
        self.masked_tokens = [stop_token]
        assert self.T == self.tokens.shape[1], f"sequence length {self.T} does not match token shape {self.tokens.shape[1]}"
        self.num_sequences = self.tokens.shape[0]
        if split == 'train':
            self.tokens = self.tokens[:int(0.95*self.num_sequences)]
        else:
            self.tokens = self.tokens[int(0.95*self.num_sequences):]
        log.info(f"found {len(self.tokens)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_position = self.B * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B]
        x = buf[:, :-1]
        y = buf[:, 1:]
        # advance the position in the tensor
        self.current_position += B * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * self.num_processes + 1) > len(self.tokens):
            self.current_position = B * self.process_rank
        return x, y
    def __len__(self):
        return len(self.tokens) // self.B

class DataLoader:
    def __init__(
        self, 
        token_dir: str, 
        batch_size: int, 
        sequence_length: int, 
        process_rank: int, 
        num_processes: int,
        stop_token: int = 50256,
        ):
        self.train: DataLoaderLite = DataLoaderLite(token_dir, batch_size, sequence_length, process_rank, num_processes, 'train', stop_token)
        self.validation: DataLoaderLite = DataLoaderLite(token_dir, batch_size, sequence_length, process_rank, num_processes, 'val', stop_token)
        self.test: DataLoaderLite = DataLoaderLite(token_dir, batch_size, sequence_length, process_rank, num_processes, 'val', stop_token)
class GarmentTokenDataWrapper:
    def __init__(
        self, 
        token_dir: str, 
        batch_size: int, 
        sequence_length: int, 
        process_rank: int, 
        num_processes: int, 
        panel_classification: str = './assets/data_configs/panel_classes_condenced.json',
        filter_by_params: str = './assets/data_configs/param_filter.json',
        stop_token: int = 50256,
        panel_start: int = 50257,
        panel_end: int = 50258,
        bin_size: int = 128,
        ):
        self.loaders = DataLoader(token_dir, batch_size, sequence_length, process_rank, num_processes, stop_token)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.panel_classification= panel_classification
        self.filter_by_params = filter_by_params    
        self.panel_start = panel_start
        self.panel_end = panel_end
        self.stop_token = stop_token
        self.bin_size=bin_size