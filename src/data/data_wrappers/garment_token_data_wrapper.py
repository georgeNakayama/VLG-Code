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
    stop_token: int = 142
    class_tokens: List = field(default_factory=lambda: [128, 129, 130, 131, 132, 133, 134, 135, 136, 137])

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
        stop_token: int = 142,
        class_tokens: List = [128, 129, 130, 131, 132, 133, 134, 135, 136, 137]
    ):
        self.B = batch_size
        self.T = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        self.tokens =  torch.tensor(np.load(token_dir), dtype=torch.long)
        self.descriptors = None
        self.stop_token = stop_token
        self.class_tokens = class_tokens
        self.masked_tokens = class_tokens + [stop_token]
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
        x = buf
        x[:, -1] = self.stop_token # last token is always a stop token
        y = F.pad(buf[:, 1:], (0, 1), value=self.stop_token) # shift y by one position to the right
        mask = np.isin(y, self.masked_tokens)
        y[mask] = self.stop_token # replace masked tokens with stop token
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
        stop_token: int = 142,
        class_tokens: List = [128, 129, 130, 131, 132, 133, 134, 135, 136, 137],
        ):
        self.train: DataLoaderLite = DataLoaderLite(token_dir, batch_size, sequence_length, process_rank, num_processes, 'train', stop_token, class_tokens)
        self.validation: DataLoaderLite = DataLoaderLite(token_dir, batch_size, sequence_length, process_rank, num_processes, 'val', stop_token, class_tokens)
        self.test: DataLoaderLite = DataLoaderLite(token_dir, batch_size, sequence_length, process_rank, num_processes, 'val', stop_token, class_tokens)
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
        stop_token: int = 142,
        garment_start: int = 138, 
        garment_end: int = 141, 
        panel_start: int = 139, 
        panel_end: int = 140,
        bin_size: int = 128,
        class_tokens: List = [128, 129, 130, 131, 132, 133, 134, 135, 136, 137],
        ):
        self.loaders = DataLoader(token_dir, batch_size, sequence_length, process_rank, num_processes, stop_token, class_tokens)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.panel_classification= panel_classification
        self.filter_by_params = filter_by_params    
        self.stop_token = stop_token
        self.class_tokens = class_tokens
        self.garment_start = garment_start
        self.garment_end = garment_end
        self.panel_start = panel_start
        self.panel_end = panel_end
        self.bin_size = bin_size