import numpy as np
import os 
import torch 
from dataclasses import dataclass
from typing import Literal
from argparse import Namespace
import logging 
log = logging.getLogger(__name__)
@dataclass 
class TextDataWrapperConfig:
    batch_size: int
    sequence_length: int
    root_dir: str

def load_tokens(filename):
    npt = np.load(filename)
    npt = npt.astype(np.int32) # added after video
    ptt = torch.tensor(npt, dtype=torch.long)
    return ptt

class DataLoaderLite:
    def __init__(
        self, 
        root_dir: str, 
        batch_size: int, 
        sequence_length: int, 
        process_rank: int, 
        num_processes: int, 
        split: Literal['train', 'val']):
        self.B = batch_size
        self.T = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {'train', 'val'}

        # get the shard filenames
        data_root = root_dir
        shards = os.listdir(data_root)
        shards = [s for s in shards if split in s]
        shards = sorted(shards)
        shards = [os.path.join(data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"no shards found for split {split}"
        log.info(f"found {len(shards)} shards for split {split}")
        self.reset()

    def reset(self):
        # state, init at shard zero
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self):
        B, T = self.B, self.T
        buf = self.tokens[self.current_position : self.current_position+B*T+1]
        x = (buf[:-1]).view(B, T) # inputs
        y = (buf[1:]).view(B, T) # targets
        # advance the position in the tensor
        self.current_position += B * T * self.num_processes
        # if loading the next batch would be out of bounds, advance to next shard
        if self.current_position + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_position = B * T * self.process_rank
        return x, y
    def __len__(self):
        return len(self.tokens) // (self.B * self.T)

class DataLoader:
    def __init__(self, root_dir: str, batch_size: int, sequence_length: int, process_rank: int, num_processes: int):
        self.train: DataLoaderLite = DataLoaderLite(root_dir, batch_size, sequence_length, process_rank, num_processes, 'train')
        self.validation: DataLoaderLite = DataLoaderLite(root_dir, batch_size, sequence_length, process_rank, num_processes, 'val')
        self.test: DataLoaderLite = DataLoaderLite(root_dir, batch_size, sequence_length, process_rank, num_processes, 'val')
class TextDataWrapper:
    def __init__(self, root_dir: str, batch_size: int, sequence_length: int, process_rank: int, num_processes: int):
        self.loaders = DataLoader(root_dir, batch_size, sequence_length, process_rank, num_processes)
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.process_rank = process_rank
        self.num_processes = num_processes