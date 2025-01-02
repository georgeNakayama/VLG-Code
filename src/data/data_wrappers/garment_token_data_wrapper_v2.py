import numpy as np
import os 
import torch 
from dataclasses import dataclass, field
from typing import Literal, List, Optional
from argparse import Namespace
from torch.utils.data import DataLoader
from data.data_loaders.infinite_loader import InfiniteDataLoader
import logging 
import torch.nn.functional as F
import time
import random
import json
log = logging.getLogger(__name__)
from ..datasets.dataset_garment_as_token import GarmentTokenDataset, SpecialTokens
from ..datasets.panel_configs import StandardizeConfig
from .data_split_config import DataSplitConfig

def collate_fn(batch, mask_token):
    len_to_pad = max(len(x) for x in batch)
    out_input = torch.ones(len(batch), len_to_pad, dtype=torch.long) * mask_token
    out_target = torch.ones(len(batch), len_to_pad, dtype=torch.long) * mask_token
    for i, x in enumerate(batch):
        out_input[i, :len(x)] = x
        out_target[i, :len(x)-1] = x[1:]
    return out_input, out_target

@dataclass 
class Loaders: 
    train: InfiniteDataLoader = None
    validation: DataLoader = None
    test: DataLoader = None
    full: DataLoader = None

@dataclass 
class GarmentTokenDataWrapperV2Config:
    batch_size: int
    

class GarmentTokenDataWrapperV2:
    def __init__(
        self, 
        dataset: GarmentTokenDataset,
        data_split: DataSplitConfig,
        batch_size: int, 
        process_rank: int, 
        num_processes: int, 
        output_dir: str
        ):
        self.dataset = dataset
        self.loaders = Loaders()
        self.data_split = data_split
        self.batch_size = batch_size
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.output_dir = output_dir
        self.load_split()
        
    def load_split(self, shuffle_train: bool = True, random_seed: Optional[int] = None):
        """Get the split by provided parameters. Can be used to reproduce splits on the same dataset.
            NOTE this function re-initializes torch random number generator!
        """
        
        if random_seed is None:
            random_seed = int(time.time())
        # init for all libs =)
        torch.manual_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        # if file is provided
        if self.data_split.load_from_split_file is not None:
            log.info('Loading data split from {}'.format(self.data_split.load_from_split_file))
            with open(self.data_split.load_from_split_file, 'r') as f_json:
                split_dict = json.load(f_json)
            self.training, self.validation, self.test = self.dataset.split_from_dict(
                split_dict)
        else:
            log.info('Loading data split from split config: {}: valid per type {} / test per type {}'.format(
                self.data_split.type, self.data_split.valid_per_type, self.data_split.test_per_type))
            self.training, self.validation, self.test = self.dataset.random_split_by_dataset(
                self.data_split.valid_per_type, 
                self.data_split.test_per_type,
                self.data_split.type, 
                self.data_split.split_on)

            self.new_loaders()  # s.t. loaders could be used right away
            
            log.info('Dataset split: {} / {} / {}'.format(
                len(self.training) if self.training else None, 
                len(self.validation) if self.validation else None, 
                len(self.test) if self.test else None))
            
            self.get_data_lists()
        
        return self.training, self.validation, self.test
    
    def new_loaders(self):
        """Create loaders for current data split. Note that result depends on the random number generator!
        
            if the data split was not specified, only the 'full' loaders are created
        """
        full_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset, shuffle=False, num_replicas=self.num_processes, rank=self.process_rank)
        self.loaders.full = DataLoader(self.dataset, self.batch_size, num_workers=0, pin_memory=True, sampler=full_sampler)
        if self.validation is not None and self.test is not None:

            train_sampler = torch.utils.data.distributed.DistributedSampler(self.training, drop_last=True, shuffle=True, num_replicas=self.num_processes, rank=self.process_rank)
            self.loaders.train = InfiniteDataLoader(self.training, self.batch_size, 
                                        pin_memory=True,
                                        num_workers=12,
                                        collate_fn=lambda x: collate_fn(x, self.dataset.stop_token),    
                                        sampler=train_sampler)
            
            self.loaders.validation = DataLoader(self.validation, self.batch_size, collate_fn=lambda x: collate_fn(x, self.dataset.stop_token), num_workers=12)
            self.loaders.test = DataLoader(self.test, self.batch_size, collate_fn=lambda x: collate_fn(x, self.dataset.stop_token), num_workers=12)
        return self.loaders.train, self.loaders.validation, self.loaders.test
    
    def get_data_lists(self):

        if self.data_split.load_from_split_file is not None and  os.path.exists(self.data_split.load_from_split_file):
            log.info('Load Dataset split: {} '.format(self.data_split.load_from_split_file))
        else:
            data_lists = {"train": [], "validation": [], "test": []}
            for name, split in {"train":self.training, "validation": self.validation, "test": self.test}.items():
                split_idxs = split.indices
                for idx in split_idxs:
                    datanames, _ = self.dataset.get_item_infos(idx)
                    data_lists[name].append(datanames)

            save_path = os.path.join(self.output_dir, "data_split.json")
            json.dump(data_lists, open(save_path, "w"), indent=2)
            self.data_split.load_from_split_file = save_path

            log.info('Save Dataset split: {} '.format(self.data_split.load_from_split_file))