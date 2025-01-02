import numpy as np
import os 
import torch 
from dataclasses import dataclass, field
from typing import Literal, List, Optional, Union, Callable
from argparse import Namespace
from torch.utils.data import DataLoader
from data.data_loaders.infinite_loader import InfiniteDataLoader
import logging 
import torch.nn.functional as F
import time
import random
import json
import hydra
import transformers
log = logging.getLogger(__name__)
from ..datasets.dataset_garment_as_token_qva import QVAGarmentTokenDataset
from ..datasets.dataset_garmentcodedata_token_qva import GarmentCodeDatasetQVA
from ..datasets.dataset_mm_benchmark import GarmentCodeMMBenchmark
from ..datasets.dataset_garmentcodedata_retarget import GarmentCodeDatasetResize
from ..datasets.dataset_dresscode_caption import DressCodeCaptionDataset
from ..datasets.panel_configs import StandardizeConfig
from .data_split_config import DataSplitConfig
from .collate_fns import collate_fn_default



@dataclass 
class GarmentImageDataWrapperConfig:
    data_split: DataSplitConfig
    output_dir: str
    

class GarmentImageDataWrapper:
    def __init__(
        self, 
        dataset: Union[QVAGarmentTokenDataset, GarmentCodeDatasetQVA, GarmentCodeMMBenchmark, GarmentCodeDatasetResize],
        data_split: DataSplitConfig,
        output_dir: str,
        collate_fn: str,
        ):
        self.dataset = dataset
        self.data_split = data_split
        self.output_dir = output_dir
        self.collate_fn = hydra.utils.get_method(collate_fn)
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
            training_datapoints, validation_datapoints, test_datapoints = self.dataset.split_from_dict(
                split_dict)
        else:
            log.info('Loading data split from split config: {}: valid per type {} / test per type {}'.format(
                self.data_split.type, self.data_split.valid_per_type, self.data_split.test_per_type))
            training_datapoints, validation_datapoints, test_datapoints = self.dataset.random_split(
                self.data_split.valid_per_type, 
                self.data_split.test_per_type,
                self.data_split.type, 
                self.data_split.split_on)
        if isinstance(self.dataset, GarmentCodeDatasetQVA):
            self.training = GarmentCodeDatasetQVA(
                self.dataset.root_path, 
                self.dataset.editing_dir,
                self.dataset.caption_dir,
                self.dataset.sampling_rate,
                self.dataset.vision_tower,
                self.dataset.image_size,
                self.dataset.editing_flip_prob,
                self.dataset.body_type,
                self.dataset.original_data_folders,
                self.dataset.garment_tokenizer,
                panel_classification=self.dataset.panel_classification,
                filtered_data_txt=self.dataset.filtered_data_txt,
                load_by_dataname=training_datapoints,
            )
            self.validation = GarmentCodeDatasetQVA(
                self.dataset.root_path, 
                self.dataset.editing_dir,
                self.dataset.caption_dir,
                self.dataset.sampling_rate,
                self.dataset.vision_tower,
                self.dataset.image_size,
                self.dataset.editing_flip_prob,
                self.dataset.body_type,
                self.dataset.original_data_folders,
                self.dataset.garment_tokenizer,
                panel_classification=self.dataset.panel_classification,
                filtered_data_txt=self.dataset.filtered_data_txt,
                load_by_dataname=validation_datapoints,
                inference=True
            )
            if test_datapoints is not None:
                self.test = GarmentCodeDatasetQVA(
                    self.dataset.root_path, 
                    self.dataset.editing_dir,
                    self.dataset.caption_dir,
                    self.dataset.sampling_rate,
                    self.dataset.vision_tower,
                    self.dataset.image_size,
                    self.dataset.editing_flip_prob,
                    self.dataset.body_type,
                    self.dataset.original_data_folders,
                    self.dataset.garment_tokenizer,
                    panel_classification=self.dataset.panel_classification,
                    filtered_data_txt=self.dataset.filtered_data_txt,
                    load_by_dataname=test_datapoints,
                    inference=True
                )
            else:
                self.test = None
        elif isinstance(self.dataset, GarmentCodeMMBenchmark):
            self.test = GarmentCodeMMBenchmark(
                self.dataset.root_path, 
                self.dataset.editing_dir,
                self.dataset.caption_dir,
                self.dataset.mm_dir,
                self.dataset.sampling_rate,
                self.dataset.vision_tower,
                self.dataset.image_size,
                self.dataset.body_type,
                self.dataset.original_data_folders,
                self.dataset.garment_tokenizer,
                panel_classification=self.dataset.panel_classification,
                filtered_data_txt=self.dataset.filtered_data_txt,
                load_by_dataname=test_datapoints,
                inference=True)
        elif isinstance(self.dataset, DressCodeCaptionDataset):
            self.training = DressCodeCaptionDataset(
                self.dataset.root_path, 
                training_datapoints,
                self.dataset.garment_tokenizer,
                self.dataset.panel_classification,
            )
            self.validation = DressCodeCaptionDataset(
                self.dataset.root_path, 
                validation_datapoints,
                self.dataset.garment_tokenizer,
                self.dataset.panel_classification,
            )
            if test_datapoints is not None:
                self.test = DressCodeCaptionDataset(
                    self.dataset.root_path, 
                    test_datapoints,
                    self.dataset.garment_tokenizer,
                    self.dataset.panel_classification,
                )
            else:
                self.test = None
        elif isinstance(self.dataset, GarmentCodeDatasetResize):
            self.training = GarmentCodeDatasetResize(
                self.dataset.root_path, 
                self.dataset.size_file,
                self.dataset.vision_tower,
                self.dataset.image_size,
                self.dataset.body_type,
                self.dataset.data_folders,
                self.dataset.garment_tokenizer,
                panel_classification=self.dataset.panel_classification,
                load_by_dataname=training_datapoints,
            )
            self.validation = GarmentCodeDatasetResize(
                self.dataset.root_path, 
                self.dataset.size_file,
                self.dataset.vision_tower,
                self.dataset.image_size,
                self.dataset.body_type,
                self.dataset.data_folders,
                self.dataset.garment_tokenizer,
                panel_classification=self.dataset.panel_classification,
                load_by_dataname=validation_datapoints,
                inference=True
            )
            if test_datapoints is not None:
                self.test = GarmentCodeDatasetResize(
                    self.dataset.root_path, 
                    self.dataset.size_file,
                    self.dataset.vision_tower,
                    self.dataset.image_size,
                    self.dataset.body_type,
                    self.dataset.data_folders,
                    self.dataset.garment_tokenizer,
                    panel_classification=self.dataset.panel_classification,
                    load_by_dataname=test_datapoints,
                    inference=True
                )
            else:
                self.test = None
        else:
            self.training = QVAGarmentTokenDataset(
                self.dataset.root_path, 
                self.dataset.vision_tower,
                self.dataset.image_size,
                self.dataset.garment_tokenizer,
                panel_classification=self.dataset.panel_classification,
                filter_params=self.dataset.filter_params,
                load_by_dataname=training_datapoints,
            )
            self.validation = QVAGarmentTokenDataset(
                self.dataset.root_path, 
                self.dataset.vision_tower,
                self.dataset.image_size,
                self.dataset.garment_tokenizer,
                panel_classification=self.dataset.panel_classification,
                filter_params=self.dataset.filter_params,
                load_by_dataname=validation_datapoints,
                inference=True
            )
            if test_datapoints is not None:
                self.test = QVAGarmentTokenDataset(
                    self.dataset.root_path, 
                    self.dataset.vision_tower,
                    self.dataset.image_size,
                    self.dataset.garment_tokenizer,
                    panel_classification=self.dataset.panel_classification,
                    filter_params=self.dataset.filter_params,
                    load_by_dataname=test_datapoints,
                    inference=True
                )
            else:
                self.test = None

        log.info('Dataset split: {} / {} / {}'.format(
            len(self.training) if self.training else None, 
            len(self.validation) if self.validation else None, 
            len(self.test) if self.test else None))
            
        self.get_data_lists()
            
        return self.training, self.validation, self.test
    

    def get_data_lists(self):

        if self.data_split.load_from_split_file is not None and  os.path.exists(self.data_split.load_from_split_file):
            log.info('Load Dataset split: {} '.format(self.data_split.load_from_split_file))
        else:
            data_lists = {"train": [], "validation": [], "test": []}
            for name, split in {"train":self.training, "validation": self.validation, "test": self.test}.items():
                for idx in range(len(split)):
                    if isinstance(split, GarmentCodeDatasetQVA) or isinstance(split, DressCodeCaptionDataset):
                        dataname = split.get_item_infos(idx)
                    else:
                        image_name, garment_name, spec_sheet = split.get_item_infos(idx)
                        dataname = "_".join(garment_name)
                    data_lists[name].append(dataname)
                    
            save_path = os.path.join(self.output_dir, "data_split.json")
            json.dump(data_lists, open(save_path, "w"), indent=2)
            self.data_split.load_from_split_file = save_path

            log.info('Save Dataset split: {} '.format(self.data_split.load_from_split_file))
