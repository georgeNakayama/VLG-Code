from argparse import Namespace
import json
import numpy as np
import random
import time
from datetime import datetime
import os
from typing import Literal, Optional
from dataclasses import dataclass, field
import logging
log = logging.getLogger(__name__)
import torch
from torch.utils.data import DataLoader, Subset
from ..data_loaders.infinite_loader import InfiniteDataLoader
from ..datasets.dataset_pcd_hydra import GarmentPCDDataset
from omegaconf import OmegaConf
# My modules
import data.transforms as transforms
from .data_split_config import DataSplitConfig

def collate_fn_pcd(batches):
    # start_time = time.time()
    if isinstance(batches[0]["ground_truth"], dict):
        bdict = {key: [] for key in batches[0].keys()}
        bdict["ground_truth"] = {key:[] for key in batches[0]["ground_truth"]}
        cum_sum = 56
        for i, batch in enumerate(batches):
            for key, val in batch.items():
                if key == "ground_truth":
                    for k, v in batch["ground_truth"].items():
                        if k != "label_indices":
                            bdict["ground_truth"][k].append(v)
                        else:
                            new_label_indices = v.clone()
                            new_label_indices[:, :, 0] += cum_sum * i
                            bdict["ground_truth"][k].append(new_label_indices)
                else:
                    bdict[key].append(val)
                    
        
        for key in bdict.keys():
            if key in ["pcd", "image"]:
                bdict[key] = torch.stack(bdict[key])
            elif key == "ground_truth":
                for k in bdict[key]:
                    if k in ["label_indices", "masked_stitches", "stitch_edge_mask", "reindex_stitches"]:
                        bdict[key][k] = torch.vstack(bdict[key][k])
                    else:
                        bdict[key][k] = torch.stack(bdict[key][k])
        # print("collate_fn: {}".format(time.time() - start_time))
        return bdict
    else:
        bdict = {key: [] for key in batches[0].keys()}
        for i, batch in enumerate(batches):
            for key, val in batch.items():
                bdict[key].append(val)
        bdict["features"] = torch.stack(bdict["features"])
        bdict["ground_truth"] = torch.stack(bdict["ground_truth"])
        return bdict



@dataclass 
class PCDDataWrapperConfig:
    dataset: GarmentPCDDataset
    data_split: DataSplitConfig
    batch_size: int
    shuffle_train=True
    num_workers=12    
    
class PCDDatasetWrapper(object):
    """Resposible for keeping dataset, its splits, loaders & processing routines.
        Allows to reproduce earlier splits
    """

    def __init__(
        self, 
        dataset: GarmentPCDDataset, 
        data_split: DataSplitConfig, 
        batch_size: int, 
        shuffle_train: bool = True, 
        random_seed: Optional[int] = None,
        num_workers: int = 12,
        multiprocess: bool = False,
        output_dir: str = "./output"):
        
        self.dataset = dataset
        self.data_split = data_split
        self.data_section_list = ['full', 'train', 'validation', 'test']

        self.training = dataset
        self.validation = None
        self.test = None
        self.num_workers=num_workers

        self.batch_size = batch_size
        self.output_dir=output_dir

        self.loaders = Namespace(
            full=None,
            train=None,
            test=None,
            real_test=None,
            validation=None
        )

        self.load_split(shuffle_train=shuffle_train, multiprocess=multiprocess, random_seed=random_seed)
    
    def get_loader(self, data_section='full'):
        """Return loader that corresponds to given data section. None if requested loader does not exist"""
        try:
            return getattr(self.loaders, data_section)
        except AttributeError:
            raise ValueError('RealisticDataWrapper::requested loader on unknown data section {}'.format(data_section))
    
    def new_loaders(self, shuffle_train=True, multiprocess=False):
        """Create loaders for current data split. Note that result depends on the random number generator!
        
            if the data split was not specified, only the 'full' loaders are created
        """
        if multiprocess:
            full_sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
            self.loaders.full = DataLoader(self.dataset, self.batch_size, shuffle=False, num_workers=0, pin_memory=True,
                                        sampler=full_sampler)
        else:
            self.loaders.full = DataLoader(self.dataset, self.batch_size)
        if self.validation is not None and self.test is not None:

            if multiprocess:
                train_sampler = torch.utils.data.distributed.DistributedSampler(self.training, drop_last=True)
                self.loaders.train = DataLoader(self.training, self.batch_size, 
                                                shuffle=False,
                                                pin_memory=True,
                                                num_workers=12,
                                                collate_fn=collate_fn_pcd,
                                                sampler=train_sampler)
            else:
                log.info("No Multiprocess")
                self.loaders.train = DataLoader(self.training, self.batch_size, 
                                                #collate_fn=collate_fn, 
                                                num_workers=12, 
                                                pin_memory=True,
                                                shuffle=shuffle_train)

            self.loaders.validation = DataLoader(self.validation, self.batch_size, collate_fn=collate_fn_pcd, num_workers=12)
            self.loaders.test = DataLoader(self.test, self.batch_size, collate_fn=collate_fn_pcd, num_workers=12)
        return self.loaders.train, self.loaders.validation, self.loaders.test

    def _loaders_dict(self, subsets_dict, batch_size, shuffle=False):
        """Create loaders for all subsets in dict"""
        loaders_dict = {}
        for name, subset in subsets_dict.items():
            loaders_dict[name] = DataLoader(subset, batch_size, shuffle=shuffle)
        return loaders_dict
    
    # -------- Reproducibility ---------------
    def new_split(self, valid, test=None, random_seed=None):
        """Creates train/validation or train/validation/test splits
            depending on provided parameters
            """
        self.data_split.valid_per_type=valid
        self.data_split.test_per_type=test
        self.data_split.type='count'
        self.data_split.split_on='folder'
        
        return self.load_split(random_seed=random_seed)
    
    
    def load_split(self, shuffle_train: bool = True, multiprocess: bool = False, random_seed: Optional[int] = None):
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

        self.new_loaders(shuffle_train, multiprocess)  # s.t. loaders could be used right away
        self.standardize_data()
        
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
                split_idxs = split.indices
                for idx in split_idxs:
                    datanames, _ = self.dataset.get_item_infos(idx)
                    data_lists[name].append(datanames)

            save_path = os.path.join(self.output_dir, "data_split.json")
            json.dump(data_lists, open(save_path, "w"), indent=2)
            self.data_split.load_from_split_file = save_path

            log.info('Save Dataset split: {} '.format(self.data_split.load_from_split_file))


    def get_real_test_ids(self, batch_size, fpose=False):

        if self.test is None:
            log.warn("No Test set, Stop")
            return None
        training_idxs = self.training.indices
        test_idxs = self.test.indices
        training_spec_fns = []
        for idx in training_idxs:
            spec_fns, _ = self.dataset.get_item_infos(idx)
            training_spec_fns.extend(spec_fns)
        training_spec_fns = set(training_spec_fns)
        real_test_ids = []
        for idx in test_idxs:
            spec_fns, _ = self.dataset.get_item_infos(idx)
            valid = True
            for spec_fn in spec_fns:
                if spec_fn in training_spec_fns:
                    valid = False 
                    continue
            if valid:
                real_test_ids.append(idx)
        real_test = Subset(self.dataset, real_test_ids)
        log.info("Real Test has total {} examples".format(len(real_test)))
        self.real_test = real_test
        self.loaders.real_test = DataLoader(self.real_test, self.batch_size)
        
    
    def save_to_wandb(self, experiment):
        """Save current data info to the wandb experiment"""
        # Split
        experiment.add_config('data_split', OmegaConf.to_container(self.data_split))
        # save serialized split s.t. it's loaded to wandb
        split_datanames = {}
        split_datanames['training'] = [self.dataset.datapoints_names[idx]["dataname"] for idx in self.training.indices]
        split_datanames['validation'] = [self.dataset.datapoints_names[idx]["dataname"] for idx in self.validation.indices]
        split_datanames['test'] = [self.dataset.datapoints_names[idx]["dataname"] for idx in self.test.indices]
        with open(os.path.join(experiment.local_output_dir, 'data_split.json'), 'w') as f_json:
            json.dump(split_datanames, f_json, indent=2, sort_keys=True)

        # data info
        self.dataset.save_to_wandb(experiment)
    
    # ---------- Standardinzation ----------------
    def standardize_data(self):
        """Apply data normalization based on stats from training set"""
        self.dataset.standardize(self.training)
        
    
    # --------- Managing predictions on this data ---------
    def predict(self, model, save_to, sections=['test'], single_batch=False, orig_folder_names=False, use_gt_stitches=False):
        """Save model predictions on the given dataset section"""
        prediction_path = os.path.join(save_to, ('nn_pred_' + datetime.now().strftime('%y%m%d-%H-%M-%S')))
        os.makedirs(prediction_path, exist_ok=True)
        model.module.eval()
        self.dataset.set_training(False)

        for section in sections:
            section_dir = prediction_path 
            os.makedirs(section_dir, exist_ok=True)
            cnt = 0
            with torch.no_grad():
                loader = self.get_loader(section)
                if loader:
                    for batch in loader:
                        cnt += 1
                        pcds = batch["pcd"].to(model.device_ids[0])
                        b = pcds.shape[0]
                        preds = model(pcds)

                        panel_shape = np.linalg.norm(preds["outlines"].cpu().detach().numpy().reshape((b, -1)) - batch["ground_truth"]["outlines"].cpu().detach().numpy().reshape(b, -1), axis=1)

                        self.dataset.save_prediction_batch(
                            preds, batch['pcd_fn'], batch['data_folder'], section_dir, 
                            model=model, orig_folder_names=orig_folder_names, pcds=pcds, use_gt_stitches=use_gt_stitches, panel_shape=panel_shape)

        return prediction_path
    

    def predict_single(self, model, pcd, dataname, save_to):
        device = model.device_ids[0] if hasattr(model, 'device_ids') and len(model.device_ids) > 0 else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        pcd_transform = transforms.tv_make_geo_transform()
        pcd = pcd_transform(pcd).to(device).unsqueeze(0)
        output = model(pcd)
        panel_order, panel_idx, prediction_img = self.dataset.save_prediction_single(output, dataname, save_to)
        return panel_order, panel_idx, prediction_img
    
    def run_single_pcd(self, pcd, model, datawrapper):
        device = model.device_ids[0] if hasattr(model, 'device_ids') and len(model.device_ids) > 0 else torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        pcd_transform = transforms.tv_make_geo_transform()
        pcd = pcd_transform(pcd).to(device).unsqueeze(0)
        output = model(pcd), pcd
        return output, pcd

