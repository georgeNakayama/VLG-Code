import json
import numpy as np
import os
from pathlib import Path, PureWindowsPath
import shutil
import glob
from PIL import Image
import random
import time
import logging
from typing import Literal
import tiktoken
log = logging.getLogger(__name__)
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset, Subset
import torch.nn.functional as F
from torchvision.io import read_image
import torchvision.transforms as T
from dataclasses import dataclass
from enum import Enum
# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
grandparentdir = os.path.dirname(parentdir)
sys.path.insert(0, grandparentdir)
root_path = os.path.dirname(os.path.dirname(os.path.abspath(parentdir)))
pkg_path = "{}/SewFactory/packages".format(root_path)
print(pkg_path)
sys.path.insert(0, pkg_path)


# My modules
from customconfig import Properties
from data.pattern_converter import NNSewingPattern, InvalidPatternDefError
import data.transforms as transforms
from data.panel_classes import PanelClasses
from data.datasets.utils import PanelEdgeType, SpecialTokens
from data.human_body_prior.body_model import BodyModel
from data.utils import euler_angle_to_rot_6d
from data.datasets.panel_configs import *


class GarmentTokenDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        base_offset: int,
        standardize: StandardizeConfig, 
        text_encoder: Literal["gpt2"],
        bin_size: int,
        panel_classification: str,
        filter_params: str,):

        self.root_path = root_dir
        self.base_offset = base_offset
        self.bin_size = bin_size
        self.gt_stats = standardize
        self.special_token_offset = base_offset
        self.panel_edge_type_offset = self.special_token_offset + len(SpecialTokens)
        self.edge_offset = len(PanelEdgeType) + self.panel_edge_type_offset
        self.total_vocab_size = self.edge_offset + bin_size
        self.config = {}
        self.config['class'] = self.__class__.__name__
        self.text_encoder = tiktoken.get_encoding(text_encoder)
        self.stop_token = self.text_encoder._special_tokens['<|endoftext|>']
        self.datapoints_names = []
        self.dataset_start_ids = []  # (folder, start_id) tuples -- ordered by start id
        self.panel_classification = panel_classification
        self.filter_params = filter_params

        try:
            folders = [folder for folder in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, folder))]
            for folder in folders:
                self.dataset_start_ids.append((folder, len(self.datapoints_names)))
                gt_folder = os.path.join(self.root_path, folder, "static")
                if os.path.exists(gt_folder):
                    spec_sheet = json.load(open(os.path.join(gt_folder, "spec_config.json"), "r"))
                    garment_names = [spec_sheet[k]["spec"].split("\\")[-1] for k in spec_sheet.keys()]
                    for name in garment_names:
                        sheet = os.path.join(gt_folder, name + "_specification.json")
                        self.datapoints_names.append((name, sheet))
        except Exception as e:
            print(e)
        
        self.dataset_start_ids.append((None, len(self.datapoints_names)))
        self.config['size'] = len(self)
        print("GarmentDetrDataset::Info::Total valid datanames is {}".format(self.config['size']))

        self.gt_cached, self.gt_caching = {}, True
        if self.gt_caching:
            print('GarmentDetrDataset::Info::Storing datapoints ground_truth info in memory')
        

        self.is_train = False

        # Load panel classifier
        if self.panel_classification is not None:
            self.panel_classifier = PanelClasses(self.panel_classification)
            self.config.update(max_pattern_len=len(self.panel_classifier))
        else:
            raise RuntimeError('GarmentDetrDataset::Error::panel_classification not found')
    
    def invalid_lists(self):
        invalid_fn = "./utilities/invalid_files.txt"
        with open(invalid_fn, "r") as f:
            invalid_lst = f.readlines()
        invalid_lst = [fn.strip() for fn in invalid_lst]
        invalid_lst.append("tee_SHPNN0VX1Z_wb_pants_straight_UBP5W37LKV")
        invalid_lst.append("jumpsuit_sleeveless_QI8XX8SQAQ")
        return invalid_lst
    
    def compute_stats(self):
        """Compute data statistics for normalization"""
        log.info('Computing data statistics for normalization')
        all_pts = []
        for name, spec_sheet in tqdm(self.datapoints_names):
            pattern = NNSewingPattern(spec_sheet, panel_classifier=self.panel_classifier, template_name="_".join(name.split("_")[:-1]))
            panel_vertices = pattern.pattern_vertices()
            all_pts.append(panel_vertices)


        all_pts = np.concatenate(all_pts, axis=0)
        all_pts_min = np.min(all_pts, axis=0)
        all_pts_max = np.max(all_pts, axis=0)
        print("Vector min: ", all_pts_min)
        print("Vector max: ", all_pts_max)

    def standardize(self, training=None):
        """Use shifting and scaling for fitting data to interval comfortable for NN training.
            Accepts either of two inputs: 
            * training subset to calculate the data statistics -- the stats are only based on training subsection of the data
            * if stats info is already defined in config, it's used instead of calculating new statistics (usually when calling to restore dataset from existing experiment)
            configuration has a priority: if it's given, the statistics are NOT recalculated even if training set is provided
                => speed-up by providing stats or speeding up multiple calls to this function
        """
        log.info('Using data normalization for features & ground truth')
        stats = self.gt_stats
        log.info(stats)
        gt_shift = {
            "outlines": stats.outlines.shift, 
            "rotations": stats.rotations.shift, 
            "stitch_tags": stats.stitch_tags.shift, 
            "translations": stats.translations.shift
            }
        gt_scale = {
            "outlines": stats.outlines.scale, 
            "rotations": stats.rotations.scale, 
            "stitch_tags": stats.stitch_tags.scale, 
            "translations": stats.translations.scale
            }
        self.transforms.append(transforms.GTtandartization(gt_shift, gt_scale))

    def save_to_wandb(self, experiment):
        """Save data cofiguration to current expetiment run"""
        # config
        experiment.add_config('dataset', self.config)
        # panel classes
        if self.panel_classifier is not None:
            shutil.copy(
                self.panel_classifier.filename, 
                experiment.local_ouptut_dir() / ('panel_classes.json'))
    
    def set_training(self, is_train=True):
        self.is_train = is_train
    
    def update_config(self, in_config):
        """Define dataset configuration:
            * to be part of experimental setup on wandb
            * Control obtainign values for datapoints"""

        # initialize keys for correct dataset initialization
        if ('max_pattern_len' not in in_config
                or 'max_panel_len' not in in_config
                or 'max_num_stitches' not in in_config):
            in_config.update(max_pattern_len=None, max_panel_len=None, max_num_stitches=None)
            pattern_size_initialized = False
        else:
            pattern_size_initialized = True
        
        if 'obj_filetag' not in in_config:
            in_config['obj_filetag'] = ''  # look for objects with this tag in filename when loading 3D models

        if 'panel_classification' not in in_config:
            in_config['panel_classification'] = None
        
        self.config.update(in_config)
        # check the correctness of provided list of datasets
        if ('data_folders' not in self.config 
                or not isinstance(self.config['data_folders'], list)
                or len(self.config['data_folders']) == 0):
            #raise RuntimeError('BaseDataset::Error::information on datasets (folders) to use is missing in the incoming config')
            print(f'{self.__class__.__name__}::Info::Collecting all datasets (no sub-folders) to use')
        return pattern_size_initialized

    def __len__(self, ):
        """Number of entries in the dataset"""
        return len(self.datapoints_names) 

    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data. 
        Does not support list indexing"""
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        if idx in self.gt_cached:   
            encoded_pattern = self.gt_cached[idx]
        else:
            template_name, spec_sheet = self.datapoints_names[idx]
            template_name = "_".join(template_name.split("_")[:-1])
            pattern = NNSewingPattern(
                spec_sheet, panel_classifier=self.panel_classifier, 
                template_name=template_name)

            # TODO: no translation and rotations for now
            pattern_edges, pattern_trans, pattern_rots, panel_names = pattern.pattern_as_tokens()
            import code; code.interact(local=locals())
            encoded_pattern = self.encode_pattern(template_name, pattern_edges, panel_names)
            encoded_pattern = torch.tensor(encoded_pattern, dtype=torch.long)
            if self.gt_caching:
                self.gt_cached[idx] = encoded_pattern
            

        return encoded_pattern
    

    # version with command but without absolute vertices: 

    def encode_pattern(self, template_name, pattern_edges, panel_names): 
        template_fn = lambda s: f"A sewing pattern for {s}:"
        template_name = " ".join(template_name.split("_"))
        template_name_tokens = self.text_encoder.encode(template_fn(template_name))
        out_tokens = template_name_tokens + [SpecialTokens.PATTERN_START.value + self.special_token_offset]
        for panel_edges, panel_name in zip(pattern_edges, panel_names):
            panel_text_tokens = self.text_encoder.encode(panel_name)
            panel_tokens = panel_text_tokens + [SpecialTokens.PANEL_START.value + self.special_token_offset]
            for panel_edge in panel_edges:
                edge_type: PanelEdgeType = panel_edge[0]
                edge_params: np.ndarray = (panel_edge[1].reshape(-1, 2)  - self.gt_stats.vertices.shift) / self.gt_stats.vertices.scale
                edge_params = np.clip(edge_params, 0, 1) * self.bin_size
                edge_params = edge_params.astype(int).clip(0, self.bin_size - 1) + self.edge_offset
                panel_tokens.append(edge_type.value + self.panel_edge_type_offset)
                param_num = edge_type.get_num_params()
                if param_num > 0:
                    panel_tokens.extend(edge_params.flatten()[:param_num].tolist())
            panel_tokens.append(SpecialTokens.PANEL_END.value + self.special_token_offset)
            out_tokens.extend(panel_tokens)
        out_tokens.append(SpecialTokens.PATTERN_END.value + self.special_token_offset)
        return out_tokens


    # def encode_pattern(self, template_name, pattern_edges, panel_names):
    #     template_fn = lambda s: f"A sewing pattern for {s}:"
    #     template_name = " ".join(template_name.split("_"))
    #     template_name_tokens = self.text_encoder.encode(template_fn(template_name))
    #     out_tokens = template_name_tokens + [SpecialTokens.PATTERN_START.value + self.special_token_offset]
    #     for panel_edges, panel_name in zip(pattern_edges, panel_names):
    #         panel_text_tokens = self.text_encoder.encode(panel_name)
    #         panel_tokens = panel_text_tokens + [SpecialTokens.PANEL_START.value + self.special_token_offset]
    #         for panel_edge in panel_edges:
    #             edge_type: PanelEdgeType = panel_edge[0]
    #             edge_params: np.ndarray = (panel_edge[1].reshape(-1, 2)  - self.gt_stats.vertices.shift) / self.gt_stats.vertices.scale
    #             edge_params = np.clip(edge_params, 0, 1) * self.bin_size
    #             edge_params = edge_params.astype(int).clip(0, self.bin_size - 1) + self.edge_offset
    #             panel_tokens.append(edge_type.value + self.panel_edge_type_offset)
    #             param_num = edge_type.get_num_params()
    #             if param_num > 0:
    #                 panel_tokens.extend(edge_params.flatten()[:param_num].tolist())
    #         panel_tokens.append(SpecialTokens.PANEL_END.value + self.special_token_offset)
    #         out_tokens.extend(panel_tokens)
    #     out_tokens.append(SpecialTokens.PATTERN_END.value + self.special_token_offset)
    #     return out_tokens
    
    def decode_pattern(self, token_sequence): 
        # find the panel starts and ends
        garment_end = np.where(token_sequence == SpecialTokens.PATTERN_END.value + self.special_token_offset)[0]
        if len(garment_end) > 0:
            token_sequence = token_sequence[:garment_end[0]]
        else:
            token_sequence = token_sequence
        garment_start = np.where(token_sequence == SpecialTokens.PATTERN_START.value + self.special_token_offset)[0]    
        if len(garment_start) > 0:
            try:
                pattern_description = self.text_encoder.decode(token_sequence[:garment_start[0]].tolist())
            except:
                pattern_description = None
            token_sequence = token_sequence[garment_start[0]+1:]
        else:
            pattern_description = None
        panel_starts = np.where(token_sequence == SpecialTokens.PANEL_START.value + self.special_token_offset)[0]
        panel_ends = np.where(token_sequence == SpecialTokens.PANEL_END.value + self.special_token_offset)[0]

        n_panels = min(len(panel_starts), len(panel_ends))
        panels = []
        panel_names = ["NONE" for _ in range(n_panels)]
        if n_panels == 0:
            return panels, panel_names, pattern_description

        if len(panel_starts) < len(panel_ends):
            panel_ends = panel_ends[:len(panel_starts)]

        if len(panel_starts) > len(panel_ends):
            panel_starts = panel_starts[:len(panel_ends)]

        if np.any(panel_starts > panel_ends):
            return panels, panel_names, pattern_description
        
        current_mark = 0
        for i in range(n_panels):
            panel_start = panel_starts[i]
            panel_end = panel_ends[i]
            panel = token_sequence[panel_start+1:panel_end]
            
            commands = np.isin(panel, [e.value + self.panel_edge_type_offset for e in PanelEdgeType]).nonzero()[0]
            if len(commands) == 0:
                log.error(f"Panel {i} has no edges. Skipping.")
                panels.append(all_edges)
                current_mark = panel_end + 1
                continue
            
            try: 
                panel_description = tokenizer.decode(panel[:commands[0]].tolist())
            except:
                panel_description = ''
            panel_names[i] = panel_description if panel_description != '' else "NONE"
            last_point = np.array([0, 0])
            all_edges = []
            for j in range(len(commands)):
                command = commands[j]
                command_end = commands[j+1] if j+1 < len(commands) else len(panel)
                edge_type = PanelEdgeType(panel[command].item() - self.panel_edge_type_offset)
                edge_params = panel[command+1:command_end]
                if len(edge_params) != edge_type.get_num_params():
                    # force close the loop and break out of this panel
                    edge_params = np.concatenate([-last_point, np.array([0, 0])])
                    all_edges.append(edge_params)
                    break
                if edge_type == PanelEdgeType.CLOSURE_LINE:
                    # Start point is always 0
                    edge_params = np.concatenate([-last_point, np.array([0, 0])])
                elif edge_type == PanelEdgeType.LINE:
                    edge_params = edge_params.astype(float)
                    edge_params = (edge_params - self.edge_offset + 0.5).clip(0, self.bin_size) / self.bin_size
                    edge_params = edge_params * np.array(self.gt_stats.vertices.scale) + np.array(self.gt_stats.vertices.shift)
                    edge_params = np.concatenate([edge_params - last_point, np.array([0, 0])])
                elif edge_type == PanelEdgeType.CURVE:
                    
                    edge_params = edge_params.astype(float)
                    edge_params = (edge_params - self.edge_offset + 0.5).clip(0, self.bin_size) / self.bin_size
                    edge_params = edge_params.reshape(2, 2) * self.gt_stats.vertices.scale + self.gt_stats.vertices.shift
                    abs_ctrl_pt, edge_end = edge_params[0], edge_params[1]
                    ctrl_pt = NNSewingPattern._control_to_relative_coord(last_point, edge_end, abs_ctrl_pt)
                    edge_params = np.concatenate([edge_end - last_point, ctrl_pt])
                elif edge_type == PanelEdgeType.CLOSURE_CURVE:
                    edge_params = edge_params.astype(float)
                    edge_params = (edge_params - self.edge_offset + 0.5).clip(0, self.bin_size) / self.bin_size
                    edge_params = edge_params * self.gt_stats.vertices.scale + self.gt_stats.vertices.shift
                    ctrl_pt = NNSewingPattern._control_to_relative_coord(last_point, np.array([0, 0]), edge_params)
                    edge_params = np.concatenate([-last_point, ctrl_pt])
                last_point = edge_params[:2] + last_point
                all_edges.append(edge_params)
            all_edges = np.stack(all_edges).astype(float)
            panels.append(all_edges)
            current_mark = panel_end + 1
        max_edge_len = max(len(edges) for edges in panels)
        out_panel_tensor = np.zeros((n_panels, max_edge_len, 4))
        for i, edges in enumerate(panels):
            out_panel_tensor[i, :len(edges)] = edges
        return out_panel_tensor, panel_names, pattern_description
    # def decode_pattern(self, token_sequence): 
    #     # find the panel starts and ends
    #     garment_end = np.where(token_sequence == SpecialTokens.PATTERN_END.value + self.special_token_offset)[0]
    #     if len(garment_end) > 0:
    #         token_sequence = token_sequence[:garment_end[0]]
    #     else:
    #         token_sequence = token_sequence
    #     garment_start = np.where(token_sequence == SpecialTokens.PATTERN_START.value + self.special_token_offset)[0]    
    #     if len(garment_start) > 0:
    #         try:
    #             pattern_description = self.text_encoder.decode(token_sequence[:garment_start[0]].tolist())
    #         except:
    #             pattern_description = None
    #         token_sequence = token_sequence[garment_start[0]+1:]
    #     else:
    #         pattern_description = None
    #     panel_starts = np.where(token_sequence == SpecialTokens.PANEL_START.value + self.special_token_offset)[0]
    #     panel_ends = np.where(token_sequence == SpecialTokens.PANEL_END.value + self.special_token_offset)[0]

    #     n_panels = len(panel_starts)
    #     panels = []
    #     panel_names = ["NONE" for _ in range(n_panels)]

    #     if len(panel_starts) < len(panel_ends):
    #         panel_ends = panel_ends[:len(panel_starts)]
    #         panel_names = panel_names[:len(panel_starts)]
    #     if len(panel_starts) > len(panel_ends):
    #         panel_starts = panel_starts[:len(panel_ends)]
    #         panel_names = panel_names[:len(panel_ends)]
    #     if np.any(panel_starts > panel_ends):
    #         return panels, panel_names
        
    #     current_mark = 0
    #     for i in range(n_panels):
    #         panel_start = panel_starts[i]
    #         panel_end = panel_ends[i]
    #         try: 
    #             panel_description = self.text_encoder.decode(token_sequence[current_mark:panel_start].tolist())
    #         except:
    #             panel_description = ''
    #         panel_names[i] = panel_description if panel_description != '' else "NONE"
    #         panel = token_sequence[panel_start+1:panel_end]
    #         commands = np.isin(panel, [e.value + self.panel_edge_type_offset for e in PanelEdgeType]).nonzero()[0]
    #         last_point = np.array([0, 0])
    #         all_edges = []
    #         for j in range(len(commands) - 1):
    #             command = commands[j]
    #             command_end = commands[j+1] if j+1 < len(commands) else len(panel)
    #             edge_type = PanelEdgeType(panel[command].item() - self.panel_edge_type_offset)
    #             edge_params = panel[command+1:command_end]
    #             assert len(edge_params) == edge_type.get_num_params()
    #             if edge_type == PanelEdgeType.CLOSURE_LINE:
    #                 # Start point is always 0
    #                 edge_params = np.concatenate([-last_point, np.array([0, 0])])
    #                 break
    #             if edge_type == PanelEdgeType.LINE:
    #                 edge_params = edge_params.astype(float)
    #                 edge_params = (edge_params - self.edge_offset + 0.5) / self.bin_size
    #                 edge_params = edge_params * self.gt_stats.outlines.scale[:2] + self.gt_stats.outlines.shift[:2]
    #                 edge_params = np.concatenate([edge_params, np.array([0, 0])])
    #             if edge_type == PanelEdgeType.CURVE:
    #                 edge_params = edge_params.astype(float)
    #                 edge_params = (edge_params - self.edge_offset + 0.5) / self.bin_size
    #                 edge_params = edge_params * self.gt_stats.outlines.scale + self.gt_stats.outlines.shift
    #             if edge_type == PanelEdgeType.CLOSURE_CURVE:
    #                 edge_params = edge_params.astype(float)
    #                 edge_params = (edge_params - self.edge_offset + 0.5) / self.bin_size
    #                 edge_params = edge_params * self.gt_stats.outlines.scale[2:] + self.gt_stats.outlines.shift[2:]
    #                 edge_params = np.concatenate([-last_point, edge_params])
    #                 break
    #             last_point = edge_params[:2]
    #             all_edges.append(edge_params)
    #         all_edges = np.stack(all_edges).astype(float)
    #         panels.append(all_edges)
    #         current_mark = panel_end + 1
    #     max_edge_len = max(len(edges) for edges in panels)
    #     out_panel_tensor = np.zeros((n_panels, max_edge_len, 4))
    #     for i, edges in enumerate(panels):
    #         out_panel_tensor[i, :len(edges)] = edges
    #     return out_panel_tensor, panel_names, pattern_description
    
    
    def _swap_name(self, name):
        # sim root to original root
        name = name.replace(self.sim_root, self.root_path).split('/')
        return '/'.join(name[:-1] + ["renders"] + name[-1:])
    
    def _load_gt_folders_from_indices(self, indices):
        gt_folders = [self.datapoints_names[idx][-1] for idx in indices]
        return list(set(gt_folders))
    
    def get_smpl_pose_fn(self, datapoint_name, gt_folder):
        return os.path.join(os.path.dirname(gt_folder), "poses", os.path.basename(datapoint_name).split("_")[0] + "__body_info.json")

    
    def _drop_cache(self):
        """Clean caches of datapoints info"""
        self.gt_cached = {}
        self.feature_cached = {}
    
    def _renew_cache(self):
        """Flush the cache and re-fill it with updated information if any kind of caching is enabled"""
        self.gt_cached = {}
        self.feature_cached = {}
        if self.feature_caching or self.gt_caching:
            for i in range(len(self)):
                self[i]
            print('Data cached!')
    
    def random_split_by_dataset(self, valid_per_type, test_per_type, split_type='count', split_on="pattern"):
        if split_type != 'count' and split_type != 'percent':
            raise NotImplementedError('{}::Error::Unsupported split type <{}> requested'.format(
                self.__class__.__name__, split_type))
        train_ids, valid_ids, test_ids = [], [], []

        if split_on == "pattern":
            # valid_per_type & test_per_type used on folder
            data_len = len(self)
            perm = np.random.permutation(len(self))
            valid_size = int(data_len * valid_per_type / 100) if split_type == 'percent' else valid_per_type
            test_size = int(data_len * test_per_type / 100) if split_type == 'percent' else test_per_type
            train_size = data_len - valid_size - test_size
            train_ids = perm[:train_size]
            valid_ids = perm[train_size:train_size+valid_size]
            test_ids = perm[train_size+valid_size:]
            
            if test_size:
                random.shuffle(test_ids)
        return (
            Subset(self, train_ids), 
            Subset(self, valid_ids),
            Subset(self, test_ids) if test_size else None
        )

    
    def split_from_dict(self, split_dict):
        """
            Reproduce the data split in the provided dictionary: 
            the elements of the currect dataset should play the same role as in provided dict
        """
        train_ids, valid_ids, test_ids = [], [], []
        
        training_datanames = split_dict['train']
        valid_datanames = split_dict['validation']
        test_datanames = split_dict['test']

        for idx in range(len(self.datapoints_names)):
            if self.datapoints_names[idx][0] in training_datanames:  # usually the largest, so check first
                train_ids.append(idx)
            elif self.datapoints_names[idx][0] in test_datanames:
                test_ids.append(idx)
            elif self.datapoints_names[idx][0] in valid_datanames:
                valid_ids.append(idx)
            else:
                continue
            
            if idx % 1000 == 0:
                print(f"progress {idx}, #Train_Ids={len(train_ids)}, #Valid_Ids={len(valid_ids)}, #Test_Ids={len(test_ids)}")
        
        return (
            Subset(self, train_ids), 
            Subset(self, valid_ids),
            Subset(self, test_ids) if len(test_ids) > 0 else None
        )


    # ----- Sample -----
    def _get_sample_info(self, gt_folder):
        """
            Get features and Ground truth prediction for requested data example
        """
        folder_elements = [os.path.basename(file) for file in glob.glob(os.path.join(gt_folder, "*"))]  # all files in this directory

        # GT -- pattern
        if gt_folder in self.gt_cached: # might not be compatible with list indexing
            ground_truth = self.gt_cached[gt_folder]
        else:
            spec_dict = self._load_spec_dict(gt_folder)
            ground_truth = self._get_pattern_ground_truth(gt_folder, folder_elements, spec_dict)
            if self.gt_caching:
                self.gt_cached[gt_folder] = ground_truth
        return ground_truth
    
    def _load_spec_dict(self, gt_folder):
        if gt_folder in self.gt_jsons["spec_dict"]:
            return self.gt_jsons["spec_dict"][gt_folder]
        else:
            # add smpl root at static pose
            static_pose = json.load(open(gt_folder + "/static__body_info.json", "r"))
            static_root = static_pose["trans"]
            spec_dict = json.load(open(gt_folder + "/spec_config.json", "r"))
            for key, val in spec_dict.items():
                spec = PureWindowsPath(val["spec"]).parts[-1]
                spec_dict[key]["spec"] = spec
                spec_dict[key]["delta"] = np.array(val["delta"]) - np.array(static_root)
            self.gt_jsons["spec_dict"][gt_folder] = spec_dict
            return spec_dict
    
    def _get_pattern_ground_truth(self, gt_folder, folder_elements, spec_dict):
        """Get the pattern representation with 3D placement"""
        patterns = self._read_pattern(
            gt_folder, folder_elements, spec_dict,
            pad_panels_to_len=self.config['max_panel_len'],
            pad_panel_num=self.config['max_pattern_len'],
            pad_stitches_num=self.config['max_num_stitches'],
            with_placement=True, with_stitches=True, with_stitch_tags=True)
        pattern, num_edges, num_panels, rots, tranls, stitches, num_stitches, stitch_adj, stitch_tags, aug_outlines = patterns 
        free_edges_mask = self.free_edges_mask(pattern, stitches, num_stitches)
        empty_panels_mask = self._empty_panels_mask(num_edges)  # useful for evaluation
        
        ground_truth = {
            'outlines': pattern, 'num_edges': num_edges,
            'rotations': rots, 'translations': tranls, 
            'num_panels': num_panels, 'empty_panels_mask': empty_panels_mask, 'num_stitches': num_stitches,
            'stitches': stitches, 'stitch_adj': stitch_adj, 'free_edges_mask': free_edges_mask, 'stitch_tags': stitch_tags
        }

        if aug_outlines[0] is not None:
            ground_truth.update({"aug_outlines": aug_outlines})

        return ground_truth
    
    def _load_ground_truth(self, gt_folder):
        folder_elements = [os.path.basename(file) for file in glob.glob(os.path.join(gt_folder, "*"))] 
        spec_dict = self._load_spec_dict(gt_folder)
        ground_truth = self._get_pattern_ground_truth(gt_folder, folder_elements, spec_dict)
        return ground_truth
    
    def _empty_panels_mask(self, num_edges):
        """Empty panels as boolean mask"""

        mask = np.zeros(len(num_edges), dtype=bool)
        mask[num_edges == 0] = True

        return mask
    
    @staticmethod
    def match_edges(free_edge_mask, stitches=None, max_num_stitch_edges=56):
        stitch_edges = np.ones((1, max_num_stitch_edges)) * (-1)
        valid_edges = (~free_edge_mask.reshape(-1)).nonzero()
        stitch_edge_mask = np.zeros((1, max_num_stitch_edges))
        if stitches is not None:
            stitches = np.transpose(stitches)
            reindex_stitches = np.zeros((1, max_num_stitch_edges, max_num_stitch_edges))
        else:
            reindex_stitches = None
        
        batch_edges = valid_edges[0]
        num_edges = batch_edges.shape[0]
        stitch_edges[:, :num_edges] = batch_edges
        stitch_edge_mask[:, :num_edges] = 1
        if stitches is not None:
            for stitch in stitches:
                side_i, side_j = stitch
                if side_i != -1 and side_j != -1:
                    reindex_i, reindex_j = np.where(stitch_edges[0] == side_i)[0], np.where(stitch_edges[0] == side_j)[0]
                    reindex_stitches[0, reindex_i, reindex_j] = 1
                    reindex_stitches[0, reindex_j, reindex_i] = 1
        
        return stitch_edges * stitch_edge_mask, stitch_edge_mask, reindex_stitches
    
    @staticmethod
    def split_pos_neg_pairs(stitches, num_max_edges=3000):
        stitch_ind = np.triu_indices_from(stitches[0], 1)
        pos_ind = [[stitch_ind[0][i], stitch_ind[1][i]] for i in range(stitch_ind[0].shape[0]) if stitches[0, stitch_ind[0][i], stitch_ind[1][i]]]
        neg_ind = [[stitch_ind[0][i], stitch_ind[1][i]] for i in range(stitch_ind[0].shape[0]) if not stitches[0, stitch_ind[0][i], stitch_ind[1][i]]]

        assert len(neg_ind) >= num_max_edges
        neg_ind = neg_ind[:num_max_edges]
        pos_inds = np.expand_dims(np.array(pos_ind), axis=1)
        neg_inds = np.repeat(np.expand_dims(np.array(neg_ind), axis=0), repeats=pos_inds.shape[0], axis=0)
        indices = np.concatenate((pos_inds, neg_inds), axis=1)
        return indices
    

    # ------------- Datapoints Utils --------------
    def template_name(self, spec):
        """Get name of the garment template from the path to the datapoint"""
        return "_".join(spec.split('_')[:-1]) 
    
    def _read_pattern(self, gt_folder, folder_elements, spec_dict,
                      pad_panels_to_len=None, pad_panel_num=None, pad_stitches_num=None,
                      with_placement=False, with_stitches=False, with_stitch_tags=False):
        """Read given pattern in tensor representation from file"""
        

        spec_list = {}
        for key, val in spec_dict.items():
            spec_file = [file for file in folder_elements if val["spec"] in file and "specification.json" in file]
            if len(spec_file) > 0:
                spec_list[key] = spec_file[0]
            else:
                raise ValueError("Specification Cannot be found in folder_elements for {}".format(gt_folder))
        
        if not spec_list:
            raise RuntimeError('GarmentDetrDataset::Error::*specification.json not found for {}'.format(gt_folder))
        patterns = []

        for key, spec in spec_list.items():
            if gt_folder + "/" + spec in self.gt_jsons["specs"]:
                pattern = self.gt_jsons["specs"][gt_folder + "/" + spec]
            else:
                pattern = NNSewingPattern(
                    gt_folder + "/" + spec, 
                    panel_classifier=self.panel_classifier, 
                    template_name=self.template_name(spec_dict[key]['spec']))
                self.gt_jsons["specs"][gt_folder + "/" + spec] = pattern
            patterns.append(pattern)

        pat_tensor = NNSewingPattern.multi_pattern_as_tensors(patterns,
            pad_panels_to_len, pad_panels_num=pad_panel_num, pad_stitches_num=pad_stitches_num,
            with_placement=with_placement, with_stitches=with_stitches, 
            with_stitch_tags=with_stitch_tags, spec_dict=spec_dict)
        return pat_tensor
    
    
    def get_item_infos(self, idx):
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()
        datapoint_name, pattern_sheet = self.datapoints_names[idx]
        return datapoint_name, pattern_sheet
    
    def _unpad(self, element, tolerance=1.e-5):
        """Return copy of input element without padding from given element. Used to unpad edge sequences in pattern-oriented datasets"""
        # NOTE: might be some false removal of zero edges in the middle of the list.
        if torch.is_tensor(element):        
            bool_matrix = torch.isclose(element, torch.zeros_like(element), atol=tolerance)  # per-element comparison with zero
            selection = ~torch.all(bool_matrix, axis=1)  # only non-zero rows
        else:  # numpy
            selection = ~np.all(np.isclose(element, 0, atol=tolerance), axis=1)  # only non-zero rows
        return element[selection]
    
    def _get_distribution_stats(self, input_batch, padded=False):
        """Calculates mean & std values for the input tenzor along the last dimention"""

        input_batch = input_batch.view(-1, input_batch.shape[-1])
        if padded:
            input_batch = self._unpad(input_batch)  # remove rows with zeros

        # per dimention means
        mean = input_batch.mean(axis=0)
        # per dimention stds
        stds = ((input_batch - mean) ** 2).sum(0)
        stds = torch.sqrt(stds / input_batch.shape[0])

        return mean, stds
    
    def _get_norm_stats(self, input_batch, padded=False):
        """Calculate shift & scaling values needed to normalize input tenzor 
            along the last dimention to [0, 1] range"""
        input_batch = input_batch.view(-1, input_batch.shape[-1])
        if padded:
            input_batch = self._unpad(input_batch)  # remove rows with zeros

        # per dimention info
        min_vector, _ = torch.min(input_batch, dim=0)
        max_vector, _ = torch.max(input_batch, dim=0)
        scale = torch.empty_like(min_vector)

        # avoid division by zero
        for idx, (tmin, tmax) in enumerate(zip(min_vector, max_vector)): 
            if torch.isclose(tmin, tmax):
                scale[idx] = tmin if not torch.isclose(tmin, torch.zeros(1)) else 1.
            else:
                scale[idx] = tmax - tmin
        
        return min_vector, scale
    
    # ----- Saving predictions -----
    @staticmethod
    def free_edges_mask(pattern, stitches, num_stitches):
        """
        Construct the mask to identify edges that are not connected to any other
        """
        mask = np.ones((pattern.shape[0], pattern.shape[1]), dtype=bool)
        max_edge = pattern.shape[1]

        for side in stitches[:, :num_stitches]:  # ignore the padded part
            for edge_id in side:
                mask[edge_id // max_edge][edge_id % max_edge] = False
        
        return mask
    
    @staticmethod
    def prediction_to_stitches(free_mask_logits, similarity_matrix, return_stitches=False):
        free_mask = (torch.sigmoid(free_mask_logits.squeeze(-1)) > 0.5).flatten()
        if not return_stitches:
            simi_matrix = similarity_matrix + similarity_matrix.transpose(0, 1)
            simi_matrix = torch.masked_fill(simi_matrix, (~free_mask).unsqueeze(0), -float("inf"))
            simi_matrix = torch.masked_fill(simi_matrix, (~free_mask).unsqueeze(-1), 0)
            num_stitches = free_mask.nonzero().shape[0] // 2
        else:
            simi_matrix = similarity_matrix
            num_stitches = simi_matrix.shape[0] // 2
        simi_matrix = torch.triu(simi_matrix, diagonal=1)
        stitches = []
        
        for i in range(num_stitches):
            index = (simi_matrix == torch.max(simi_matrix)).nonzero()
            stitches.append((index[0, 0].cpu().item(), index[0, 1].cpu().item()))
            simi_matrix[index[0, 0], :] = -float("inf")
            simi_matrix[index[0, 1], :] = -float("inf")
            simi_matrix[:, index[0, 0]] = -float("inf")
            simi_matrix[:, index[0, 1]] = -float("inf")
        
        if len(stitches) == 0:
            stitches = None
        else:
            stitches = np.array(stitches)
            if stitches.shape[0] != 2:
                stitches = np.transpose(stitches, (1, 0))
        return stitches


    def save_gt_batch_imgs(self, gt_batch, datanames, data_folders, save_to):
        gt_imgs = []
        for idx, (name, folder) in enumerate(zip(datanames, data_folders)):
            gt = {}
            for key in gt_batch:
                gt[key] = gt_batch[key][idx]
                if (('order_matching' in self.config and self.config['order_matching'])
                    or 'origin_matching' in self.config and self.config['origin_matching']
                    or not self.gt_caching):
                    print(f'{self.__class__.__name__}::Warning::Propagating '
                        'information from GT on prediction is not implemented in given context')
                else:
                    if self.gt_caching and folder + '/static' in self.gt_cached:
                        gtc = self.gt_cached[folder + '/static']
                    else:
                        gtc = self._load_ground_truth(folder + "/static")
                    for key in gtc:
                        if key not in gt:
                            gt[key] = gtc[key]
            
            # Transform to pattern object
            pname = os.path.basename(folder) + "__" + os.path.basename(name.replace(".png", ""))
            pattern = self._pred_to_pattern(gt, pname)

            try: 
                # log gt number of panels
                # pattern.spec['properties']['correct_num_panels'] = gtc['num_panels']
                final_dir = pattern.serialize(save_to, to_subfolder=True, tag=f'_gt_')
            except (RuntimeError, InvalidPatternDefError, TypeError) as e:
                print('GarmentDetrDataset::Error::{} serializing skipped: {}'.format(name, e))
                continue

            final_file = pattern.name + '_gt__pattern.png'
            gt_imgs.append(Path(final_dir) / final_file)
        return gt_imgs
    
    def save_prediction_single(self, prediction, dataname="outside_dataset", save_to=None, return_stitches=False):
        

        for key in prediction.keys():
            prediction[key] = prediction[key][0]
        
        pattern = self._pred_to_pattern(prediction, dataname, return_stitches=return_stitches)
        try: 
            final_dir = pattern.serialize(save_to, to_subfolder=True, tag='_predicted_single_')
        except (RuntimeError, InvalidPatternDefError, TypeError) as e:
            print('GarmentDetrDataset::Error::{} serializing skipped: {}'.format(dataname, e))

        final_file = pattern.name + '_predicted__pattern.png'
        prediction_img = Path(final_dir) / final_file
        
        return pattern.pattern['panel_order'], pattern.pattern['new_panel_ids'], prediction_img


    def save_prediction_batch(self, predictions, datanames, data_folders, 
            save_to, features=None, weights=None, orig_folder_names=False, **kwargs):
        """ 
            Saving predictions on batched from the current dataset
            Saves predicted params of the datapoint to the requested data folder.
            Returns list of paths to files with prediction visualizations
            Assumes that the number of predictions matches the number of provided data names"""

        prediction_imgs = []
        for idx, (name, folder) in enumerate(zip(datanames, data_folders)):
            # "unbatch" dictionary
            prediction = {}
            pname = os.path.basename(folder) + "__" + os.path.basename(name.replace(".png", ""))
            tmp_path = os.path.join(save_to, pname, '_predicted_specification.json')
            if os.path.exists(tmp_path):
                continue
            
            print("Progress {}".format(tmp_path))

            for key in predictions:
                prediction[key] = predictions[key][idx]
            if "images" in kwargs:
                prediction["input"] = kwargs["images"][idx]
            if "panel_shape" in kwargs:
                prediction["panel_l2"] = kwargs["panel_shape"][idx]

            # add values from GT if not present in prediction
            if "use_gt_stitches" in kwargs and kwargs["use_gt_stitches"]:
                gt = self._load_ground_truth(folder + "/static")
                for key in gt:
                    if key not in prediction:
                        prediction[key] = gt[key]
            else:
                if "edge_cls" in prediction and "edge_similarity" in prediction:
                    print("Use the predicted stitch infos ")
                else:
                    gt = self.gt_cached[folder + '/static']
                    for key in gt:
                        if key not in prediction:
                            prediction[key] = gt[key]
            # Transform to pattern object
            
            pattern = self._pred_to_pattern(prediction, pname)
            # log gt number of panels
            if self.gt_caching and folder + "/static" in self.gt_cached:
                gt = self.gt_cached[folder + '/static']
                pattern.spec['properties']['correct_num_panels'] = gt['num_panels']
            elif "use_gt_stitches" in kwargs and kwargs["use_gt_stitches"]:
                pattern.spec['properties']['correct_num_panels'] = gt['num_panels']

            try: 
                tag = f'_predicted_{prediction["panel_l2"]}_' if "panel_l2" in prediction else f"_predicted_"
                final_dir = pattern.serialize(save_to, to_subfolder=True, tag=tag)
            except (RuntimeError, InvalidPatternDefError, TypeError) as e:
                print('GarmentDetrDataset::Error::{} serializing skipped: {}'.format(folder, e))
                continue
            final_file = pattern.name + '_predicted__pattern.png'
            prediction_imgs.append(Path(final_dir) / final_file)
            # save input img
            T.ToPILImage()(prediction["input"]).save(os.path.join(final_dir, "input.png")) 
            shutil.copy2(name, str(final_dir))
            # shutil.copy2(name.replace(".png", "cam_pos.json"), str(final_dir))
            shutil.copy2(os.path.join(folder, "static", "spec_config.json"), str(final_dir))

            # copy originals for comparison
            data_prop_file = os.path.join(folder, "data_props.json")
            if os.path.exists(data_prop_file):
                shutil.copy2(data_prop_file, str(final_dir))
        return prediction_imgs
    
    def _pred_to_pattern(self, prediction, dataname, return_stitches=False):
        """Convert given predicted value to pattern object
        """
        # undo standardization  (outside of generinc conversion function due to custom std structure)
        gt_shifts = self.config['standardize']['gt_shift']
        gt_scales = self.config['standardize']['gt_scale']

        for key in gt_shifts:
            if key == 'stitch_tags':  
                # ignore stitch tags update if explicit tags were not used
                continue
            
            pred_numpy = prediction[key].detach().cpu().numpy()
            if key == 'outlines' and len(pred_numpy.shape) == 2: 
                pred_numpy = pred_numpy.reshape(self.config["max_pattern_len"], self.config["max_panel_len"], 4)

            prediction[key] = pred_numpy * gt_scales[key] + gt_shifts[key]

        # recover stitches
        if 'stitches' in prediction:  # if somehow prediction already has an answer
            stitches = prediction['stitches']
        elif 'stitch_tags' in prediction: # stitch tags to stitch list 
            pass
        elif 'edge_cls' in prediction and "edge_similarity" in prediction:
            stitches = self.prediction_to_stitches(prediction['edge_cls'], prediction['edge_similarity'], return_stitches=return_stitches)
        else:
            stitches = None
        
        # Construct the pattern from the data
        pattern = NNSewingPattern(view_ids=False, panel_classifier=self.panel_classifier)
        pattern.name = dataname

        try: 
            pattern.pattern_from_tensors(
                prediction['outlines'], 
                panel_rotations=prediction['rotations'],
                panel_translations=prediction['translations'], 
                stitches=stitches,
                padded=True)   
        except (RuntimeError, InvalidPatternDefError) as e:
            print('GarmentDetrDataset::Warning::{}: {}'.format(dataname, e))
            pass
        
        return pattern
    

if __name__ == '__main__':
    conf = StandardizeConfig(
        outlines=StatsConfig(shift=[-120 , -42, 0.05 , -0.55], scale=[280, 152, 1, 4.15]),
        rotations=StatsConfig(shift=[0, 0], scale=[0.5, 0.5]),
        stitch_tags=StatsConfig(shift=[0, 0], scale=[0.5, 0.5]),
        translations=StatsConfig(shift=[0, 0], scale=[0.5, 0.5]),
        vertices=StatsConfig(shift=[-30, -70], scale=[190, 180])
    )
    dataset = GarmentTokenDataset(
        "/miele/george/sewfactory", 
        50257, 
        conf, 
        "gpt2", 
        256, 
        panel_classification="/home/w4756677/garment/garment_foundation_model/sewformer/SewFormer/assets/data_configs/panel_classes_condenced.json",
        filter_params="/home/w4756677/garment/garment_foundation_model/sewformer/SewFormer/assets/data_configs/param_filter.json")
    residual_error_sum = 0
    all_rots = []
    all_trans = []
    all_num_stitches = []
    for idx in tqdm(range(len(dataset))):
        template_name, spec_sheet = dataset.datapoints_names[idx]
        template_name = "_".join(template_name.split("_")[:-1])
        pattern = NNSewingPattern(
            spec_sheet, panel_classifier=dataset.panel_classifier, 
            template_name=template_name)
        # TODO: no translation and rotations for now
        # pattern_edges, pattern_trans, pattern_rots, panel_names = pattern.patten_as_tokens(version=2)
        outlines, num_edges, num_panels, pattern_rots, pattern_transl, stich_indices, num_stiches, stich_tags= pattern.pattern_as_tensors(with_placement=True, with_stitches=True, with_stitch_tags=True)
        all_num_stitches.append(num_stiches.item())
        # encoded_pattern = dataset.encode_pattern(template_name, pattern_edges, panel_names)
        # encoded_pattern = torch.tensor(encoded_pattern, dtype=torch.long)
        # decoded_pattern_tensor, panel_names, panel_descriptions = dataset.decode_pattern(encoded_pattern.cpu().numpy())
        # panel_order = pattern.panel_order()
        # gt_panel_tensor = []
        # for panel_name in panel_order:
        #     if panel_name is not None:
        #         edges, rot, transl, aug_edges = pattern.panel_as_numeric(panel_name, pad_to_len=decoded_pattern_tensor.shape[1])
        #         gt_panel_tensor.append(edges)
        # gt_panel_tensor = np.stack(gt_panel_tensor)
        # residual_error = np.linalg.norm(gt_panel_tensor - decoded_pattern_tensor) / np.linalg.norm(gt_panel_tensor)
        # residual_error_sum += residual_error
    import ipdb; ipdb.set_trace()

    print(residual_error_sum / len(dataset))
        
        
        
    pattern_tokens = dataset[0]
    decoded_pattern_tensor, panel_names, panel_descriptions = dataset.decode_pattern(pattern_tokens.cpu().numpy())
    pattern = NNSewingPattern(view_ids=False, panel_classifier=None, template_name="recon")
    pattern.pattern_from_tensors(decoded_pattern_tensor, 
                    panel_rotations=None,
                    panel_translations=None, 
                    stitches=None,
                    padded=True, 
                    panel_names=panel_names)
    pattern.name = "recon"
    pattern.serialize("/home/w4756677/garment/garment_foundation_model/sewformer/SewFormer/", to_subfolder=True)
    split_info = {"type": "percent", "split_on": "folder", "valid_per_type": 5, "test_per_type": 10}
    datawrapper = RealisticDatasetDetrWrapper(dataset, batch_size=64)
    datawrapper.load_split(split_info, batch_size=64)
    datawrapper.standardize_data()




    



