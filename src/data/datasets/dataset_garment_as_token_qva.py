import json
import numpy as np
import os
from pathlib import Path, PureWindowsPath
import shutil
import glob
import random
import logging
import warnings
log = logging.getLogger(__name__)
# Hack for suppressing scipy warning of gimbal lock. 
warnings.filterwarnings("ignore", category=UserWarning, message="Gimbal lock detected.*")
from tqdm import tqdm

import torch
import cv2
from torch.utils.data import DataLoader, Dataset, Subset
from transformers import CLIPImageProcessor
from transformers import PreTrainedTokenizer
import torch.nn.functional as F
from torchvision.io import read_image
from data.transforms import ResizeLongestSide
import torchvision.transforms as T
from dataclasses import dataclass
from enum import Enum
import pickle
from typing import List, Dict, Tuple, Union, Any
from data.garment_tokenizers.default_garment_tokenizer import GarmentTokenizer
# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 
root_path = os.path.dirname(os.path.dirname(os.path.abspath(parentdir)))
pkg_path = "{}/SewFactory/packages".format(root_path)
print(pkg_path)
sys.path.insert(0, pkg_path)


# My modules
from data.pattern_converter import NNSewingPattern, InvalidPatternDefError, EmptyPanelError
from data.panel_classes import PanelClasses

from data.datasets.panel_configs import *
from models.llava import conversation as conversation_lib
from data.datasets.utils import SHORT_QUESTION_LIST, ANSWER_LIST, DEFAULT_PLACEHOLDER_TOKEN

class QVAGarmentTokenDatasetConfig():
    root_dir: str 
    vision_tower: str
    image_size: int 
    panel_classification: str
    filter_params: str
        

class QVAGarmentTokenDataset(Dataset):
    def __init__(
        self, 
        root_dir: str, 
        vision_tower: str,
        image_size: int, 
        garment_tokenizer: GarmentTokenizer,
        panel_classification: str,
        filter_params: str,
        load_by_dataname: Optional[List[Tuple[str, str, str]]] = None,
        inference: bool = False,):

        self.root_path = root_dir
        self.inference = inference
        self.garment_tokenizer = garment_tokenizer
        
        self.config = {}
        self.config['class'] = self.__class__.__name__
        self.datapoints_names = []
        self.dataset_start_ids = []  # (folder, start_id) tuples -- ordered by start id
        self.panel_classification = panel_classification
        self.filter_params = filter_params
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.vision_tower=vision_tower
        self.image_size = image_size    
        self.transform = ResizeLongestSide(image_size)
        self.short_question_list = SHORT_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        if load_by_dataname is not None:
            if isinstance(load_by_dataname, str) and load_by_dataname.endswith('.pkl'):
                load_by_dataname = pickle.load(open(load_by_dataname, 'rb'))
            self.datapoints_names = load_by_dataname
            self.dataset_start_ids = []
            for i, datapoint_name in enumerate(self.datapoints_names):
                folder = "_".join(datapoint_name[1])
                self.dataset_start_ids.append((folder, i))
        else:
            try:
                folders = [folder for folder in os.listdir(self.root_path) if os.path.isdir(os.path.join(self.root_path, folder))]
                for folder in tqdm(folders):
                    if os.path.exists(os.path.join(self.root_path, folder, "renders")):
                        self.dataset_start_ids.append((folder, len(self.datapoints_names)))
                        gt_folder = os.path.join(self.root_path, folder, "static")
                        if not os.path.exists(gt_folder):
                            continue
                        spec_sheet = json.load(open(os.path.join(gt_folder, "spec_config.json"), "r"))
                        garment_names = [spec_sheet[k]["spec"].split("\\")[-1] for k in spec_sheet.keys()]
                        garment_spec = os.path.join(folder, "static", folder + "_specification_fixed.json")
                        # We don't have sim2real images
                        # img_names = [os.path.join(self.sim_root, folder, fn) for fn in os.listdir(os.path.join(self.root_path, folder, "renders")) if fn.endswith(".png")]
                        img_names = [os.path.join(folder, "renders", fn) for fn in os.listdir(os.path.join(self.root_path, folder, "renders")) if fn.endswith(".png")]
                        merge_names = [(img_names, garment_names, garment_spec)]
                        self.datapoints_names += merge_names
            except Exception as e:
                print(e)
        self.dataset_start_ids.append((None, len(self.datapoints_names)))
        self.config['size'] = len(self)
        log.info("Total valid datanames is {}".format(self.config['size']))

        self.gt_cached, self.gt_caching = {}, True
        if self.gt_caching:
            log.info('Storing datapoints ground_truth info in memory')
        

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


    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pixel values and pad to a square input."""
        # Normalize colors
        x = (x - self.pixel_mean) / self.pixel_std

        # Pad
        h, w = x.shape[-2:]
        padh = self.img_size - h
        padw = self.img_size - w
        x = F.pad(x, (0, padw, 0, padh))
        return x
    
    def _fetch_pattern(self, idx):
        """Fetch the pattern and image for the given index"""
        _, garment_names, garment_spec = self.datapoints_names[idx]
        gt_pattern = NNSewingPattern(os.path.join(self.root_path, garment_spec))
        gt_pattern.name = "_".join(garment_names)
        return gt_pattern
    
    def get_img_transforms(self):
        return T.Compose([
            T.CenterCrop(400),
            T.Resize(384),
        ])
    
    def _fetch_image(self, idx):
        """Fetch the image for the given index"""
        image_paths, _, _ = self.datapoints_names[idx]
        image_path = random.choice(image_paths)
        image = cv2.imread(os.path.join(self.root_path, image_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (512, 512))
        image = image[56:456, 56:456, :]
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]  # preprocess image for clip
        return image_clip, os.path.join(self.root_path, image_path)
    
    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data. 
        Does not support list indexing"""
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()
        
        # fetch the pattern and image
        gt_pattern = self._fetch_pattern(idx)
        image_clip, image_path = self._fetch_image(idx)

        # encode the pattern
        pattern_dict = self.garment_tokenizer.encode(gt_pattern)
        # questions and answers
        questions = []
        answers = []
        for i in range(1):
            question_template = random.choice(self.short_question_list)
            # question_template = self.short_question_list[i]
            questions.append(question_template)
            answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
            answers.append(answer_template)

        conversations = []
        question_only_convs = []
        conv = conversation_lib.default_conversation.copy()

        i = 0
        while i < len(questions):
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], answers[i])
            conversations.append(conv.get_prompt())
            conv.messages = []
            conv.append_message(conv.roles[0], questions[i])
            conv.append_message(conv.roles[1], "")
            question_only_convs.append(conv.get_prompt())
            i += 1

        # image = self.preprocess(torch.from_numpy(image).permute(2, 0, 1).contiguous())
        return (
            pattern_dict,
            {},
            image_path,
            image_clip,
            conversations,
            question_only_convs,
            questions,
            [gt_pattern],
            1,
            self.inference
        )    

    
    
    def _to_verts(self, panel_edges):
        """Convert normalized panel edges into the vertex representation"""

        vert_list = [np.array([0, 0])]  # always starts at zero
        # edge: first two elements are the 2D vector coordinates, next two elements are curvature coordinates
        for edge in panel_edges:
            next_vertex = vert_list[-1] + edge[:2]
            edge_perp = np.array([-edge[1], edge[0]])

            # NOTE: on non-curvy edges, the curvature vertex in panel space will be on the previous vertex
            #       it might result in some error amplification, but we could not find optimal yet simple solution
            next_curvature = vert_list[-1] + edge[2] * edge[:2]  # X curvature coordinate
            next_curvature = next_curvature + edge[3] * edge_perp  # Y curvature coordinate

            vert_list.append(next_curvature)
            vert_list.append(next_vertex)

        vertices = np.stack(vert_list)

        # align with the center
        vertices = vertices - np.mean(vertices, axis=0)  # shift to average coordinate

        return vertices
    def evaluate_patterns(self, pred_patterns: List[NNSewingPattern], gt_patterns: List[NNSewingPattern]):
        return self.garment_tokenizer.evaluate_patterns(pred_patterns, gt_patterns)
    
    def decode(self, output_dict: Dict[str, Any], tokenizer: PreTrainedTokenizer):
        """Decode output ids to text"""
        return self.garment_tokenizer.decode(output_dict, tokenizer)

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
        
    def random_split(self, valid_per_type, test_per_type, split_type='percent', split_on="pattern"):
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
            [self.datapoints_names[i] for i in train_ids], 
            [self.datapoints_names[i] for i in valid_ids],
            [self.datapoints_names[i] for i in test_ids] if test_size else None
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
            if "_".join(self.datapoints_names[idx][1]) in training_datanames:  # usually the largest, so check first
                train_ids.append(idx)
            elif "_".join(self.datapoints_names[idx][1]) in test_datanames:
                test_ids.append(idx)
            elif "_".join(self.datapoints_names[idx][1]) in valid_datanames:
                valid_ids.append(idx)
            else:
                continue
            
            if idx % 1000 == 0:
                print(f"progress {idx}, #Train_Ids={len(train_ids)}, #Valid_Ids={len(valid_ids)}, #Test_Ids={len(test_ids)}")
        
        return (
            [self.datapoints_names[i] for i in train_ids], 
            [self.datapoints_names[i] for i in valid_ids],
            [self.datapoints_names[i] for i in test_ids] if len(test_ids) > 0 else None
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
        image_names, garment_name, pattern_sheet = self.datapoints_names[idx]
        return image_names, garment_name, pattern_sheet
    
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
        rotations=StatsConfig(shift=[-180, -180, -180], scale=[360, 360, 360]),
        stitch_tags=StatsConfig(shift=[0, 0], scale=[0.5, 0.5]),
        translations=StatsConfig(shift=[-55.25636 , -20.001333, -17.086796], scale=[109.58753, 51.449017, 37.846794]),
        vertices=StatsConfig(shift=[-30, -70], scale=[190, 180])
    )
    dataset = QVAGarmentTokenDataset(
        root_dir = "/miele/george/sewfactory", 
        vision_tower = "openai/clip-vit-large-patch14",
        image_size = 1024, 
        standardize = conf, 
        bin_size = 256,
        inference = True,
        encode_transformations=True,
        panel_classification="/home/w4756677/garment/garment_foundation_model/sewformer/SewFormer/assets/data_configs/panel_classes_condenced.json",
        filter_params="/home/w4756677/garment/garment_foundation_model/sewformer/SewFormer/assets/data_configs/param_filter.json")
    import ipdb; ipdb.set_trace()
    dataset[0]




    



