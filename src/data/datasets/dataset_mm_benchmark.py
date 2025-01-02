import numpy as np
from scipy.sparse import csr_matrix
import os
from pathlib import Path
import shutil
import pickle 
from collections import defaultdict
import sys
import igl
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from scipy.spatial.transform import Rotation as scipy_rot
from typing import List, Dict, Tuple, Union, Optional, Literal
from transformers import CLIPImageProcessor, PreTrainedTokenizer
import cv2
import random
import json
# My
from data.garment_tokenizers.default_garment_tokenizer import GarmentTokenizer
import data.datasets.garmentcodedata.transforms as transforms
from data.datasets.garmentcodedata.pattern_converter import NNSewingPattern, InvalidPatternDefError
from data.datasets.garmentcodedata.external.customconfig import Properties
from data.datasets.garmentcodedata.panel_classes import PanelClasses
from data.datasets.utils import (DEFAULT_IMAGE_TOKEN, SHORT_QUESTION_LIST, 
                                 ANSWER_LIST, 
                                 DEFAULT_PLACEHOLDER_TOKEN, 
                                 DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST, 
                                 SPECULATIVE_TEXT_SHORT_QUESTION_LIST, 
                                 SHORT_QUESTION_WITH_TEXT_LIST,
                                 EDITING_QUESTION_LIST
                                 )
from models.llava import conversation as conversation_lib
from data.transforms import ResizeLongestSide

## incorperating the changes from maria's three dataset classes into a new
## dataset class. this also includes features from sewformer, for interoperability
class GarmentCodeMMBenchmark(Dataset):  
    def __init__(
        self, 
        root_dir: str, 
        editing_dir: str, 
        caption_dir: str, 
        mm_dir: str, # Added that
        sampling_rate: List[int],
        vision_tower: str,
        image_size: int, 
        body_type: Literal['default_body'], 
        data_folders: List[str],
        garment_tokenizer: GarmentTokenizer,
        panel_classification: Optional[str]=None,
        filtered_data_txt: Optional[str] = None,
        load_by_dataname: Optional[List[Tuple[str, str, str]]] = None,
        inference: bool = False): 

        self.editing_dir = editing_dir
        self.caption_dir = caption_dir
        self.mm_dir = mm_dir
        self.sampling_rate = sampling_rate
        self.panel_classifier = None
        self.panel_classification = panel_classification
        self.original_data_folders = data_folders
        self.filtered_data_txt = filtered_data_txt
        
        if filtered_data_txt is not None:
            with open(filtered_data_txt, 'r') as f:
                filtered_data = [line.strip() for line in f.readlines()]
        else:
            filtered_data = None
        self.filtered_data = filtered_data

        #################################
        # init from the basedataset class
        self.root_path = Path(root_dir)
        self.config = {}
        self.config['class'] = self.__class__.__name__

        # Load for appropriate body type
        self.body_type = body_type
        self.data_folders = [f'{df}/{self.body_type}' for df in data_folders]
        self.data_folders_nicknames = dict(zip(self.data_folders, data_folders))

        self.datapoints_names = [] # contains all the garments by id 
        self.dataset_start_ids = [] # (folder, start_id) tuples -- ordered by start id
        self.panel_classes = []
        self.dict_panel_classes = defaultdict(int)
        self.garment_tokenizer = garment_tokenizer
        self.inference = inference
        
        self.clip_image_processor = CLIPImageProcessor.from_pretrained(vision_tower)
        self.vision_tower=vision_tower
        self.image_size = image_size    
        self.transform = ResizeLongestSide(image_size)

        # Specific for original task
        self.short_question_list = SHORT_QUESTION_LIST
        self.descriptive_text_question_list = DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST
        self.speculative_text_question_list = SPECULATIVE_TEXT_SHORT_QUESTION_LIST
        self.text_image_question_list = SHORT_QUESTION_WITH_TEXT_LIST
        self.editing_question_list = EDITING_QUESTION_LIST
        self.answer_list = ANSWER_LIST

        self.information = defaultdict(dict)  # TODO REMOVE Needed? Should be part of experiment object/initial config? I don't see it used at all 
        ## collect information about data folders: 

        if load_by_dataname is not None:
            if isinstance(load_by_dataname, str) and load_by_dataname.endswith('.txt'):
                with open(load_by_dataname, 'r') as f:
                    load_by_dataname = [line.strip() for line in f.readlines()]
            self.datapoints_names = load_by_dataname
            # Wrong Don't use
            self.dataset_start_ids = [(dn, i) for i, dn in enumerate(load_by_dataname)]
        else:
            for data_folder in self.data_folders: 
            
                print(f'Loading data from {self.root_path / data_folder}')  # DEBUG
                try:
                    _, dirs, _ = next(os.walk(self.root_path / data_folder))
                except StopIteration:
                    print(f"{self.root_path / data_folder} is empty!")
                    continue
                
                self.information[data_folder]['all_datapoints'] = dirs
                
                non_empty_dirs = self.non_empty_dirs(dirs, (self.root_path / data_folder))
                # TODO stanford this can be eliminated / abridged  
                self.information[data_folder]['non_empty_datapoints'] = non_empty_dirs

                if panel_classification is None:
                    new_panel_classes = self._collect_panel_classes(non_empty_dirs, (self.root_path / data_folder))
                    for key in new_panel_classes.keys():
                        self.dict_panel_classes[key] += new_panel_classes[key]
                
                self.dataset_start_ids.append((data_folder, len(self.datapoints_names)))
                # dataset name as part of datapoint name
                datapoints_names = [data_folder + '/' + name for name in non_empty_dirs]
                
                #  we currently do not need to clean anything yet  
                # todo stanford this can already be done beforehand - we do not need to keep the failure cases in the dataset
                clean_list = self._clean_datapoint_list(datapoints_names, data_folder)
                self.information[data_folder]['clean_datapoints'] = clean_list
                # clean_list = datapoints_names

                self.datapoints_names += clean_list
                print('Data folder {} has {} clean datapoints'.format(data_folder, len(clean_list)))
            self.dataset_start_ids.append((None, len(self.datapoints_names)))  # add the total len as item for easy slicing
            self.config['size'] = len(self)
           
                
        if panel_classification is not None:
            self.panel_classifier = PanelClasses(classes_file=panel_classification)
            self.panel_classes = self.panel_classifier.classes
        else:
            self.panel_classes = list(sorted(self.dict_panel_classes, key=self.dict_panel_classes.get, reverse=True))
            self.panel_classifier = PanelClasses(self.panel_classes)

        print("The panel classes in this dataset are :", self.panel_classes)
        

        self.gt_cached = {}
        self.gt_caching = True

        # Use default tensor transform + the ones from input
        self.transforms = [transforms.SampleToTensor()]

        ########################################
        # end of init from the basedataset class

        # To make sure the datafolder names are unique after updates
        # precaution, taken from maria's code
        all_nicks = self.data_folders_nicknames.values()
        if len(all_nicks) > len(set(all_nicks)):
            print('{}::Warning::Some data folder nicknames are not unique: {}. Reverting to the use of original folder names'.format(
                self.__class__.__name__, self.data_folders_nicknames
            ))
            self.data_folders_nicknames = dict(zip(self.data_folders, self.data_folders))
        
        # self._drop_cache()
        self.gt_cached = {}


        self.ready = True

    def non_empty_dirs(self, dirs, root_path):
        """Collect non-empty directories in the dataset
            Usable sample directory should contain a spec file and a geometry file (obj or point cloud)
        """
        non_empty = []
        for dir in dirs:
            dir_path = root_path / dir
            _, _, files = next(os.walk(dir_path))

            spec_list = [f for f in files if 'specification_shifted.json' in f]
            img_list = [file for file in files if 'render_front.png' in file or "render_back.png" in file]

            if len(spec_list) and len(img_list):
                non_empty.append(dir)
        
        return non_empty

    def _pattern_size(self):
        num_panels = []
        num_edges_in_panel = []
        num_stitches = []

        for data_point in self.datapoints_names: 
            folder_elements = [file.name for file in (self.root_path / data_point).glob('*')]
            pattern_flat, _, _, stitches, _, _, _ = self._read_pattern(data_point, folder_elements, with_stitches=True)
            num_panels.append(pattern_flat.shape[0])
            num_edges_in_panel.append(pattern_flat.shape[1])  # after padding
            num_stitches.append(stitches.shape[1])

        return num_panels, num_edges_in_panel, num_stitches

    # my method for colleting all the panel classes in the dataset 
    # might need to be seperated into a different class to be evaluated
    # independent of every run of the dataset
    def _collect_panel_classes(self, dirs, root_path):
        all_classes = defaultdict(int)

        for dir in dirs:
            dir_path = root_path / dir
            _, _, files = next(os.walk(dir_path))

            spec = [f for f in files if 'specification.json' in f]
            if len(spec) == 0:
                continue

            # TODO use it? Since it's loaded
            pattern = NNSewingPattern(dir_path / spec[0])
            panel_names = pattern.pattern['panels'].keys() 
            for p in panel_names:
                all_classes[p] += 1

        return all_classes

    # added from maria
    def _estimate_data_shape(self):
            """Get sizes/shapes of a datapoint for external references"""
            elem = self[0]
            feature_size = elem['features'].shape[0]
            gt_size = elem['ground_truth'].shape[0] if hasattr(elem['ground_truth'], 'shape') else None

            self.config['feature_size'], self.config['ground_truth_size'] = feature_size, gt_size
    
    # added from maria
    def update_config(self, in_config):
        """Define dataset configuration:
            * to be part of experimental setup on wandb
            * Control obtainign values for datapoints"""
        self.config.update(in_config)

        # check the correctness of provided list of datasets
        if ('data_folders' not in self.config 
                or not isinstance(self.config['data_folders'], list)
                or len(self.config['data_folders']) == 0):
            raise RuntimeError('BaseDataset::Error::information on datasets (folders) to use is missing in the incoming config')

    # added from maria
    def _read_pattern(self, datapoint_name, folder_elements, 
                      pad_panels_to_len=None, pad_panel_num=None, pad_stitches_num=None,
                      with_placement=False, with_stitches=False, with_stitch_tags=False):
        """Read given pattern in tensor representation from file"""
        spec_list = [file for file in folder_elements if 'specification.json' in file]
        if not spec_list:
            raise RuntimeError('GarmentBaseDataset::Error::*specification.json not found for {}'.format(datapoint_name))
        pattern = NNSewingPattern(
            self.root_path / datapoint_name / spec_list[0], 
            panel_classifier=self.panel_classifier, 
            template_name=self.template_name(datapoint_name))
        return pattern.pattern_as_tensors(
            pad_panels_to_len, pad_panels_num=pad_panel_num, pad_stitches_num=pad_stitches_num,
            with_placement=with_placement, with_stitches=with_stitches, 
            with_stitch_tags=with_stitch_tags)

    # added from maria 
    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints_names)  

    def _fetch_pattern(self, idx):
        """Fetch the pattern and image for the given index"""
        _, garment_names, garment_spec = self.datapoints_names[idx]
        gt_pattern = NNSewingPattern(garment_spec)
        gt_pattern.name = "_".join(garment_names)
        return gt_pattern
    
    def _prepare_image(self, image_paths):
        """Fetch the image for the given index"""
        image_path = random.choice(image_paths)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_clip = self.clip_image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0]  # preprocess image for clip
        return image_clip, image_path
    
    def get_mode_names(self):
        return ['image', 'description','occasion', 'text_image', 'editing']

    # added from maria 
    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data. 
        Does not support list indexing"""

        # TODO(jan): CHANGE THIS
        
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        datapoint_name = self.datapoints_names[idx]
        data_name = datapoint_name.split('/')[-1]

        #### DONT TOUCH
        if data_name in self.gt_cached:
            gt_pattern, pattern_dict, edited_pattern, edited_pattern_dict, editing_captions, captions = self.gt_cached[data_name]
        else:
            spec_file = os.path.join(self.root_path, datapoint_name, f'{data_name}_specification_shifted.json')
            gt_pattern = NNSewingPattern(spec_file, panel_classifier=self.panel_classifier, template_name=self.template_name(datapoint_name))
            gt_pattern.name = data_name
            pattern_dict = self.garment_tokenizer.encode(gt_pattern)
            
            # maybe adapt paths
            editing_spec_file = os.path.join(self.editing_dir, data_name, f'edited_specification.json')
            editing_caption_json = os.path.join(self.editing_dir, data_name, f'editing_caption.json')
            if (not os.path.exists(editing_spec_file)) or (not os.path.exists(editing_caption_json)):
                edited_pattern = None
                edited_pattern_dict = None
                editing_captions = None
            else:
                edited_pattern = NNSewingPattern(editing_spec_file, panel_classifier=self.panel_classifier, template_name=self.template_name(datapoint_name))
                edited_pattern.name = data_name
                edited_pattern_dict = self.garment_tokenizer.encode(edited_pattern)
                editing_captions = json.load(open(editing_caption_json, 'r'))
            
            caption_json = os.path.join(self.caption_dir, data_name, f'captions.json')
            if os.path.exists(caption_json):
                captions = json.load(open(caption_json, 'r'))
            else:
                captions = None
            
            self.gt_cached[data_name] = (gt_pattern, pattern_dict, edited_pattern, edited_pattern_dict, editing_captions, captions)
        #### DONT TOUCH END
            
            
            # make sure to have that as default if no iamge is given
            image_clip = torch.zeros((3, 224, 224))

            with open(os.path.join(self.mm_dir, data_name, 'desc.json'), 'r') as f:
                data = json.load(f)
            
            assert data

            if 'image_path' in data:
                image_clip, image_path = self._prepare_image([data['image_path']])
            
            text_data_point: str = data['text']

            # questions and answers
            questions = []
            answers = []

            for i in range(1):
                question_template = DEFAULT_IMAGE_TOKEN + "\n" + text_data_point + " Respond with a sewing pattern."
                questions.append(question_template)
                answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
                answers.append(answer_template)
            
            if data['editing']:
                out_pattern = [edited_pattern, gt_pattern]
                if data['editing_direction'] == 'forward':
                    before_pattern_dict = pattern_dict
                    after_pattern_dict = edited_pattern_dict
                else:
                    before_pattern_dict = edited_pattern_dict
                    after_pattern_dict = pattern_dict

                out_pattern_dict = {k: before_pattern_dict[k] + after_pattern_dict[k] for k in before_pattern_dict.keys()}
                question_pattern_dict = before_pattern_dict

            else:
                out_pattern = [gt_pattern]
                out_pattern_dict = pattern_dict
                question_pattern_dict = {}

            # empty or input_pattern
            question_pattern_dict = before_pattern_dict

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
        
        sample_type = 4 if data['editing'] else 3

        return (
            out_pattern_dict,
            question_pattern_dict,
            image_path,
            image_clip,
            conversations,
            question_only_convs,
            questions,
            out_pattern,
            sample_type,
            self.inference
        ) 
        
    def evaluate_patterns(self, pred_patterns: List[NNSewingPattern], gt_patterns: List[NNSewingPattern]):
        return self.garment_tokenizer.evaluate_patterns(pred_patterns, gt_patterns)
    
    def _as_vertices(self, outlines, num_edges):
        
        
        vertices = np.zeros(outlines.shape)
        
        valid_edges = outlines[num_edges > 0]
        valid_vertices = outlines[num_edges > 0]

        #vertices themselves
        for i in range(valid_edges.shape[1]):
            valid_vertices[:,i,:2] += valid_edges[:, i, :2]

        # for control points 
        for i in range(valid_edges.shape[0]):
            for j in range(valid_edges.shape[1]):
                if valid_edges[i, j, 2:4].sum() > 0:
                    valid_vertices[i,j,2:4] = NNSewingPattern.control_to_abs_coord(valid_vertices[i,j,:2], valid_vertices[i,j+1,:2], valid_edges[i,j,2:4])
                if valid_edges[i, j, 4:6].sum() > 0:
                    valid_vertices[i,j,4:6] = NNSewingPattern.control_to_abs_coord(valid_vertices[i,j,:2], valid_vertices[i+1,j,:2], valid_edges[i,j,4:6])
        
        vertices[num_edges > 0] = valid_vertices
        
        return vertices
    
    
    # added from maria
    def _get_pattern_ground_truth(self, datapoint_name, folder_elements):
        """Get the pattern representation with 3D placement"""
        pattern, num_edges, num_panels, rots, tranls, stitches, num_stitches, stitch_adj, stitch_tags, aug_outlines = self._read_pattern(
            datapoint_name, folder_elements, 
            pad_panels_to_len=self.config['max_panel_len'],
            pad_panel_num=self.config['max_pattern_len'],
            pad_stitches_num=self.config['max_num_stitches'],
            with_placement=True, with_stitches=True, with_stitch_tags=True)
        inv_free_edges_mask = self.inv_free_edges_mask(self.config['max_pattern_len'], self.config['max_panel_len'], stitches, num_stitches)
        # FIXME Condition on the order-invariant lossempty_panels_mask = self._empty_panels_mask(num_edges)  # useful for evaluation
        
        if self.rot_type == 'euler':
            pos_rots = rots[num_edges > 0]
            pos_rots = scipy_rot.from_quat(pos_rots).as_euler('xyz', degrees=True)
            rots = np.zeros((rots.shape[0], 3))
            rots[num_edges > 0] = pos_rots

        ground_truth = {
            'outlines': pattern, 'num_edges': num_edges,
            'rotations': rots, 'translations': tranls, 
            'num_panels': num_panels, 
            # FIXME Condition on the order-invariant loss 'empty_panels_mask': empty_panels_mask,   
            'num_stitches': num_stitches,
            'stitches': stitches, 'free_edges_mask': inv_free_edges_mask,  
            # NOTE: not used in the current setup 'stitch_tags': stitch_tags,
            # FIXME exclude properly 'stitch_adj': stitch_adj   # NOTE: Only needed for quality checks on SF in final configuration
        }
    
        if aug_outlines[0] is not None and 'use_augmented_edges' in self.config and self.config['use_augmented_edges']:   # NOTE: Switches augmentation on and off
            ground_truth.update({"aug_outlines": aug_outlines})

        # stitches post-eval (used to be in get_item in Sewformer code)
        # TODOLOW this should in be in coverter though
        # TODO Not needed for NT?
        if stitch_adj is not None:
            masked_stitches, stitch_edge_mask, reindex_stitches = self.match_edges(
                inv_free_edges_mask,
                stitches=stitches,
                max_num_stitch_edges=2 * self.config["max_num_stitches"]
            )
            label_indices = self.split_pos_neg_pairs(
                reindex_stitches, 
                num_max_edges=self.config['max_panel_len'],   #NOTE: way smaller number then original SewFormer
                num_max_stitches=self.config["max_num_stitches"]
            )  
            ground_truth.update({"masked_stitches": masked_stitches,
                                 "stitch_edge_mask": stitch_edge_mask,
                                 # NOTE: not used "reindex_stitches": reindex_stitches,
                                 "label_indices": label_indices}) 
            
        return ground_truth


    # added from maria
    # does not make sense in the modern code - needs to be reconsidered, no templates used! 
    def template_name(self, datapoint_name):
        """Get name of the garment template from the path to the datapoint"""
        # TODO Clean this up
        return "new_template"
        return self.data_folders_nicknames[datapoint_name.split('/')[0]]



     # ----- Mesh tools -----
    
    # entire section added from maria
    # helpful likely to actually sample the mesh points
    def _sample_points(self, datapoint_name, folder_elements):
        """Make a sample from the 3d surface from a given datapoint files

            Returns: sampled points and vertices of original mesh
        
        """
        pc_list = [file for file in folder_elements if '.xyz' in file]
        if pc_list:  # We found a point cloud already sampled!
            with open(str(self.root_path / datapoint_name / pc_list[0]), 'r') as f:
                points = np.loadtxt(f)
            # FIXME Load vertices (or segmentation of point clouds directly) as well
            verts = None

        else:  # Sample from obj -- slower:
            obj_list = [file for file in folder_elements if self.config['obj_filetag'] in file and '.ply' in file]
            if not obj_list:
                raise RuntimeError('Dataset:Error: geometry file *{}*.ply not found for {}'.format(self.config['obj_filetag'], datapoint_name))
            
            verts, faces = igl.read_triangle_mesh(str(self.root_path / datapoint_name / obj_list[0]))
            points = self.sample_mesh_points(self.config['mesh_samples'], verts, faces)

        # add gaussian noise
        if self.config['point_noise_w']:
            points += np.random.normal(loc=0.0, scale=self.config['point_noise_w'], size=points.shape)

        return points, verts

    # NOTE: SewFormer methods (next two)
    @staticmethod
    def match_edges(inv_free_edge_mask, stitches=None, max_num_stitch_edges=56):
        """
            * max_num_stitch_edges -- max number of edges that participate in any stitch.
                Usually 2x max number of stitches
        """
        stitch_edges = np.ones((1, max_num_stitch_edges)) * (-1)
        valid_edges = np.asarray(inv_free_edge_mask.todense()).reshape(-1).nonzero()   # All the valid edges in a batch
        stitch_edge_mask = np.zeros((1, max_num_stitch_edges))
        if stitches is not None:
            stitches = np.transpose(stitches)
            reindex_stitches = csr_matrix((max_num_stitch_edges, max_num_stitch_edges))
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
                    reindex_stitches[reindex_i, reindex_j] = 1
                    reindex_stitches[reindex_j, reindex_i] = 1
        
        return stitch_edges * stitch_edge_mask, stitch_edge_mask, reindex_stitches
    
    @staticmethod
    def split_pos_neg_pairs(stitches, num_max_edges=3000, num_max_stitches=1000):
        stitch_ind = np.triu_indices_from(stitches[0], 1)
        pos_ind = [[stitch_ind[0][i], stitch_ind[1][i]] for i in range(stitch_ind[0].shape[0]) if stitches[stitch_ind[0][i], stitch_ind[1][i]]]
        neg_ind = [[stitch_ind[0][i], stitch_ind[1][i]] for i in range(stitch_ind[0].shape[0]) if not stitches[stitch_ind[0][i], stitch_ind[1][i]]]

        assert len(neg_ind) >= num_max_edges
        neg_ind = neg_ind[:num_max_edges]
        pos_inds = np.expand_dims(np.array(pos_ind), axis=1)
        neg_inds = np.repeat(np.expand_dims(np.array(neg_ind), axis=0), repeats=pos_inds.shape[0], axis=0)
        indices = np.concatenate((pos_inds, neg_inds), axis=1, dtype=np.int64)

        return indices


    @staticmethod
    def sample_mesh_points(num_points, verts, faces):
        """A routine to sample requested number of points from a given mesh
            Returns points in world coordinates"""
        # changed due to differnet version of igl 
        
        points, _, _  = igl.random_points_on_mesh(num_points, verts, faces)

        return points
    
    def _point_classes_from_mesh(self, points, verts, datapoint_name, folder_elements):
        """Map segmentation from original mesh to sampled points"""
        
        # load segmentation
        seg_path_list = [file for file in folder_elements if self.config['obj_filetag'] in file and 'segmentation.txt' in file]

        with open(str(self.root_path / datapoint_name / seg_path_list[0]), 'r') as f:
            vert_labels = np.array([line.rstrip() for line in f])  # remove \n
        map_list, _, _ = igl.snap_points(points, verts)

        if len(verts) > len(vert_labels):
            point_segmentation = np.zeros(len(map_list))
            print(f'{self.__class__.__name__}::{datapoint_name}::WARNING::Not enough segmentation labels -- {len(vert_labels)} for {len(verts)} vertices. Setting segmenations to zero')

            return point_segmentation.astype(np.int64)

        point_segmentation_names = vert_labels[map_list]

        # find those that map to stitches and assign them the closest panel label
        # Also doing this for occasional 'None's 
        stitch_points_ids = np.logical_or(
            np.char.startswith(point_segmentation_names, 'stitch'), point_segmentation_names == 'None')
        non_stitch_points_ids = np.logical_and(
            (~np.char.startswith(point_segmentation_names, 'stitch')), point_segmentation_names != 'None')

        map_stitches, _, _ = igl.snap_points(points[stitch_points_ids], points[non_stitch_points_ids])

        non_stitch_points_ids = np.flatnonzero(non_stitch_points_ids)
        point_segmentation_names[stitch_points_ids] = point_segmentation_names[non_stitch_points_ids[map_stitches]]

        # Map class names to int ids of loaded classes!
        if self.panel_classifier is not None:
            point_segmentation = self.panel_classifier.map(
                self.template_name(datapoint_name), point_segmentation_names)
        else:
            # assign unique ids within given list
            unique_names = np.unique(self.panel_classes)
            unique_dict = {name: idx for idx, name in enumerate(unique_names)}
            point_segmentation = np.empty(len(point_segmentation_names))
            for idx, name in enumerate(point_segmentation_names):
                point_segmentation[idx] = unique_dict[name]

        return point_segmentation.astype(np.int64)   # type conversion for PyTorch NLLoss

    def _clean_datapoint_list(self, datapoints_names, dataset_folder):

        try: 
            dataset_props = Properties(self.root_path / dataset_folder / f'dataset_properties_{self.body_type}.yaml')
        except FileNotFoundError:
            # missing dataset props file -- skip failure processing
            print(f'{self.__class__.__name__}::Warning::No `dataset_properties_{self.body_type}.json` found. Using all datapoints without filtering.')
            self.data_folders_nicknames[dataset_folder] = dataset_folder
            return datapoints_names

        fails_dict = dataset_props['sim']['stats']['fails']
        for subsection in fails_dict: 
            for fail in fails_dict[subsection]: 
                try: 
                    datapoints_names.remove(dataset_folder + '/' + fail)
                except ValueError:
                    pass
        if self.filtered_data is not None:
            for datapoint in self.filtered_data:
                try: 
                    datapoints_names.remove(datapoint)
                except ValueError:
                    pass
        
        return datapoints_names

    # TODO Why stitch pairs? 
    def _load_from_data(self, datapoint_name):

        # Get stitch pairs & mask from spec
        folder_elements = [file.name for file in (self.root_path / datapoint_name).glob('*')]
        spec_list = [file for file in folder_elements if 'specification.json' in file]

        # Load from prediction if exists
        predicted_list = [file for file in spec_list if 'predicte' in file]
        spec = predicted_list[0] if len(predicted_list) > 0 else spec_list[0]

        pattern = NNSewingPattern(self.root_path / datapoint_name / spec)
        print(self.root_path / datapoint_name / spec)

        if self.config['random_pairs_mode']:
            features, ground_truth = pattern.stitches_as_3D_pairs(
                self.config['stitched_edge_pairs_num'], self.config['non_stitched_edge_pairs_num'],
                self.config['shuffle_pairs'], self.config['shuffle_pairs_order'])
        else:
            features, _, ground_truth = pattern.all_edge_pairs()

        # save elements
        if self.gt_caching and self.feature_caching:
            self.gt_cached[datapoint_name] = ground_truth
            self.feature_cached[datapoint_name] = features

        return features, ground_truth
     
    @staticmethod
    def inv_free_edges_mask(max_panels, max_edges, stitches, num_stitches):
        """
        Construct the mask to identify edges that are not connected to any other
        """
        mask = np.ones((max_panels, max_edges), dtype=bool)

        for side in stitches[:, :num_stitches]:  # ignore the padded part
            for edge_id in side:
                mask[edge_id // max_edges][edge_id % max_edges] = False
        
        return csr_matrix(~mask)

    def _empty_panels_mask(self, num_edges):
        """Empty panels as boolean mask"""

        mask = np.zeros(len(num_edges), dtype=bool)
        mask[num_edges == 0] = True

        return mask

    # using marias version of the standardization, also adopted by sea_ai's implementation
    def standardize(self, training=None):
        """Use shifting and scaling for fitting data to interval comfortable for NN training.
            Accepts either of two inputs: 
            * training subset to calculate the data statistics -- the stats are only based on training subsection of the data
            * if stats info is already defined in config, it's used instead of calculating new statistics (usually when calling to restore dataset from existing experiment)
            configuration has a priority: if it's given, the statistics are NOT recalculated even if training set is provided
                => speed-up by providing stats or speeding up multiple calls to this function
        """
        print('Garment3DPatternFullDataset::Using data normalization for features & ground truth')

        if 'standardize' in self.config:
            print('{}::Using stats from config'.format(self.__class__.__name__))
            stats = self.config['standardize']
        elif training is not None:
            loader = DataLoader(training, batch_size=len(training), shuffle=False)
            for batch in loader:
                feature_shift, feature_scale = self._get_distribution_stats(batch['features'], padded=False)

                gt = batch['ground_truth']
                panel_shift, panel_scale = self._get_distribution_stats(gt['outlines'], padded=True)
                # NOTE mean values for panels are zero due to loop property 
                # panel components SHOULD NOT be shifted to keep the loop property intact 
                panel_shift[0] = panel_shift[1] = 0
                # Last component is a class, so the shift/scale doesn't apply to it
                panel_shift[-1] = 0.  # TODO: Debug!
                panel_scale[-1] = 1.

                # Use min\scale (normalization) instead of Gaussian stats for translation
                # No padding as zero translation is a valid value
                transl_min, transl_scale = self._get_distribution_stats(gt['translations'])
                rot_min, rot_scale = self._get_norm_stats(gt['rotations'])

                # stitch tags if given
                if 'stitch_tags' in gt:
                    st_tags_min, st_tags_scale = self._get_norm_stats(gt['stitch_tags'])

                break  # only one batch out there anyway

            self.config['standardize'] = {
                'f_shift': feature_shift.cpu().numpy(), 
                'f_scale': feature_scale.cpu().numpy(),
                'gt_shift': {
                    'outlines': panel_shift.cpu().numpy(), 
                    'rotations': rot_min.cpu().numpy(),
                    'translations': transl_min.cpu().numpy(), 
                    'stitch_tags': st_tags_min.cpu().numpy() if 'stitch_tags' in gt else None
                },
                'gt_scale': {
                    'outlines': panel_scale.cpu().numpy(), 
                    'rotations': rot_scale.cpu().numpy(),
                    'translations': transl_scale.cpu().numpy(),
                    'stitch_tags': st_tags_scale.cpu().numpy()  if 'stitch_tags' in gt else None
                }
            }
            stats = self.config['standardize']
        else:  # nothing is provided
            raise ValueError('Garment3DPatternFullDataset::Error::Standardization cannot be applied: supply either stats in config or training set to use standardization')

        # clean-up tranform list to avoid duplicates
        self.transforms = [t for t in self.transforms if not isinstance(t, transforms.GTtandartization) and not isinstance(t, transforms.FeatureStandartization)]

        self.transforms.append(transforms.GTtandartization(stats['gt_shift'], stats['gt_scale']))
        self.transforms.append(transforms.FeatureStandartization(stats['f_shift'], stats['f_scale']))

    #######################################################################################
    ### For trainining - implemneted after the initial dataset class was created
    #######################################################################################

    # taken from marias code
    def random_split_by_dataset(self, valid_per_type, test_per_type=0, split_type='count', with_breakdown=False):
        """
            Produce subset wrappers for training set, validations set, and test set (if requested)
            Supported split_types: 
                * split_type='percent' takes a given percentage of the data for evaluation subsets. It also ensures the equal 
                proportions of elements from each datafolder in each subset -- according to overall proportions of 
                datafolders in the whole dataset
                * split_type='count' takes this exact number of elements for the elevaluation subselts from each datafolder. 
                    Maximizes the use of training elements, and promotes fair evaluation on uneven datafolder distribution. 

        Note: 
            * it's recommended to shuffle the training set on batching as random permute is not 
            guaranteed in this function
        """

        if split_type != 'count' and split_type != 'percent':
            raise NotImplementedError('{}::Error::Unsupported split type <{}> requested'.format(
                self.__class__.__name__, split_type))

        train_ids, valid_ids, test_ids = [], [], []

        train_breakdown, valid_breakdown, test_breakdown = {}, {}, {}

        for dataset_id in range(len(self.data_folders)):
            folder_nickname = self.data_folders_nicknames[self.data_folders[dataset_id]]

            start_id = self.dataset_start_ids[dataset_id][1]
            end_id = self.dataset_start_ids[dataset_id + 1][1]   # marker of the dataset end included
            data_len = end_id - start_id

            permute = (torch.randperm(data_len) + start_id).tolist()

            # size defined according to requested type
            valid_size = int(data_len * valid_per_type / 100) if split_type == 'percent' else valid_per_type
            test_size = int(data_len * test_per_type / 100) if split_type == 'percent' else test_per_type



            train_size = data_len - valid_size - test_size

            train_sub, valid_sub = permute[:train_size], permute[train_size:train_size + valid_size]

            train_ids += train_sub
            valid_ids += valid_sub

            if test_size:
                test_sub = permute[train_size + valid_size:train_size + valid_size + test_size]
                test_ids += test_sub
            
            if with_breakdown:
                train_breakdown[folder_nickname] = Subset(self, train_sub)
                valid_breakdown[folder_nickname] = Subset(self, valid_sub)
                test_breakdown[folder_nickname] = Subset(self, test_sub) if test_size else None

        if with_breakdown:
            return (
                Subset(self, train_ids), 
                Subset(self, valid_ids),
                Subset(self, test_ids) if test_per_type else None, 
                train_breakdown, valid_breakdown, test_breakdown
            )
            
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
            if self.datapoints_names[idx] in training_datanames:  # usually the largest, so check first
                train_ids.append(idx)
            elif self.datapoints_names[idx] in test_datanames:
                test_ids.append(idx)
            elif self.datapoints_names[idx] in valid_datanames:
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
    def subsets_per_datafolder(self, index_list=None):
        """
            Group given indices by datafolder and Return dictionary with Subset objects for each group.
            if None, a breakdown for the full dataset is given
        """
        if index_list is None:
            index_list = range(len(self))
        per_data, _ = self.indices_by_data_folder(index_list)
        breakdown = {}
        for folder, ids_list in per_data.items():
            breakdown[self.data_folders_nicknames[folder]] = Subset(self, ids_list)
        return breakdown
    
    def indices_by_data_folder(self, index_list):
        """
            Separate provided indices according to dataset folders used in current dataset
        """
        ids_dict = dict.fromkeys(self.data_folders)  # consists of elemens of index_list
        ids_mapping_dict = dict.fromkeys(self.data_folders)  # reference to the elements in index_list
        index_list = np.array(index_list)
        
        # assign by comparing with data_folders start & end ids
        # enforce sort Just in case
        self.dataset_start_ids = sorted(self.dataset_start_ids, key=lambda idx: idx[1])

        for i in range(0, len(self.dataset_start_ids) - 1):
            ids_filter = (index_list >= self.dataset_start_ids[i][1]) & (index_list < self.dataset_start_ids[i + 1][1])
            ids_dict[self.dataset_start_ids[i][0]] = index_list[ids_filter]
            ids_mapping_dict[self.dataset_start_ids[i][0]] = np.flatnonzero(ids_filter)
        
        return ids_dict, ids_mapping_dict
    
    # def split_from_dict(self, split_dict, with_breakdown=False):
    #     """
    #         Reproduce the data split in the provided dictionary: 
    #         the elements of the currect dataset should play the same role as in provided dict
    #     """
    #     train_ids, valid_ids, test_ids = [], [], []
    #     train_breakdown, valid_breakdown, test_breakdown = {}, {}, {}

    #     # Allow to use the same split for both random and default bodies
    #     if self.body_type not in split_dict['training'][0]:   # Wrong body type in loaded data split. If correct, skip this step
    #         for key in ('training', 'validation', 'test'):
    #             for i in range(len(split_dict[key])):
    #                 name = split_dict[key][i]
    #                 new_name = name.replace('default_body' if 'default_body' in name else 'random_body', self.body_type)
    #                 split_dict[key][i] = new_name
        
    #     training_datanames = set(split_dict['training'])
    #     valid_datanames = set(split_dict['validation'])
    #     test_datanames = set(split_dict['test'])
    #     for idx in range(len(self.datapoints_names)):
    #         if self.datapoints_names[idx] in training_datanames:  # usually the largest, so check first
    #             train_ids.append(idx)
    #         elif len(test_datanames) > 0 and self.datapoints_names[idx] in test_datanames:
    #             test_ids.append(idx)
    #         elif len(valid_datanames) > 0 and self.datapoints_names[idx] in valid_datanames:
    #             valid_ids.append(idx)
    #         # othervise, just skip

    #     if with_breakdown:
    #         train_breakdown = self.subsets_per_datafolder(train_ids)
    #         valid_breakdown = self.subsets_per_datafolder(valid_ids)
    #         test_breakdown = self.subsets_per_datafolder(test_ids)

    #         return (
    #             Subset(self, train_ids), 
    #             Subset(self, valid_ids),
    #             Subset(self, test_ids) if len(test_ids) > 0 else None,
    #             train_breakdown, valid_breakdown, test_breakdown
    #         )

    #     return (
    #         Subset(self, train_ids), 
    #         Subset(self, valid_ids),
    #         Subset(self, test_ids) if len(test_ids) > 0 else None
    #     )

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
    
    def _unpad(self, element, tolerance=1.e-5):
        """Return copy of input element without padding from given element. Used to unpad edge sequences in pattern-oriented datasets"""
        # NOTE: might be some false removal of zero edges in the middle of the list.
        if torch.is_tensor(element):        
            bool_matrix = torch.isclose(element, torch.zeros_like(element), atol=tolerance)  # per-element comparison with zero
            selection = ~torch.all(bool_matrix, axis=1)  # only non-zero rows
        else:  # numpy
            selection = ~np.all(np.isclose(element, 0, atol=tolerance), axis=1)  # only non-zero rows
        return element[selection]
    
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

    def save_to_wandb(self, experiment):
        """Save data cofiguration to current expetiment run"""
        experiment.add_config('dataset', self.config)

        # FIXME Use experiment.save_file()
        # dataset props files
        # FIXME props files now have different name!
        for dataset_folder in self.data_folders:
            try:
                shutil.copy(
                    self.root_path / dataset_folder / 'dataset_properties.json', 
                    experiment.local_wandb_path() / (dataset_folder + '_properties.json'))
            except FileNotFoundError:
                pass
        
        # panel classes
        if self.panel_classifier is not None:
            self.panel_classifier.save_to(experiment.local_wandb_path() / ('panel_classes.json'))

        # param filter file
        if 'filter_by_params' in self.config and self.config['filter_by_params']:
            shutil.copy(
                self.config['filter_by_params'], 
                experiment.local_wandb_path() / ('param_filter.json'))
            
    # TODOLOW Why do I need a 'return_stitches' switch if I could just check if prediction contains this info
    def save_prediction_batch(
            self, predictions, datanames, data_folders, 
            save_to, features=None, weights=None, orig_folder_names=False, **kwargs):
        """ 
            Saving predictions on batched from the current dataset
            Saves predicted params of the datapoint to the requested data folder.
            Returns list of paths to files with prediction visualizations
            Assumes that the number of predictions matches the number of provided data names"""

        save_to = Path(save_to)
        prediction_imgs = []
        for idx, (name, folder) in enumerate(zip(datanames, data_folders)):

            # "unbatch" dictionary
            prediction = {}
            for key in predictions:
                prediction[key] = predictions[key][idx]

            # add values from GT if not present in prediction
            if (('order_matching' in self.config and self.config['order_matching'])
                    or 'origin_matching' in self.config and self.config['origin_matching']
                    or not self.gt_caching):
                print(f'{self.__class__.__name__}::Warning::Propagating '
                      'information from GT on prediction is not implemented in given context')
            else:
                gt = self.gt_cached[folder + '/' + name]
                for key in gt:
                    if key not in prediction:
                        prediction[key] = gt[key]
            try: 
                # Transform to pattern object
                pattern = self._pred_to_pattern(prediction, name)

                # log gt number of panels
                if self.gt_caching:
                    gt = self.gt_cached[folder + '/' + name]
                    pattern.spec['properties']['correct_num_panels'] = gt['num_panels']

                # save prediction
                folder_nick = self.data_folders_nicknames[folder] if not orig_folder_names else folder
                final_dir = pattern.serialize(
                    save_to / folder_nick, 
                    to_subfolder=True, 
                    tag='_predicted_', 
                    with_3d=False, with_text=False, view_ids=False
                )
            except (RuntimeError, InvalidPatternDefError, TypeError) as e:
                print('Garment3DPatternDataset::Error::{} serializing skipped: {}'.format(name, e))
                continue
            
            final_file = pattern.name + '_predicted__pattern.png'
            prediction_imgs.append(Path(final_dir) / final_file)

            # copy originals for comparison
            for file in (self.root_path / folder / name).glob('*'):
                if ('.png' in file.suffix) or ('.json' in file.suffix):
                    shutil.copy2(str(file), str(final_dir))

            # save point samples if given 
            if features is not None:
                shift = self.config['standardize']['f_shift']
                scale = self.config['standardize']['f_scale']
                point_cloud = features[idx] * scale + shift

                np.savetxt(
                    save_to / folder_nick / name / (name + '_point_cloud.txt'), 
                    point_cloud
                )
            # save per-point weights if given
            if 'att_weights' in prediction:
                np.savetxt(
                    save_to / folder_nick / name / (name + '_att_weights.txt'), 
                    prediction['att_weights'].cpu().numpy()
                )
                    
        return prediction_imgs

    def _pred_to_pattern(self, prediction, dataname):
        """Convert given predicted value to pattern object
        """

        # Outlines -- curve class determination 
        curve_class_score = prediction['outlines'][:, :, -1]
        prediction['outlines'][:, :, -1] = torch.round(torch.sigmoid(curve_class_score)).type(torch.BoolTensor)

        # undo standardization  (outside of generinc conversion function due to custom std structure)
        gt_shifts = self.config['standardize']['gt_shift']
        gt_scales = self.config['standardize']['gt_scale']
        for key in gt_shifts:
            if key == 'stitch_tags' and not self.config['explicit_stitch_tags']:  
                # ignore stitch tags update if explicit tags were not used
                continue
            prediction[key] = prediction[key].cpu().numpy() * gt_scales[key] + gt_shifts[key]

        # recover stitches
        if 'stitches' in prediction:  # if somehow prediction already has an answer
            stitches = prediction['stitches']
        elif 'edge_cls' in prediction and "edge_similarity" in prediction:
            stitches = self.prediction_to_stitches(prediction['edge_cls'], prediction['edge_similarity'], return_stitches=False)
        else:  # stitch tags to stitch list
            stitches = self.tags_to_stitches(   # FIXME Function is missing
                torch.from_numpy(prediction['stitch_tags']) if isinstance(prediction['stitch_tags'], np.ndarray) else prediction['stitch_tags'],
                prediction['free_edges_mask']
            )

        # Construct the pattern from the data
        pattern = NNSewingPattern(view_ids=False, panel_classifier=self.panel_classifier)
        pattern.name = dataname
        pattern.pattern_from_tensors(
            prediction['outlines'], 
            panel_rotations=prediction['rotations'],
            panel_translations=prediction['translations'], 
            stitches=stitches,
            padded=True)   

        return pattern


    @staticmethod
    def prediction_to_stitches(free_mask_logits, similarity_matrix, return_stitches=False):
        free_mask = (torch.sigmoid(free_mask_logits.squeeze(-1)) > 0.5).flatten()
        if not return_stitches:
            # NOTE: Apply free edge mask
            simi_matrix = similarity_matrix + similarity_matrix.transpose(0, 1)
            simi_matrix = torch.masked_fill(simi_matrix, (~free_mask).unsqueeze(0), -float("inf"))
            simi_matrix = torch.masked_fill(simi_matrix, (~free_mask).unsqueeze(-1), 0)
            num_stitches = free_mask.nonzero().shape[0] // 2
        else:
            # FIXME Why is this branch needed?? 
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
    
    def get_item_infos(self, index):
        return self.datapoints_names[index]
    
    def decode(self, output_ids: torch.Tensor, tokenizer: PreTrainedTokenizer):
        """Decode output ids to text"""
        return self.garment_tokenizer.decode(output_ids, tokenizer)
    
if __name__ == "__main__": 
    
    root_dir = "/miele/timur/garmentcodedata"
    dataset = GarmentCodeData(root_dir, start_config={'random_pairs_mode': False, 'body_type' : 'default_body',
                        'panel_classification' : '/home/timur/sewformer-garments/garment-estimator/nn/data_configs/panel_classes.json',
                        'max_pattern_len' : 37,    
                        'as_vertices' : True,                         
                        'max_panel_len' : 40,
                        'max_num_stitches' : 108,
                        'mesh_samples' : 2000, 
                        'max_datapoints_per_folder' : 10,                          
                        'stitched_edge_pairs_num': 100, 'non_stitched_edge_pairs_num': 100,
                        'shuffle_pairs': False, 'shuffle_pairs_order': False, 
                        'rot_type': 'euler', 
                        'data_folders': [
                        "garments_5000_0",
                        # "garments_5000_1", "garments_5000_2", "garments_5000_3", "garments_5000_4", "garments_5000_5", "garments_5000_6", "garments_5000_7", "garments_5000_8"
                        ]})
    dp = dataset[0]
    # gt = dp['ground_truth']
    # prediction = gt
    # pattern = NNSewingPattern(view_ids=False, panel_classifier=dataset.panel_classifier)
    # pattern.name = 'test'
    
    # pattern.pattern_from_tensors(prediction['outlines'].detach().numpy(), panel_rotations=prediction['rotations'].detach().numpy(),panel_translations=prediction['translations'].detach().numpy(), stitches=gt['stitches'].detach().numpy(), padded=True) 
    # final_dir = pattern.serialize('/home/timur/garment_foundation_model/', to_subfolder=True, tag='_predicted_', with_3d=False, with_text=False, view_ids=False)
    import code; code.interact(local=locals())
    print("done")
