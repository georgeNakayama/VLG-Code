
import torch, numpy as np, os, sys
from pathlib import Path 
from enum import Enum
from typing import List, Tuple, Dict, Union, Any
from scipy.spatial.transform import Rotation as R  
from collections import defaultdict

import logging 
log = logging.getLogger(__name__)   
from transformers import PreTrainedTokenizer
from data.datasets.garmentcodedata.garmentcode_dataset import GarmentCodeData
from data.datasets.garmentcodedata.pattern_converter import NNSewingPattern as GCD_NNSewingPattern
from data.garment_tokenizers.utils import arc_rad_flags_to_three_point, control_to_abs_coord, discretize, control_to_relative_coord, arc_from_three_points, panel_universal_transtation, is_colinear
from scipy.spatial.transform import Rotation
from data.datasets.utils import IMAGE_TOKEN_INDEX
from data.pattern_converter import NNSewingPattern as SF_SewingPattern
from data.panel_classes import PanelClasses
from data.datasets.panel_configs import *
from data.garment_tokenizers.special_tokens import SpecialTokensV2, SpecialTokensIndices, PanelEdgeTypeV3, PanelEdgeTypeIndices, DecodeErrorTypes
from .garment_tokenizer_for_regression import GarmentTokenizerForRegression

class DressCodeGarmentTokenizerForRegression(GarmentTokenizerForRegression): 
    def __init__(self, 
                 standardize: StandardizeConfig,
                 random_tag = True,
                 num_tags = 108,
                 include_template_name=True,
                 encode_stitches_as_tags=True
                 ):
        super().__init__(
            standardize=standardize,
            random_tag=random_tag,
            num_tags=num_tags,
            convert_qradratic_to_cubic=False,
            sf_only=True,
            include_template_name=include_template_name,
            encode_stitches_as_tags=encode_stitches_as_tags
        )
        self.include_template_name = False # No Token names for now
        self.panel_names: List[str] = None
        self.pattern_names: List[str] = None
        self.panel_names2idx: Optional[Dict[str, int]] = None
        self.idx2panel_names: Optional[Dict[int, str]] = None
        self.pattern_names2idx: Optional[Dict[str, int]] = None
        self.idx2pattern_names: Optional[Dict[int, str]] = None
        
    def set_panel_names(self, panel_names: List[str]):
        self.panel_names = panel_names
        
    def set_pattern_names(self, pattern_names: List[str]):
        self.pattern_names = pattern_names
        
    def get_panel_names(self):
        return self.panel_names
    
    def get_pattern_names(self):
        return self.pattern_names
    
    def get_all_token_names(self):
        token_names = super().get_all_token_names()
        return token_names + self.panel_names + self.pattern_names
    
    def set_token_indices(self, token2idx: Dict[str, int]):
        super().set_token_indices(token2idx)
        self.panel_edge_type_indices = PanelEdgeTypeIndices(token2idx, rot_as_quat=True)
        assert self.panel_names is not None, "Panel names not set"
        self.panel_names2idx = {}
        self.idx2panel_names = {}
        for panel_name in self.panel_names:
            self.panel_names2idx[panel_name] = token2idx[panel_name]
            self.idx2panel_names[token2idx[panel_name]] = panel_name
        self.pattern_names2idx = {}
        self.idx2pattern_names = {}
        for pattern_name in self.pattern_names:
            self.pattern_names2idx[pattern_name] = token2idx[pattern_name]
            self.idx2pattern_names[token2idx[pattern_name]] = pattern_name
    
    def assign_tags_to_stitches(self, stitches, pattern):
        new_stitch_dict = {}
        if self.random_tag:
            tag_ids = np.random.permutation(self.num_tags)
        else:
            tag_ids = np.arange(self.num_tags)
        assert len(stitches) <= self.num_tags
        for stitch_id, stitch_tuple in enumerate(stitches):
            stitch_tuple = [s for s in stitch_tuple if isinstance(s, dict)]
            assert len(stitch_tuple) == 2
            for stitch in stitch_tuple:
                class_idx = pattern.panel_classifier.class_idx(pattern.template_name, stitch['panel'])
                panel_name = pattern.panel_classifier.class_name(class_idx)
                new_stitch_dict[(panel_name, stitch['edge'])] = tag_ids[stitch_id]
        return new_stitch_dict
    def encode(self, pattern: Union[GCD_NNSewingPattern, SF_SewingPattern], return_type="pt"):
        assert self.panel_edge_type_indices is not None, "Panel edge type indices not set"
        assert self.special_token_indices is not None, "Special token indices not set"
        if self.sf_only:
            assert isinstance(pattern, SF_SewingPattern), "Pattern must be an instance of SF_SewingPattern"
            
        if self.encode_stitches_as_tags:
            tag_tokens = self.get_stitch_tag_names()
            
        pattern_edges, panel_names, panel_rotations, panel_translations, stitches = self._pattern_as_list_sf(pattern, as_quat=True, endpoint_first=True)
        stitches = self.assign_tags_to_stitches(stitches, pattern) if self.encode_stitches_as_tags else {}
        params_output = defaultdict(list)
        endpoints = []
        transformations = []
        out_description = [self.special_token_indices.get_token_indices(SpecialTokensV2.PATTERN_START), self.pattern_names2idx[pattern.template_name]]
        for panel_edges, panel_name, panel_tran, panel_rot in zip(pattern_edges, panel_names, panel_translations, panel_rotations):
            out_description += [
                self.special_token_indices.get_token_indices(SpecialTokensV2.PANEL_START), 
                self.panel_names2idx[panel_name]
                ]
            transl_params = (panel_tran.flatten() - self.gt_stats.translations.shift) / self.gt_stats.translations.scale
            rot_params = (panel_rot.flatten() - self.gt_stats.rotations.shift) / self.gt_stats.rotations.scale
            trans_params = np.concatenate([transl_params.flatten(), rot_params.flatten()])
            transformations.append(trans_params)
            params_output[self.panel_edge_type_indices.get_token_indices(PanelEdgeTypeV3.MOVE)].append(trans_params)
            out_description += [self.panel_edge_type_indices.get_token_indices(PanelEdgeTypeV3.MOVE)]
            for edge_id, panel_edge in enumerate(panel_edges):
                edge_type: PanelEdgeTypeV3 = panel_edge[0]
                edge_params = (panel_edge[1].reshape(-1, 2) - self.gt_stats.vertices.shift)/ self.gt_stats.vertices.scale
                token_indices = self.panel_edge_type_indices.get_token_indices(edge_type)
                if edge_type.is_closure():
                    endpoints.append(- np.array(self.gt_stats.vertices.shift) / np.array(self.gt_stats.vertices.scale))
                else:
                    endpoints.append(edge_params[0])
                out_description += [token_indices]
                param_num = edge_type.get_num_params()
                if param_num > 0:
                    params = edge_params.flatten()[:param_num]
                    params_output[token_indices].append(params)
                if self.encode_stitches_as_tags:
                    # last entry is null
                    tag = stitches.get((panel_name, edge_id), self.num_tags)
                    out_description += [self.tag_name2tag_idx[tag_tokens[tag]]]
            out_description += [self.special_token_indices.get_token_indices(SpecialTokensV2.PANEL_END)]
        out_description += [self.special_token_indices.get_token_indices(SpecialTokensV2.PATTERN_END)]
        params_output = {k: np.stack(v) for k, v in params_output.items()}
        if return_type == "pt":
            params_output = {k: torch.from_numpy(v).float() for k, v in params_output.items()}
            endpoints = torch.from_numpy(np.stack(endpoints)).float()
            transformations = torch.from_numpy(np.stack(transformations)).float()
            out_description = torch.tensor(out_description).long()
        out = {
            "description": out_description,
            "params": params_output,
            "endpoints": endpoints,
            "transformations": transformations
        }
        return out
            

    def decode(self, output_dict: Dict[str, Any], panel_classifier: PanelClasses): 
        """Decode output ids to text"""
        output_ids = output_dict['output_ids'][0]
        output_param_dict = output_dict['params']
        if isinstance(output_ids, torch.Tensor):
            output_ids = output_ids.cpu().numpy().copy()
        garment_ends = np.where(output_ids == self.special_token_indices.get_token_indices(SpecialTokensV2.PATTERN_END))[0]
        garment_starts = np.where(output_ids == self.special_token_indices.get_token_indices(SpecialTokensV2.PATTERN_START))[0]
        garment_ends = np.where(output_ids == self.special_token_indices.get_token_indices(SpecialTokensV2.PATTERN_END))[0]
        garment_starts = np.where(output_ids == self.special_token_indices.get_token_indices(SpecialTokensV2.PATTERN_START))[0]
        pattern = SF_SewingPattern()
        if len(garment_starts) != len(garment_ends) or \
            len(garment_starts) == 0 or \
            len(garment_ends) == 0 or \
            np.any(garment_starts > garment_ends):
            log.error("Garment Decoding Error: Unmatched or Invalid number of garment starts and ends")    
            return output_ids, pattern, DecodeErrorTypes.UNMATCHED_PATTERN_TOKENS
        garment_start, garment_end = garment_starts[0], garment_ends[0]
        pattern_tokens = output_ids[garment_start+1:garment_end]
        template_name = self.idx2pattern_names.get(pattern_tokens[0], None)
        pattern.template_name = template_name
        pattern_tokens = pattern_tokens[1:]
        
        for k, v in output_param_dict.items():
            param_mask = pattern_tokens == k
            if v.shape[0] != param_mask.sum():
                log.error(f"Param {k} has wrong shape {v.shape} vs {param_mask.sum()}. Output is {output_ids}")
                output_param_dict[k] = v[:param_mask.sum()]
            
        pattern_dict, error_type = self.decode_pattern(pattern_tokens, output_param_dict, template_name, panel_classifier)
        pattern.pattern_from_pattern_dict(pattern_dict)
        return output_ids, pattern, error_type
        
    def decode_pattern(self, token_sequence: np.ndarray, param_dict: Dict[int, np.ndarray], template_name: Optional[str], panel_classifier: PanelClasses): 
        assert self.bin_idx2bin_number is not None \
            and self.bin_idx2bin_name is not None  \
            and self.bin_name2bin_idx is not None  \
            and self.special_token_indices is not None \
            and self.panel_edge_type_indices is not None, "token indices needs to be set before decoding"
        # find the panel starts and ends
        panel_starts = np.where(token_sequence == self.special_token_indices.get_token_indices(SpecialTokensV2.PANEL_START))[0]
        panel_ends = np.where(token_sequence == self.special_token_indices.get_token_indices(SpecialTokensV2.PANEL_END))[0]
        if len(panel_starts) != len(panel_ends) or \
            len(panel_starts) == 0 or \
            len(panel_ends) == 0 or \
            np.any(panel_starts > panel_ends):
                log.error("Garment Decoding Error: Unmatched or Invalid number of panel starts and ends")
                return {'panels': {},'stitches': []}, DecodeErrorTypes.UNMATCHED_PANEL_TOKENS
        
        n_panels = len(panel_starts)
        pattern = {'panels': {},'stitches': [], 'panel_order':[]}
        edge_stitches = {}
        for i in range(n_panels):
            panel_dict = {
                'rotation': [0, 0, 0],
                'translation': [0, 0, 0],
                'vertices': [[0, 0]],
                'edges': []
            }
            panel_start = panel_starts[i]
            panel_end = panel_ends[i]
            panel = token_sequence[panel_start+1:panel_end]
            panel_idx = panel[0]
            panel = panel[1:]
            commands = np.isin(panel, self.panel_edge_type_indices.get_all_indices()).nonzero()[0]
            if len(commands) == 0:
                log.error(f"Panel {i} has no edges. Skipping.")
                continue
            panel_name = self.idx2panel_names.get(panel_idx,'panel_{i}')
            # panel_class = panel_classifier.classes.get(panel_name, None)
            # if panel_class is not None:
            #     panel_names = [n[1] for n in panel_class if n[0] == template_name]
            #     if len(panel_names) == 1:
            #         panel_name = panel_names[0]
            pattern['panel_order'].append(panel_name)
            num_edges = 0
            for j in range(len(commands)):
                command = commands[j]
                command_end = commands[j+1] if j+1 < len(commands) else len(panel)
                edge_type = self.panel_edge_type_indices.get_index_token(panel[command].item())
                param_array = param_dict.get(panel[command].item(), None)
                if param_array is not None:
                    params = param_array[0]
                    param_dict[panel[command].item()] = param_array[1:]
                else:
                    # e.g., closure line
                    params = None
                edge_params = panel[command+1:command_end]
                
                    
                if edge_type == PanelEdgeTypeV3.MOVE:
                    transl_params, rot_params = params[:3], params[3:]
                    transl_params = transl_params * np.array(self.gt_stats.translations.scale) + np.array(self.gt_stats.translations.shift)
                    rot_params = rot_params * np.array(self.gt_stats.rotations.scale) + np.array(self.gt_stats.rotations.shift)
                    rot_euler = R.from_quat(rot_params).as_euler('xyz', degrees=True)
                    panel_dict['rotation'] = rot_euler.tolist()
                    panel_dict['translation'] = transl_params.tolist()
                    continue
                
                if self.encode_stitches_as_tags:
                    if len(edge_params) == 0:
                        stitch_tag = -1 * np.ones(1, dtype=int)
                    else:
                        stitch_tag = edge_params[0]
                    stitch_tag = self.get_tag_from_index(stitch_tag)
                    if stitch_tag != -1:
                        if stitch_tag not in edge_stitches:
                            edge_stitches[stitch_tag] = []
                        edge_stitches[stitch_tag].append({
                            'panel': panel_name,
                            'edge': num_edges
                        })
                edge_dict = {}
                last_point = np.array(panel_dict['vertices'][-1])
                if edge_type == PanelEdgeTypeV3.CLOSURE_LINE:
                    # Start point is always 0
                    pass
                elif edge_type == PanelEdgeTypeV3.LINE:
                    endpoint = params * np.array(self.gt_stats.vertices.scale) + np.array(self.gt_stats.vertices.shift)
                elif edge_type == PanelEdgeTypeV3.CURVE:
                    params = params.reshape(2, 2) * np.array(self.gt_stats.vertices.scale) + np.array(self.gt_stats.vertices.shift)
                    endpoint, abs_ctrl_pt  = params[0], params[1]
                    ctrl_pt = control_to_relative_coord(last_point, endpoint, abs_ctrl_pt)
                    edge_dict['curvature'] = ctrl_pt
                elif edge_type == PanelEdgeTypeV3.CLOSURE_CURVE:
                    params = params * np.array(self.gt_stats.vertices.scale) + np.array(self.gt_stats.vertices.shift)
                    ctrl_pt = control_to_relative_coord(last_point, np.array([0, 0]), params)
                    edge_dict['curvature'] = ctrl_pt
                
                edge_dict['endpoints'] = [num_edges, num_edges+1] if not edge_type.is_closure() else [num_edges, 0]
                panel_dict['edges'].append(edge_dict)
                if edge_type.is_closure():
                    break
                panel_dict['vertices'].append(endpoint.tolist())
                num_edges += 1
                    
            transl_origin = panel_universal_transtation(np.array(panel_dict['vertices']), panel_dict['rotation'], [0, 0, 0])[1]
            shift = np.append(transl_origin, 0)  # to 3D
            panel_rotation = Rotation.from_euler('xyz', panel_dict['rotation'], degrees=True)
            comenpensating_shift = - panel_rotation.as_matrix().dot(shift)
            panel_dict['translation'] = np.array(panel_dict['translation']) + comenpensating_shift
            panel_dict['translation'] = panel_dict['translation'].tolist()
            
            pattern['panels'][panel_name] = panel_dict

        pattern['stitches'] = [stitches for stitches in edge_stitches.values() if len(stitches) == 2]
        return pattern, DecodeErrorTypes.NO_ERROR

if __name__ == "__main__":
    root_dir = "/miele/timur/garmentcodedata"
    dataset = GarmentCodeData(root_dir, start_config={'random_pairs_mode': False, 'body_type' : 'default_body',
                        'panel_classification' : '/home/timur/sewformer-garments/garment-estimator/nn/data_configs/panel_classes.json',
                        'max_pattern_len' : 37,                            
                        'max_panel_len' : 40,
                        'max_num_stitches' : 108,
                        'mesh_samples' : 2000, 
                        'max_datapoints_per_folder' : 10,                          
                        'stitched_edge_pairs_num': 100, 'non_stitched_edge_pairs_num': 100,
                        'shuffle_pairs': False, 'shuffle_pairs_order': False, 
                        'data_folders': [
                        "garments_5000_0",
                        # "garments_5000_1", "garments_5000_2", "garments_5000_3", "garments_5000_4", "garments_5000_5", "garments_5000_6", "garments_5000_7", "garments_5000_8"
                        ]})
    
    conf = StandardizeConfig(
        outlines=StatsConfig(shift=[-51.205456, -49.627438, 0.0, -0.35, 0.0, -0.35, 0.0], scale=[104.316666, 99.264435, 0.5, 0.7, 0.8, 0.7, 1.0]),
        rotations=StatsConfig(shift=[0, 0], scale=[0.5, 0.5]),
        stitch_tags=StatsConfig(shift=[0, 0], scale=[0., 0.5]),
        translations=StatsConfig(shift=[0, 0], scale=[0.5, 0.5]),
        vertices=StatsConfig(shift=[-30, -70], scale=[190, 180])
    )

    garment_code = DefaultGarmentCode(standardize=conf)
    tokens = garment_code.encode(dataset, 0)

    print(tokens)

    tensor, names, description = garment_code.decode(np.array(tokens))

    import code; code.interact(local=locals())