
import torch, numpy as np
from typing import List, Tuple, Dict, Union, Any
from scipy.spatial.transform import Rotation as R  
from collections import defaultdict

import logging 
log = logging.getLogger(__name__)   
from transformers import PreTrainedTokenizer
from data.patterns.gcd_pattern.pattern_converter import NNSewingPattern as GCD_NNSewingPattern
from data.garment_tokenizers.utils import control_to_relative_coord, arc_from_three_points, panel_universal_transtation, is_colinear
from scipy.spatial.transform import Rotation
from data.datasets.utils import IMAGE_TOKEN_INDEX
from data.patterns.sf_pattern.pattern_converter import NNSewingPattern as SF_SewingPattern
from data.datasets.panel_configs import *
from data.garment_tokenizers.special_tokens import SpecialTokensV2, PanelEdgeTypeV3, PanelEdgeTypeIndices, DecodeErrorTypes
from .default_garment_tokenizer import GarmentTokenizer

class GarmentTokenizerForRegression(GarmentTokenizer): 
    def __init__(self, 
                 standardize: StandardizeConfig,
                 random_tag = True,
                 num_tags = 108,
                 convert_qradratic_to_cubic = False,
                 sf_only=False,
                 include_template_name=False,
                 encode_stitches_as_tags=True
                 ):
        super().__init__(
            standardize=standardize,
            bin_size=256,
            random_tag=random_tag,
            num_tags=num_tags,
            convert_qradratic_to_cubic=convert_qradratic_to_cubic,
            sf_only=sf_only,
            include_template_name=include_template_name,
            encode_stitches_as_tags=encode_stitches_as_tags
        )
        
    def get_bin_token_names(self):
        return []
    
    def get_zero_vertices(self):
        return -torch.tensor(self.gt_stats.vertices.shift) / torch.tensor(self.gt_stats.vertices.scale)
    
    def set_token_indices(self, token2idx: Dict[str, int]):
        super().set_token_indices(token2idx)
        self.panel_edge_type_indices = PanelEdgeTypeIndices(token2idx, rot_as_quat=True)
    
    def _pattern_as_list_sf(self, pattern: SF_SewingPattern, as_quat=False, endpoint_first=True):
        panel_order = pattern.panel_order(filter_nones=True)
        panel_edges, panel_rotations, panel_translations = [], [], []
        panel_names = []
        for panel_name in panel_order:
            panel_edge, rotation, translation = pattern.panel_as_list(panel_name, rot_in_euler=not as_quat, version=3, endpoint_first=endpoint_first)
            panel_edges.append(panel_edge)
            panel_rotations.append(rotation)
            panel_translations.append(translation)
            p_name = pattern.panel_classifier.class_name(pattern.panel_classifier.class_idx(pattern.template_name, panel_name)) if pattern.panel_classifier is not None else panel_name
            panel_names.append(p_name)

        return panel_edges, panel_names, panel_rotations, panel_translations, pattern.pattern['stitches']

    def encode(self, pattern: Union[GCD_NNSewingPattern, SF_SewingPattern], return_type="pt"):
        assert self.panel_edge_type_indices is not None, "Panel edge type indices not set"
        assert self.special_token_indices is not None, "Special token indices not set"
        if self.sf_only:
            assert isinstance(pattern, SF_SewingPattern), "Pattern must be an instance of SF_SewingPattern"
            
        if self.encode_stitches_as_tags:
            tag_tokens = self.get_stitch_tag_names()
            
        pattern_edges, panel_names, panel_rotations, panel_translations, stitches = self._pattern_as_list_sf(pattern, as_quat=True, endpoint_first=True) \
            if isinstance(pattern, SF_SewingPattern) else self._pattern_as_list_gcd(pattern, as_quat=True, endpoint_first=True)
        stitches = self.assign_tags_to_stitches(stitches) if self.encode_stitches_as_tags else {}
        params_output = defaultdict(list)
        endpoints = []
        transformations = []
        if self.include_template_name:
            template_name = pattern.name
            out_description =  [template_name, SpecialTokensV2.PATTERN_START.value]
        else:
            out_description = [SpecialTokensV2.PATTERN_START.value]
        for panel_edges, panel_name, panel_tran, panel_rot in zip(pattern_edges, panel_names, panel_translations, panel_rotations):
            out_description += [SpecialTokensV2.PANEL_START.value, panel_name]
            transl_params = (panel_tran.flatten() - self.gt_stats.translations.shift) / self.gt_stats.translations.scale
            rot_params = (panel_rot.flatten() - self.gt_stats.rotations.shift) / self.gt_stats.rotations.scale
            trans_params = np.concatenate([transl_params.flatten(), rot_params.flatten()])
            transformations.append(trans_params)
            params_output[self.panel_edge_type_indices.get_token_indices(PanelEdgeTypeV3.MOVE)].append(trans_params)
            out_description += [PanelEdgeTypeV3.MOVE.value]
            for edge_id, panel_edge in enumerate(panel_edges):
                edge_type: PanelEdgeTypeV3 = panel_edge[0]
                edge_params = (panel_edge[1].reshape(-1, 2) - self.gt_stats.vertices.shift)/ self.gt_stats.vertices.scale
                if edge_type.is_closure():
                    endpoints.append(- np.array(self.gt_stats.vertices.shift) / np.array(self.gt_stats.vertices.scale))
                else:
                    endpoints.append(edge_params[0])
                out_description += [edge_type.value]
                param_num = edge_type.get_num_params()
                if param_num > 0:
                    token_indices = self.panel_edge_type_indices.get_token_indices(edge_type)
                    params = edge_params.flatten()[:param_num]
                    params_output[token_indices].append(params)
                if self.encode_stitches_as_tags:
                    # last entry is null
                    tag = stitches.get((panel_name, edge_id), self.num_tags)
                    out_description += [tag_tokens[tag]]
            out_description += [SpecialTokensV2.PANEL_END.value]
        out_description += [SpecialTokensV2.PATTERN_END.value]
        params_output = {k: np.stack(v) for k, v in params_output.items()}
        if return_type == "pt":
            params_output = {k: torch.from_numpy(v).float() for k, v in params_output.items()}
            endpoints = torch.from_numpy(np.stack(endpoints)).float()
            transformations = torch.from_numpy(np.stack(transformations)).float()
        out = {
            "description": out_description,
            "params": params_output,
            "endpoints": endpoints,
            "transformations": transformations
        }
        return out
            

    def decode(self, output_dict: Dict[str, Any], tokenizer: PreTrainedTokenizer): 
        """Decode output ids to text"""
        output_ids = output_dict['output_ids'][0]
        output_param_dict = {k:v for k, v in output_dict['params'].items()}
        input_mask = output_dict['input_mask'][0]
        
            
        text_output = tokenizer.decode(output_ids[output_ids != IMAGE_TOKEN_INDEX], skip_special_tokens=True)
        output_ids = output_ids.cpu().numpy().copy()
        garment_ends = np.where(output_ids == self.special_token_indices.get_token_indices(SpecialTokensV2.PATTERN_END))[0]
        garment_starts = np.where(output_ids == self.special_token_indices.get_token_indices(SpecialTokensV2.PATTERN_START))[0]
        garment_ends = np.where(
            np.logical_and(output_ids == self.special_token_indices.get_token_indices(SpecialTokensV2.PATTERN_END),
            input_mask))[0]
        garment_starts = np.where(
            np.logical_and(output_ids == self.special_token_indices.get_token_indices(SpecialTokensV2.PATTERN_START),
                           input_mask))[0]
        pattern = GCD_NNSewingPattern()
        if len(garment_starts) != len(garment_ends) or \
            len(garment_starts) == 0 or \
            len(garment_ends) == 0 or \
            np.any(garment_starts > garment_ends):
            log.error("Garment Decoding Error: Unmatched or Invalid number of garment starts and ends")    
            return text_output, pattern, DecodeErrorTypes.UNMATCHED_PATTERN_TOKENS
        garment_start, garment_end = garment_starts[0], garment_ends[0]
        pattern_tokens = output_ids[garment_start+1:garment_end]
        
        for k, v in output_param_dict.items():
            param_mask = pattern_tokens == k
            if v.shape[0] != param_mask.sum():
                log.error(f"Param {k} has wrong shape {v.shape} vs {param_mask.sum()}. Output is {text_output}")
                output_param_dict[k] = v[:param_mask.sum()]
            
        pattern_dict, error_type = self.decode_pattern(pattern_tokens, output_param_dict, tokenizer)
        pattern.pattern_from_pattern_dict(pattern_dict)
        return text_output, pattern, error_type
        
    def decode_pattern(self, token_sequence: np.ndarray, param_dict: Dict[int, np.ndarray], tokenizer: PreTrainedTokenizer): 
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
            commands = np.isin(panel, self.panel_edge_type_indices.get_all_indices()).nonzero()[0]
            if len(commands) == 0:
                log.error(f"Panel {i} has no edges. Skipping.")
                continue
            try: 
                panel_name = tokenizer.decode(panel[:commands[0]].tolist())
            except:
                panel_name = f'Panel_{i}'
                
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
                    edge_dict['curvature'] = {
                        "type": 'quadratic',
                        "params": [ctrl_pt]
                    }
                elif edge_type == PanelEdgeTypeV3.CLOSURE_CURVE:
                    params = params * np.array(self.gt_stats.vertices.scale) + np.array(self.gt_stats.vertices.shift)
                    ctrl_pt = control_to_relative_coord(last_point, np.array([0, 0]), params)
                    edge_dict['curvature'] = {
                        "type": 'quadratic',
                        "params": [ctrl_pt]
                    }
                elif edge_type == PanelEdgeTypeV3.CUBIC:
                    params = params.reshape(3, 2) * np.array(self.gt_stats.vertices.scale) + np.array(self.gt_stats.vertices.shift)
                    endpoint, abs_ctrl_pt1, abs_ctrl_pt2 = params[0], params[1], params[2]
                    ctrl_pt1 = control_to_relative_coord(last_point, endpoint, abs_ctrl_pt1)
                    ctrl_pt2 = control_to_relative_coord(last_point, endpoint, abs_ctrl_pt2)
                    edge_dict['curvature'] = {
                        "type": 'cubic',
                        "params": [ctrl_pt1, ctrl_pt2]
                    }
                elif edge_type == PanelEdgeTypeV3.CLOSURE_CUBIC:
                    params = params.reshape(2, 2) * np.array(self.gt_stats.vertices.scale) + np.array(self.gt_stats.vertices.shift)
                    ctrl_pt1 = control_to_relative_coord(last_point, np.array([0, 0]), params[0])
                    ctrl_pt2 = control_to_relative_coord(last_point, np.array([0, 0]), params[1])
                    edge_dict['curvature'] = {
                        "type": 'cubic',
                        "params": [ctrl_pt1, ctrl_pt2]
                    }
                elif edge_type == PanelEdgeTypeV3.ARC:
                    params = params.reshape(2, 2) * np.array(self.gt_stats.vertices.scale) + np.array(self.gt_stats.vertices.shift)
                    endpoint, abs_ctrl_pt = params[0], params[1]
                    if is_colinear(last_point, endpoint, abs_ctrl_pt):
                        # arc became colinear due to rounding errors. Drawing a line instead.
                        edge_type = PanelEdgeTypeV3.LINE
                    else:
                        _, _, rad, large_arc, right = arc_from_three_points(last_point, endpoint, abs_ctrl_pt)
                        edge_dict['curvature'] = {
                            "type": 'circle',
                            "params": [rad, int(large_arc), int(right)]
                        }
                elif edge_type == PanelEdgeTypeV3.CLOSURE_ARC:
                    params = params * np.array(self.gt_stats.vertices.scale) + np.array(self.gt_stats.vertices.shift)
                    if is_colinear(last_point, np.array([0, 0]), params):
                        # arc became colinear due to rounding errors. Drawing a line instead.
                        edge_type = PanelEdgeTypeV3.CLOSURE_LINE
                    else:
                        _, _, rad, large_arc, right = arc_from_three_points(last_point, np.array([0, 0]), params)
                        edge_dict['curvature'] = {
                            "type": 'circle',
                            "params": [rad, int(large_arc), int(right)]
                        }
                
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
