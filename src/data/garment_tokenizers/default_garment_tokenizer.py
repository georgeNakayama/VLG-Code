
import torch, numpy as np, os, sys
from pathlib import Path 
from enum import Enum
from typing import List, Tuple, Dict, Union, Any

import logging 
log = logging.getLogger(__name__)   
from transformers import PreTrainedTokenizer
from data.patterns.gcd_pattern.pattern_converter import NNSewingPattern as GCD_NNSewingPattern
from .utils import arc_rad_flags_to_three_point, control_to_abs_coord, discretize, control_to_relative_coord, arc_from_three_points, panel_universal_transtation, is_colinear
from scipy.spatial.transform import Rotation
from data.datasets.utils import IMAGE_TOKEN_INDEX
from data.patterns.sf_pattern.pattern_converter import EmptyPanelError, NNSewingPattern as SF_SewingPattern
from data.datasets.panel_configs import *
from .special_tokens import SpecialTokensV2, SpecialTokensIndices, PanelEdgeTypeV3, PanelEdgeTypeIndices, DecodeErrorTypes


class GarmentTokenizer: 
    def __init__(self, 
                 standardize: StandardizeConfig,
                 bin_size = 256,
                 random_tag = True,
                 num_tags = 108,
                 convert_qradratic_to_cubic = False,
                 sf_only=False,
                 include_template_name=True,
                 encode_stitches_as_tags=True
                 ):
        self.bin_size = bin_size
        self.gt_stats = standardize
        self.num_tags = num_tags
        self.random_tag = random_tag
        self.convert_qradratic_to_cubic = convert_qradratic_to_cubic
        self.encode_stitches_as_tags = encode_stitches_as_tags
        self.sf_only = sf_only 
        self.include_template_name = include_template_name
        if self.include_template_name:
            assert self.sf_only, "include_template_name can only be used with sewfactory dataset"
            
        self.panel_edge_type_indices: Optional[PanelEdgeTypeIndices] = None
        self.special_token_indices: Optional[SpecialTokensIndices] = None
        self.bin_name2bin_number = {k: i for i , k in enumerate(self.get_bin_token_names())}
        self.bin_number2bin_name = {v: k for k, v in self.bin_name2bin_number.items()}
        self.bin_idx2bin_number = None
        self.bin_name2bin_idx = None
        self.bin_idx2bin_name = None
    
    def get_all_token_names(self):
        all_names = self.get_bin_token_names()
        if not self.sf_only:
            all_names += PanelEdgeTypeV3.list()
        else:
            all_names += PanelEdgeTypeV3.get_sewfactory_token()
        all_names += SpecialTokensV2.list()
        if self.encode_stitches_as_tags:
            all_names += self.get_stitch_tag_names()
        return all_names
    
    def get_bin_token_names(self):
        return [f"<pattern_bin_{i}>" for i in range(self.bin_size)]
    
    def get_stitch_tag_names(self):
        return [f"<stitch_tag_{i}>" for i in range(self.num_tags)] + ["<stitch_tag_null>"]

    def get_bins_from_indices(self, params: np.ndarray):
        shapes = params.shape
        converted = []
        for ind in params.flatten():
            converted.append(self.bin_idx2bin_number.get(ind.item(), 0))
        return np.array(converted).astype(float).reshape(shapes)
    
    def get_tag_from_index(self, ind: np.ndarray):
        return self.tag_idx2tag_number.get(ind.item(), -1)

    def set_token_indices(self, token2idx: Dict[str, int]):
        self.special_token_indices = SpecialTokensIndices(token2idx)
        self.panel_edge_type_indices = PanelEdgeTypeIndices(token2idx, rot_as_quat=False)
        
        self.bin_idx2bin_number = {}
        self.bin_name2bin_idx = {}
        self.bin_idx2bin_name = {}
        
        self.tag_idx2tag_number = {}
        self.tag_name2tag_idx = {}
        self.tag_idx2tag_name = {}
            
        for i, k in enumerate(self.get_bin_token_names()):
            self.bin_name2bin_idx[k] = token2idx[k]
            self.bin_idx2bin_name[token2idx[k]] = k
            self.bin_idx2bin_number[token2idx[k]] = i
            
        if self.encode_stitches_as_tags:
            for i, k in enumerate(self.get_stitch_tag_names()):
                self.tag_name2tag_idx[k] = token2idx[k]
                self.tag_idx2tag_name[token2idx[k]] = k
                if k == "<stitch_tag_null>":
                    self.tag_idx2tag_number[token2idx[k]] = -1
                else:
                    self.tag_idx2tag_number[token2idx[k]] = i
    
    def assign_tags_to_stitches(self, stitches):
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
                new_stitch_dict[(stitch['panel'], stitch['edge'])] = tag_ids[stitch_id]
        return new_stitch_dict
    
    def encode_list(self, pattern_list: List[Union[GCD_NNSewingPattern, SF_SewingPattern]]):
        encoded_list = []
        for pattern in pattern_list:
            encoded_list.append(self.encode(pattern))
        return encoded_list
    
    def encode(self, pattern: Union[GCD_NNSewingPattern, SF_SewingPattern]):
        if self.sf_only:
            assert isinstance(pattern, SF_SewingPattern), "Pattern must be an instance of SF_SewingPattern"
            
        bin_tokens = self.get_bin_token_names() 
        if self.encode_stitches_as_tags:
            tag_tokens = self.get_stitch_tag_names()
            
        pattern_edges, panel_names, panel_rotations, panel_translations, stitches = self._pattern_as_list_sf(pattern) if isinstance(pattern, SF_SewingPattern) else self._pattern_as_list_gcd(pattern)
        stitches = self.assign_tags_to_stitches(stitches) if self.encode_stitches_as_tags else {}
        if self.include_template_name:
            template_name = pattern.name
            out_description =  [template_name, SpecialTokensV2.PATTERN_START.value]
        else:
            out_description = [SpecialTokensV2.PATTERN_START.value]
        for panel_edges, panel_name, panel_tran, panel_rot in zip(pattern_edges, panel_names, panel_translations, panel_rotations):
            out_description += [SpecialTokensV2.PANEL_START.value, panel_name]
            trans_params = discretize(panel_tran.reshape(-1, 3), self.bin_size, self.gt_stats.translations.shift, self.gt_stats.translations.scale)
            rot_params = discretize(panel_rot.reshape(-1, 3), self.bin_size, self.gt_stats.rotations.shift, self.gt_stats.rotations.scale)
            out_description += [PanelEdgeTypeV3.MOVE.value] + [bin_tokens[p] for p in trans_params.flatten().tolist()] + [bin_tokens[p] for p in rot_params.flatten().tolist()]
            for edge_id, panel_edge in enumerate(panel_edges):
                edge_type = panel_edge[0]
                edge_params = discretize(panel_edge[1].reshape(-1, 2), self.bin_size, self.gt_stats.vertices.shift, self.gt_stats.vertices.scale)
                out_description += [edge_type.value]
                param_num = edge_type.get_num_params()
                if param_num > 0:
                    out_description += [bin_tokens[p] for p in edge_params.flatten()[:param_num].tolist()]
                if self.encode_stitches_as_tags:
                    # last entry is null
                    tag = stitches.get((panel_name, edge_id), self.num_tags)
                    out_description += [tag_tokens[tag]]
            out_description += [SpecialTokensV2.PANEL_END.value]
        out_description += [SpecialTokensV2.PATTERN_END.value]
        return {"description": [out_description]}
    
    def _pattern_as_list_gcd(self, pattern: GCD_NNSewingPattern, as_quat=False, endpoint_first=False):
        panel_order = pattern.panel_order(filter_nones=True)
        all_panel_edges, panel_rotations, panel_translations = [], [], []
        for panel_name in panel_order:
            panel_edges = []
            panel = pattern.pattern['panels'][panel_name]
            vertices = np.array(panel['vertices'])
            start_point = vertices[panel['edges'][0]["endpoints"]][0].copy()
            for i, edge in enumerate(panel['edges']):
                endpoints = vertices[edge["endpoints"]]
                edge_type = PanelEdgeTypeV3.LINE
                params = endpoints[1]
                if 'curvature' in edge:
                    # in case of an arc
                    if edge['curvature']['type'] == 'circle':
                        edge_type = PanelEdgeTypeV3.ARC
                        params = edge['curvature']['params']
                        _, _, coords = arc_rad_flags_to_three_point(endpoints[0], endpoints[1], params[0], params[1], params[2], False)
                        params = np.concatenate([coords, endpoints[1]]) if not endpoint_first else np.concatenate([endpoints[1], coords])

                    # in case of a curve
                    elif edge['curvature']['type'] == 'cubic':
                        edge_type = PanelEdgeTypeV3.CUBIC
                        params = np.concatenate([control_to_abs_coord(endpoints[0], endpoints[1], edge['curvature']['params'][0]), 
                                                  control_to_abs_coord(endpoints[0], endpoints[1], edge['curvature']['params'][1]), 
                                                  endpoints[1]]) if not endpoint_first else np.concatenate([endpoints[1], 
                                                  control_to_abs_coord(endpoints[0], endpoints[1], edge['curvature']['params'][0]), 
                                                  control_to_abs_coord(endpoints[0], endpoints[1], edge['curvature']['params'][1])])
                        
                    elif edge['curvature']['type'] == 'quadratic':
                        # Convert to cubic for uniformity
                        # https://stackoverflow.com/questions/3162645/convert-a-quadratic-bezier-to-a-cubic-one
                        # NOTE: Assuming relative coor
                        if self.convert_qradratic_to_cubic:
                            cp = np.array(edge['curvature']['params'][0])
                            start, end = np.array([0, 0]), np.array([1, 0])

                            cubic_1 = start + 2. / 3. * (cp - start)
                            cubic_2 = end + 2. / 3. * (cp - end)
                            edge_type = PanelEdgeTypeV3.CUBIC
                            params = np.concatenate([control_to_abs_coord(endpoints[0], endpoints[1], cubic_1), 
                                                    control_to_abs_coord(endpoints[0], endpoints[1], cubic_2), 
                                                    endpoints[1]]) if not endpoint_first else np.concatenate([endpoints[1], 
                                                    control_to_abs_coord(endpoints[0], endpoints[1], cubic_1), 
                                                    control_to_abs_coord(endpoints[0], endpoints[1], cubic_2)])
                        else:
                            edge_type = PanelEdgeTypeV3.CURVE
                            params = np.concatenate([control_to_abs_coord(endpoints[0], endpoints[1], edge['curvature']['params'][0]), 
                                                    endpoints[1]]) if not endpoint_first else np.concatenate([endpoints[1], 
                                                    control_to_abs_coord(endpoints[0], endpoints[1], edge['curvature']['params'][0])])
                
                params = params.reshape(-1, 2) - start_point
                params = params.flatten()
                if i == len(panel['edges']) - 1:
                    edge_type = edge_type.get_closure()
                    # params = params[:-2]
                    params = params[:-2] if not endpoint_first else params[2:] # discard the last point
                panel_edges.append((edge_type, params))
            
            # ----- 3D placement convertion  ------
            # Global Translation (more-or-less stable across designs)
            translation, _ = panel_universal_transtation(vertices, panel['rotation'], panel['translation'])
            panel_translations.append(translation)
            panel_rotations.append(np.array(panel['rotation']) if not as_quat else Rotation.from_euler('xyz', panel['rotation'], degrees=True).as_quat())
            all_panel_edges.append(panel_edges)
        
        return all_panel_edges, panel_order, panel_rotations, panel_translations, pattern.pattern['stitches']
    
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
    
    def evaluate_patterns(self, pred_patterns: List[GCD_NNSewingPattern], gt_patterns: List[GCD_NNSewingPattern]):
        assert len(pred_patterns) == len(gt_patterns)
        total_num_panel_correct = torch.zeros(len(pred_patterns)).cuda()
        total_num_edge_acc = torch.ones(len(pred_patterns)).cuda() * -1
        total_num_edge_in_correct_acc = torch.ones(len(pred_patterns)).cuda() * -1
        total_residual_error = torch.ones(len(pred_patterns)).cuda() * -1
        total_translation_l2 = torch.ones(len(pred_patterns)).cuda() * -1
        total_rotation_l2 = torch.ones(len(pred_patterns)).cuda() * -1
        total_stitch_acc = torch.ones(len(pred_patterns)).cuda() * -1
        for k, (pred_pattern, gt_pattern) in enumerate(zip(pred_patterns, gt_patterns)):
            if len(pred_pattern.pattern["panels"]) == 0:
                continue
            pred_panel_names = pred_pattern.panel_order(filter_nones=True)
            gt_panel_names = gt_pattern.panel_order(filter_nones=True)
            correct = len(pred_panel_names) == len(gt_panel_names)
            total_num_panel_correct[k] = float(correct)
            correct_num_edges = 0
            for i in range(len(pred_panel_names)):
                try:
                    edges, rot, transl, aug_edges = pred_pattern.panel_as_numeric(pred_panel_names[i], pad_to_len=None)
                except EmptyPanelError:
                    continue
                if pred_panel_names[i] in gt_panel_names:
                    gt_edges, gt_rot, gt_transl, gt_aug_edges = gt_pattern.panel_as_numeric(pred_panel_names[i], pad_to_len=None)
                else:
                    continue
                correct_num_edges += int(len(edges) == len(gt_edges))
            total_num_edge_acc[k] = correct_num_edges / len(pred_panel_names)
            if not correct:
                continue
            garment_residual_error, garment_translation_l2, garment_rotation_l2 = 0, 0, 0
            num_edge_in_correct = 0
            for i in range(len(pred_panel_names)):
                edges, rot, transl, aug_edges = pred_pattern.panel_as_numeric(pred_panel_names[i], pad_to_len=None)
                gt_edges, gt_rot, gt_transl, gt_aug_edges = gt_pattern.panel_as_numeric(gt_panel_names[i], pad_to_len=None)
                num_edge_in_correct += int(len(edges) == len(gt_edges))
                max_length = max(len(edges), len(gt_edges))
                if len(edges) < max_length:
                    edges = np.pad(edges, ((0, max_length - len(edges)), (0, 0)), mode='constant', constant_values=0)
                if len(gt_edges) < max_length:
                    gt_edges = np.pad(gt_edges, ((0, max_length - len(gt_edges)), (0, 0)), mode='constant', constant_values=0)
                residual_error = np.linalg.norm(self._to_verts(edges) - self._to_verts(gt_edges), axis=1, ord=2).mean()
                transl_l2 = np.linalg.norm(transl - gt_transl, ord=2)
                rot_l2 = min(np.linalg.norm(rot - gt_rot, ord=2), np.linalg.norm(rot + gt_rot, ord=2))
                garment_residual_error += residual_error
                garment_translation_l2 += transl_l2
                garment_rotation_l2 += rot_l2
            
            total_num_edge_in_correct_acc[k] = num_edge_in_correct / max(len(pred_panel_names), 1)
            total_residual_error[k] = garment_residual_error / max(len(pred_panel_names), 1)
            total_translation_l2[k] = garment_translation_l2 / max(len(pred_panel_names), 1)
            total_rotation_l2[k] = garment_rotation_l2 / max(len(pred_panel_names), 1)
            if self.encode_stitches_as_tags:
                gt_stitches = gt_pattern.pattern['stitches']
                pred_stitches = pred_pattern.pattern['stitches']
                pred_stitches_dict = {}
                n_correct = 0
                for stitch_list in pred_stitches:
                    pred_stitches_dict[(stitch_list[0]['panel'], stitch_list[0]['edge'])] = (stitch_list[1]['panel'], stitch_list[1]['edge'])
                    pred_stitches_dict[(stitch_list[1]['panel'], stitch_list[1]['edge'])] = (stitch_list[0]['panel'], stitch_list[0]['edge'])
                for stitch_list in gt_stitches:
                    n_correct += 1 if (stitch_list[0]['panel'], stitch_list[0]['edge']) in pred_stitches_dict \
                        and pred_stitches_dict[(stitch_list[0]['panel'], stitch_list[0]['edge'])] == (stitch_list[1]['panel'], stitch_list[1]['edge']) else 0
                num_stitch_acc = n_correct / max(len(gt_stitches), 1)
                total_stitch_acc[k] = num_stitch_acc
        
        sorted_inds = np.arange(len(pred_patterns))[np.argsort(total_residual_error.cpu().numpy())[::-1]]
        total_residual_error.nan_to_num_(-1)
        return (
            total_num_panel_correct, 
            total_num_edge_acc, 
            total_num_edge_in_correct_acc, 
            total_residual_error, 
            total_translation_l2, 
            total_rotation_l2, 
            total_stitch_acc, 
            sorted_inds
        )
            
    def _pattern_as_list_sf(self, pattern: SF_SewingPattern, as_quat=False, endpoint_first=False):
        panel_order = pattern.panel_order(filter_nones=True)
        all_panel_edges, panel_rotations, panel_translations = [], [], []
        for panel_name in panel_order:
            panel_edges = []
            panel = pattern.pattern['panels'][panel_name]
            vertices = np.array(panel['vertices'])
            start_point = vertices[panel['edges'][0]["endpoints"]][0].copy()
            for i, edge in enumerate(panel['edges']):
                endpoints = vertices[edge["endpoints"]]
                edge_type = PanelEdgeTypeV3.LINE
                params = endpoints[1]
                if "curvature" in edge:
                    edge_type = PanelEdgeTypeV3.CURVE
                    params = np.concatenate([control_to_abs_coord(endpoints[0], endpoints[1], edge['curvature']), 
                                            endpoints[1]]) if not endpoint_first else np.concatenate([endpoints[1], 
                                            control_to_abs_coord(endpoints[0], endpoints[1], edge['curvature'])])
                    
                params = params.reshape(-1, 2) - start_point
                params = params.flatten()
                if i == len(panel['edges']) - 1:
                    edge_type = edge_type.get_closure()
                    # params = params[:-2]
                    params = params[:-2] if not endpoint_first else params[2:] # discard the last point
                panel_edges.append((edge_type, params))
            
            # ----- 3D placement convertion  ------
            # Global Translation (more-or-less stable across designs)
            translation, _ = panel_universal_transtation(vertices, panel['rotation'], panel['translation'])
            panel_translations.append(translation)
            panel_rotations.append(np.array(panel['rotation']) if not as_quat else Rotation.from_euler('xyz', panel['rotation'], degrees=True).as_quat())
            all_panel_edges.append(panel_edges)
        
        return all_panel_edges, panel_order, panel_rotations, panel_translations, pattern.pattern['stitches']
    
    

    def decode(self, output_dict: Dict[str, Any], tokenizer: PreTrainedTokenizer): 
        """Decode output ids to text"""
        output_ids = output_dict['output_ids']
        input_mask = output_dict['input_mask']
        text_output = tokenizer.decode(output_ids[output_ids != IMAGE_TOKEN_INDEX], skip_special_tokens=True)
        output_ids = output_ids.cpu().numpy().copy()
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
        pattern_dict, error_type = self.decode_pattern(pattern_tokens, tokenizer)
        pattern.pattern_from_pattern_dict(pattern_dict)
        return text_output, pattern, error_type
        
    def decode_pattern(self, token_sequence: np.ndarray, tokenizer: PreTrainedTokenizer): 
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
                return None, DecodeErrorTypes.UNMATCHED_PANEL_TOKENS
        
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
                edge_params = panel[command+1:command_end]
                
                num_should_predict = edge_type.get_num_params() + int(self.encode_stitches_as_tags and edge_type != PanelEdgeTypeV3.MOVE)
                if len(edge_params) < num_should_predict:
                    edge_params = np.pad(edge_params, (0, num_should_predict - len(edge_params)), 'constant')
                elif len(edge_params) > num_should_predict:
                    edge_params = edge_params[:num_should_predict]
                    
                if edge_type == PanelEdgeTypeV3.MOVE:
                    transl_params = self.get_bins_from_indices(edge_params[:3])
                    rot_params = self.get_bins_from_indices(edge_params[3:])
                    transl_params /= self.bin_size 
                    rot_params /= self.bin_size 
                    transl_params = transl_params * np.array(self.gt_stats.translations.scale) + np.array(self.gt_stats.translations.shift)
                    rot_params = rot_params * np.array(self.gt_stats.rotations.scale) + np.array(self.gt_stats.rotations.shift)
                    panel_dict['rotation'] = rot_params.tolist()
                    panel_dict['translation'] = transl_params.tolist()
                    continue
                
                if self.encode_stitches_as_tags:
                    edge_params, stitch_tag = edge_params[:-1], edge_params[-1]
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
                    edge_params = self.get_bins_from_indices(edge_params)
                    edge_params = edge_params / self.bin_size
                    endpoint = edge_params * np.array(self.gt_stats.vertices.scale) + np.array(self.gt_stats.vertices.shift)
                elif edge_type == PanelEdgeTypeV3.CURVE:
                    edge_params = self.get_bins_from_indices(edge_params)
                    edge_params = edge_params / self.bin_size
                    edge_params = edge_params.reshape(2, 2) * self.gt_stats.vertices.scale + self.gt_stats.vertices.shift
                    abs_ctrl_pt, endpoint = edge_params[0], edge_params[1]
                    ctrl_pt = control_to_relative_coord(last_point, endpoint, abs_ctrl_pt)
                    edge_dict['curvature'] = {
                        "type": 'quadratic',
                        "params": [ctrl_pt]
                    }
                elif edge_type == PanelEdgeTypeV3.CLOSURE_CURVE:
                    edge_params = self.get_bins_from_indices(edge_params)
                    edge_params = edge_params / self.bin_size
                    edge_params = edge_params * self.gt_stats.vertices.scale + self.gt_stats.vertices.shift
                    ctrl_pt = control_to_relative_coord(last_point, np.array([0, 0]), edge_params)
                    edge_dict['curvature'] = {
                        "type": 'quadratic',
                        "params": [ctrl_pt]
                    }
                elif edge_type == PanelEdgeTypeV3.CUBIC:
                    edge_params = self.get_bins_from_indices(edge_params)
                    edge_params = edge_params / self.bin_size
                    edge_params = edge_params.reshape(3, 2) * self.gt_stats.vertices.scale + self.gt_stats.vertices.shift
                    abs_ctrl_pt1, abs_ctrl_pt2, endpoint = edge_params[0], edge_params[1], edge_params[2]
                    ctrl_pt1 = control_to_relative_coord(last_point, endpoint, abs_ctrl_pt1)
                    ctrl_pt2 = control_to_relative_coord(last_point, endpoint, abs_ctrl_pt2)
                    edge_dict['curvature'] = {
                        "type": 'cubic',
                        "params": [ctrl_pt1, ctrl_pt2]
                    }
                elif edge_type == PanelEdgeTypeV3.CLOSURE_CUBIC:
                    edge_params = self.get_bins_from_indices(edge_params)
                    edge_params = edge_params / self.bin_size
                    edge_params = edge_params.reshape(2, 2) * self.gt_stats.vertices.scale + self.gt_stats.vertices.shift
                    ctrl_pt1 = control_to_relative_coord(last_point, np.array([0, 0]), edge_params[0])
                    ctrl_pt2 = control_to_relative_coord(last_point, np.array([0, 0]), edge_params[1])
                    edge_dict['curvature'] = {
                        "type": 'cubic',
                        "params": [ctrl_pt1, ctrl_pt2]
                    }
                elif edge_type == PanelEdgeTypeV3.ARC:
                    edge_params = self.get_bins_from_indices(edge_params)
                    edge_params = edge_params / self.bin_size
                    edge_params = edge_params.reshape(2, 2) * self.gt_stats.vertices.scale + self.gt_stats.vertices.shift
                    abs_ctrl_pt, endpoint = edge_params[0], edge_params[1]
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
                    edge_params = self.get_bins_from_indices(edge_params)
                    edge_params = edge_params / self.bin_size
                    edge_params = edge_params * self.gt_stats.vertices.scale + self.gt_stats.vertices.shift
                    if is_colinear(last_point, np.array([0, 0]), edge_params):
                        # arc became colinear due to rounding errors. Drawing a line instead.
                        edge_type = PanelEdgeTypeV3.CLOSURE_LINE
                    else:
                        _, _, rad, large_arc, right = arc_from_three_points(last_point, np.array([0, 0]), edge_params)
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
