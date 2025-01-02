import json
import numpy as np
import os
from pathlib import Path, PureWindowsPath
import shutil
import glob
from PIL import Image
import random
import time

import torch, code
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.io import read_image
import torchvision.transforms as T

# Do avoid a need for changing Evironmental Variables outside of this script
import os,sys
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)

sys.path.insert(0, parentdir) 
root_path = os.path.dirname(os.path.dirname(os.path.abspath(parentdir)))
pkg_path = "{}/SewFactory/packages".format(root_path)
print(pkg_path)
sys.path.insert(0, pkg_path)
sys.path.insert(0, os.path.dirname(parentdir))



# My modules
from customconfig import Properties
from data.pattern_converter import NNSewingPattern, InvalidPatternDefError
import data.transforms as transforms
from data.panel_classes import PanelClasses

from data.human_body_prior.body_model import BodyModel
from data.utils import euler_angle_to_rot_6d


class TokenDataset(Dataset):

    def __init__(self, 
                 datasets: dict, 
                 root_folder: str,
                 panel_classes_folder='/home/timur/garment_foundation_model/sewformer/SewFormer/assets/data_configs/panel_classes_condenced.json',
                 num_threads: int = 30,
                 ): 
        
        self.panel_classifier = PanelClasses(panel_classes_folder)
        self.datasets = datasets
        self.garment_files = []
        self.root_folder = root_folder
        self.categories=['dress_sleeveless', 'jacket', 'jumpsuit_sleeveless', 'skirt_4_panels', 'tee', 'wb_pants_straight', 'jacket_hood', 'pants_straight_sides', 'skirt_2_panels', 'skirt_8_panels', 'tee_sleeveless', 'wb_dress_sleeveless']
        self.cached = {}

        if 'neural_tailor' in self.datasets:
            self.garment_files += self._load_neuraltailor(datasets['neural_tailor'])
        if 'sewformer' in self.datasets:
            self.garment_files += self._load_sewformer(datasets['sewformer'])
        if 'garmentcodedata' in self.datasets:
            self.garment_files += self._load_garmentcodedata(datasets['garmentcodedata'])
        

    def _load_neuraltailor(self, path_to_neural_tailor):
        neural_tailor_path = f'{self.root_folder}/{path_to_neural_tailor}'

        data_folders = ['dress_sleeveless_2550', 'jacket_2200', 'jumpsuit_sleeveless_2000', 'skirt_4_panels_1600', 'tee_2300', 
                        'wb_pants_straight_1500', 'jacket_hood_2700', 'pants_straight_sides_1000', 'skirt_2_panels_1200', 'skirt_8_panels_1000', 
                        'tee_sleeveless_1800', 'wb_dress_sleeveless_2600']
        
        templates = [(f, '_'.join(f.split('_')[:-1])) for f in data_folders]
        all_garments = []

        for (data_folder, template_name) in templates:
              
            local_folders =  next(os.walk(f'{neural_tailor_path}/{data_folder}'))[1]
            local_garments = [(neural_tailor_path + '/' + data_folder + '/' + f + '/specification.json', template_name) for f in local_folders]
            all_garments += local_garments

        return all_garments
        
    
    def _load_sewformer(self, path_to_sewformer):
        sewformer_path = f'{self.root_folder}/{path_to_sewformer}'
        folders = next(os.walk(sewformer_path))[1]

        tee_folders = [f for f in folders if 'tee' in f]
        tee_sleeveless = [f for f in tee_folders if 'sleeveless' in f]
        tee_long_sleeve = [f for f in tee_folders if 'sleeveless' not in f]

        tee_sleeveless_bottom = [(sewformer_path + f + '/static/' + '_'.join(f.split('_')[3:]) + '_specification.json', '_'.join(f.split('_')[3:-1])) for f in tee_sleeveless]
        tee_long_sleeve_bottom = [(sewformer_path + f + '/static/' + '_'.join(f.split('_')[2:]) + '_specification.json', '_'.join(f.split('_')[2:-1])) for f in tee_long_sleeve]

        tee_sleeveless_top = [(sewformer_path + f + '/static/' + '_'.join(f.split('_')[:3]) + '_specification.json', '_'.join(f.split('_')[:2])) for f in tee_sleeveless]
        tee_long_sleev_top = [(sewformer_path + f + '/static/' + '_'.join(f.split('_')[:2]) + '_specification.json', '_'.join(f.split('_')[:1])) for f in tee_sleeveless]

        not_tee = [(sewformer_path + f + '/static/' + f + '_specification.json', '_'.join(f.split('_')[:-1])) for f in folders if f not in tee_folders and 'pycache' not in f]

        return not_tee + tee_sleeveless_bottom + tee_long_sleeve_bottom + tee_sleeveless_top + tee_long_sleev_top
    

    def _load_garmentcodedata(self, path_to_garmentcodedata):
        root_folder = Path(self.root_folder)
        garmentcodedata_path= root_folder / path_to_garmentcodedata
        garments = next(os.walk(garmentcodedata_path))[1]        
        all_garments = []
        for garment in garments:
            garment_path = garmentcodedata_path / garment
            
            all_garments.append((str((garment_path / f'{garment}_specification.json')), garment))
        return all_garments
    

    def _encode(self, 
                ground_truth, # for now just the number of edges
                category,
                base_vocab_size=0, 
                dim_edges=128, 
                categories=10, 
                min_range = np.array([-1.69588947, -2.79317152, -0.59022361, -2.00742926]),
                max_range = np.array([ 3.93797993,  2.79317152,  2.83853553,  3.74875851]),
                edge_scale = np.array([26.63403268, 29.54941741,  0.27706816,  0.2372687 ]),
                edge_shift = np.array([1.14394751e-17, 5.51454519e-19, 1.63532172e-01, 7.63001275e-02]),
               ):

        ## standardize, between 0 and 1 
        offset = base_vocab_size + dim_edges + categories
        # 1 = garment start, 2 = panel start, 3 = panel end, 4 = garment end, 5 = zero padding
        s_tokens = np.arange(5) + offset

        outlines = ground_truth['outlines']
        outlines = np.clip((outlines - edge_shift) / edge_scale, min_range, max_range)
        outlines = (outlines - min_range) / (max_range - min_range)
        outlines = outlines * dim_edges + base_vocab_size
        outlines = outlines.astype(int)
        
        valid_edges = ground_truth['num_edges']

        # garment start with category first, the garment start token
        tokens = np.array([category + dim_edges + base_vocab_size, s_tokens[0].item()])
        
        # all panels
        for panel_id in ground_truth['num_edges'].nonzero()[0]:
            tokens = np.concatenate((    tokens, 
                                    s_tokens[2:3],
                                    outlines[panel_id][:valid_edges[panel_id]].flatten(), 
                                    s_tokens[3:4]))
        #garment end
        tokens = np.concatenate((tokens, np.array([s_tokens[1]])))
        to_pad = 1024 - len(tokens)
        tokens = np.concatenate((tokens, np.array([s_tokens[4]] * to_pad)))

        return np.array(tokens, dtype=np.uint8)
        

    def _decode_tokens( self, 
                        token_sequence, 
                        bin_size=128, 
                        categories=10,
                        pad_panels_to_len=14, 
                        pad_panel_num=23,
                        base_vocab_size=0,
                        min_range = np.array([-1.69588947, -2.79317152, -0.59022361, -2.00742926]),
                        max_range = np.array([ 3.93797993,  2.79317152,  2.83853553,  3.74875851]),
                        edge_scale = np.array([26.63403268, 29.54941741,  0.27706816,  0.2372687 ]),
                        edge_shift = np.array([1.14394751e-17, 5.51454519e-19, 1.63532172e-01, 7.63001275e-02]),
                        ): 
        offset = base_vocab_size + bin_size + categories

        # 1 = garment start, 2 = panel start, 3 = panel end, 4 = garment end, 5 = zero padding
        s_tokens = np.arange(5) + offset

        # find the garment start
        start = np.where(token_sequence == s_tokens[0])[0]
        end = np.where(token_sequence == s_tokens[3])[0]


        # check of the garment start and ends correctly 
        if len(start) == 0 or len(end) == 0:
            return None, 'start-end-fault'
        if start[0] > end[0]:
            return None, 'start-end-fault'
        
        token_sequence= token_sequence[start[0]+1:end[0]]

        # find the panel starts and ends
        panel_starts = np.where(token_sequence == s_tokens[1])[0]
        panel_ends = np.where(token_sequence == s_tokens[2])[0]


        # check if panels starting and endings are correct or not
        if len(panel_starts) != len(panel_ends):
            return None, 'panel-start-end-fault'
        if np.any(panel_starts > panel_ends):
            return None, 'panel-start-end-fault'
        
        panels = []

        for panel in range(len(panel_starts)):
            panel_start = panel_starts[panel]
            panel_end = panel_ends[panel]
            panel = token_sequence[panel_start+1:panel_end]
            if len(panel) % 4 != 0:
                panels.append(None)
                continue

            edges = panel.reshape(-1, 4)
            edges = edges.astype(float)
            edges = (edges - base_vocab_size) / bin_size
            edges = edges * (max_range - min_range) + min_range
            edges = edges * edge_scale + edge_shift
            panels.append(edges)
        
        correct_panels = [panel for panel in panels if panel is not None]
        incorect_panels = len(panels) - len(correct_panels)
        
        padded_panels = []
        for i in range(pad_panel_num): 
            padded_panel = np.zeros((pad_panels_to_len, 4))
            if i < len(correct_panels):
                panel = correct_panels[i]
                padded_panel[:len(panel)] = panel
            padded_panels.append(padded_panel)

        padded_panels = np.stack(padded_panels)
        return padded_panels, incorect_panels
        

    def __len__(self):
        return len(self.garment_files)
    

    def __getitem__(self, idx):
        datapoint_name = self.garment_files[idx]
        ground_truth = self._get_sample_info(datapoint_name)
        return ground_truth
    
    def _get_sample_info(self, datapoint_name): 
        """
            Get features and Ground truth prediction for requested data example
        """
        if datapoint_name in self.cached:  # might not be compatible with list indexing
            ground_truth = self.gt_cached[datapoint_name]
        else:
            ground_truth = self._load_garmentcodedata_garment(datapoint_name)
            tokens = self._encode(ground_truth, self.categories.index(datapoint_name[1]))
            ground_truth.update({'tokens': tokens}) # add tokens to ground truth
        return ground_truth


    def _load_garment(  self,
                        source_template_name,
                        pad_panels_to_len=50, 
                        pad_panel_num=37, 
                        pad_stitches_num=None,
                        ):
            """Read given pattern in tensor representation from file"""  
            source = source_template_name[0]
            template_name = source_template_name[1]
            pattern = NNSewingPattern(
                        source, 
                        panel_classifier=self.panel_classifier, 
                        template_name=template_name)
            patterns = [pattern]

            pat_tensor = NNSewingPattern.multi_pattern_as_tensors(patterns,
                pad_panels_to_len, pad_panels_num=pad_panel_num, pad_stitches_num=pad_stitches_num,
                with_placement=True, with_stitches=True, 
                with_stitch_tags=True, spec_dict=None)
            
            pattern, num_edges, num_panels, rots, tranls, stitches, num_stitches, stitch_adj, stitch_tags, aug_outlines = pat_tensor
            free_edges_mask = self._free_edges_mask(pattern, stitches, num_stitches)
            empty_panels_mask = self._empty_panels_mask(num_edges)  # useful for evaluation
            gt = {
                    'outlines': pattern, 'num_edges': num_edges,
                    'rotations': rots, 'translations': tranls, 
                    'num_panels': num_panels, 'empty_panels_mask': empty_panels_mask, 'num_stitches': num_stitches,
                    'stitches': stitches, 'free_edges_mask': free_edges_mask, 'stitch_tags': stitch_tags
                }
            return gt





    ## need to rewrite (or copy from maria's code) the original way of 
    def _load_garmentcodedata_garment(self, 
                                      source_template_name,
                                      pad_panels_to_len=50, 
                                      pad_panel_num=37,
                                      pad_stitches_num=None,
                                      ):
        source = source_template_name[0]
        template_name = source_template_name[1]
        import code; code.interact(local=locals())
        pattern = NNSewingPattern(
                        source, 
                        panel_classifier=None, 
                        template_name=None)




        pass


    def _free_edges_mask(self, pattern, stitches, num_stitches):
        """
        Construct the mask to identify edges that are not connected to any other
        """
        mask = np.ones((pattern.shape[0], pattern.shape[1]), dtype=bool)
        max_edge = pattern.shape[1]

        for side in stitches[:, :num_stitches]:  # ignore the padded part
            for edge_id in side:
                mask[edge_id // max_edge][edge_id % max_edge] = False
        
        return mask
    
    def _empty_panels_mask(self, num_edges):
        """Empty panels as boolean mask"""

        mask = np.zeros(len(num_edges), dtype=bool)
        mask[num_edges == 0] = True

        return mask


    def standardize_garmentcode_data(self, batch): 
        feature_shift, feature_scale = self._get_distribution_stats(batch['features'], padded=False)

        gt = batch['ground_truth']
        panel_shift, panel_scale = self._get_distribution_stats(gt['outlines'], padded=True)
        # NOTE mean values for panels are zero due to loop property 
        # panel components SHOULD NOT be shifted to keep the loop property intact 
        panel_shift[0] = panel_shift[1] = 0

        # Use min\scale (normalization) instead of Gaussian stats for translation
        # No padding as zero translation is a valid value
        transl_min, transl_scale = self._get_norm_stats(gt['translations'])
        rot_min, rot_scale = self._get_norm_stats(gt['rotations'])

        # stitch tags if given
        st_tags_min, st_tags_scale = self._get_norm_stats(gt['stitch_tags'])

        self.standardization_dict_garmentcodedata = {
            'f_shift': feature_shift.cpu().numpy(), 
            'f_scale': feature_scale.cpu().numpy(),
            'gt_shift': {
                'outlines': panel_shift.cpu().numpy(), 
                'rotations': rot_min.cpu().numpy(),
                'translations': transl_min.cpu().numpy(), 
                'stitch_tags': st_tags_min.cpu().numpy()
            },
            'gt_scale': {
                'outlines': panel_scale.cpu().numpy(), 
                'rotations': rot_scale.cpu().numpy(),
                'translations': transl_scale.cpu().numpy(),
                'stitch_tags': st_tags_scale.cpu().numpy()
            }
            }
        print(self.standardization_dict_garmentcodedata)
        
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


if __name__ == '__main__':
    data = TokenDataset({'garmentcodedata': 'garmentData'}, '/miele/george')
    data[0]
    pass




    



