import numpy as np
import torch
from scipy.sparse import csr_matrix


# ------------------ Transforms ----------------
def _dict_to_tensors(dict_obj):  # helper
    """convert a dictionary with numeric values into a new dictionary with torch tensors"""
    new_dict = dict.fromkeys(dict_obj.keys())
    for key, value in dict_obj.items():
        if value is None:
            new_dict[key] = torch.Tensor()
        elif isinstance(value, dict):
            new_dict[key] = _dict_to_tensors(value)
        elif isinstance(value, str):  # no changes for strings
            new_dict[key] = value
        elif isinstance(value, np.ndarray):
            new_dict[key] = torch.from_numpy(value)

            # TODO more stable way of converting the types (or detecting ints)
            if value.dtype not in [int, np.int64, bool]:
                new_dict[key] = new_dict[key].float()  # cast all doubles and other stuff to floats
        elif isinstance(value, csr_matrix):
            new_dict[key] = torch.sparse_coo_tensor(np.array(value.nonzero()), value.data, value.shape)
            if key in ['aug_outlines', 'outlines']:  # It won't be sparse after GT standardization is applied
                new_dict[key] = new_dict[key].to_dense().reshape(dict_obj['translations'].shape[0], -1, value.shape[-1])
            elif key == 'free_edges_mask':
                new_dict[key] = ~new_dict[key].to_dense()   # Return the proper shape
            # NOTE: for the others, sparsity is preserved
            # NOTE: pin_memory=True in DataLoader won't work with sparse tensors. 
            if value.dtype not in [int, np.int64, bool]:
                new_dict[key] = new_dict[key].float() # cast all doubles and other stuff to floats
        else:
            new_dict[key] = torch.tensor(value)  # just try directly, if nothing else works
    return new_dict


# Custom transforms -- to tensor
class SampleToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __call__(self, sample):        
        return _dict_to_tensors(sample)


class FeatureStandartization():
    """Normalize features of provided sample with given stats"""
    def __init__(self, shift, scale):
        self.shift = torch.Tensor(shift)
        self.scale = torch.Tensor(scale)
    
    def __call__(self, sample):
        updated_sample = {}
        for key, value in sample.items():
            if key == 'features':
                updated_sample[key] = (sample[key] - self.shift) / self.scale
            else: 
                updated_sample[key] = sample[key]

        return updated_sample


class GTtandartization():
    """Normalize features of provided sample with given stats
        * Supports multimodal gt represented as dictionary
        * For dictionary gts, only those values are updated for which the stats are provided
    """
    def __init__(self, shift, scale):
        """If ground truth is a dictionary in itself, the provided values should also be dictionaries"""
        
        self.shift = _dict_to_tensors(shift) if isinstance(shift, dict) else torch.Tensor(shift)
        self.scale = _dict_to_tensors(scale) if isinstance(scale, dict) else torch.Tensor(scale)
    
    def __call__(self, sample):
        gt = sample['ground_truth']
        if isinstance(gt, dict):
            new_gt = dict.fromkeys(gt.keys())
            for key, value in gt.items():
                new_gt[key] = value
                if key in self.shift:
                    new_gt[key] = new_gt[key] - self.shift[key]
                if key in self.scale:
                    new_gt[key] = new_gt[key] / self.scale[key]
                
                if key == "aug_outlines" and "outlines" in self.shift:
                    new_gt[key] = new_gt[key] - self.shift["outlines"][:2]
                if key == "aug_outlines" and "outlines" in self.scale:
                     new_gt[key] = new_gt[key] / self.scale["outlines"][:2]
                if key == "aug_outlines":
                    new_gt[key] = new_gt[key].to(gt["outlines"].dtype)
                # if shift and scale are not set, the value is kept as it is
        else:
            new_gt = (gt - self.shift) / self.scale

        # gather sample
        updated_sample = {}
        for key, value in sample.items():
            updated_sample[key] = new_gt if key == 'ground_truth' else sample[key]

        return updated_sample
