"""
    List of metrics to evalute on a model and a dataset, along with pre-processing methods needed for such evaluation
"""

import torch
import torch.nn.functional as F
import logging 

log = logging.getLogger(__name__)
# My modules
# from data import Garment3DPatternFullDataset as PatternDataset
from metrics.eval_detr_metrics import * 
from data.datasets.dataset_pcd_hydra import StandardizeConfig, PanelConfig, StatsConfig

"""
# # ------- custom quality metrics --------
# class PatternStitchPrecisionRecall():
#     ""Evaluate Precision and Recall scores for pattern stitches prediction
#         NOTE: It's NOT a diffentiable evaluation
#     ""

#     def __init__(self, data_stats=None):
#         self.data_stats = data_stats
#         if data_stats is not None: 
#             for key in self.data_stats:
#                 self.data_stats[key] = torch.Tensor(self.data_stats[key])

#     def __call__(
#             self, 
#             stitch_tags, free_edge_class, 
#             gt_stitches, gt_stitches_nums, 
#             pattern_names=None, correct_mask=None):
#         ""
#          Evaluate on the batch of stitch tags
#         ""
#         # undo stats application if provided
#         if self.data_stats is not None:
#             device = stitch_tags.device
#             stitch_tags = stitch_tags * self.data_stats['scale'].to(device) + self.data_stats['shift'].to(device)
        
#         tot_precision = 0.
#         tot_recall = 0.
#         corr_precision = []
#         corr_recall = []
#         for pattern_idx in range(stitch_tags.shape[0]):
#             stitch_list = PatternDataset.tags_to_stitches(stitch_tags[pattern_idx], free_edge_class[pattern_idx]).to(gt_stitches.device)

#             num_detected_stitches = stitch_list.shape[1] if stitch_list.numel() > 0 else 0
#             if not num_detected_stitches:  # no stitches detected -- zero recall & precision
#                 continue
#             num_actual_stitches = gt_stitches_nums[pattern_idx]
            
#             # compare stitches
#             correct_stitches = 0.
#             for detected in stitch_list.transpose(0, 1):
#                 for actual in gt_stitches[pattern_idx][:, :gt_stitches_nums[pattern_idx]].transpose(0, 1):
#                     # order-invariant comparison of stitch sides
#                     correct = (all(detected == actual) or all(detected == actual.flip([0])))
#                     correct_stitches += correct
#                     if correct:  # no need to check subsequent stitches
#                         break
 
#                 if pattern_names is not None and not correct:  # never detected a match with actual stitches
#                     print('StitchPrecisionRecall::{}::Stitch {} detected wrongly'.format(pattern_names[pattern_idx], detected))

#             # precision -- how many of the detected stitches are actually there
#             precision = correct_stitches / num_detected_stitches if num_detected_stitches else 0.
#             # recall -- how many of the actual stitches were detected
#             recall = correct_stitches / num_actual_stitches if num_actual_stitches else 0.

#             tot_precision += precision
#             tot_recall += recall
#             if correct_mask is not None and correct_mask[pattern_idx]:
#                 corr_precision.append(precision)
#                 corr_recall.append(recall)
        
#         # average by batch
#         return (
#             tot_precision / stitch_tags.shape[0], 
#             tot_recall / stitch_tags.shape[0],
#             sum(corr_precision) / len(corr_precision) if len(corr_precision) else None,
#             sum(corr_recall) / len(corr_recall) if len(corr_recall) else None
#         )

#     def on_loader(self, data_loader, model):
#         ""Evaluate recall&precision of stitch detection on the full data loader""

#         with torch.no_grad():
#             tot_precision = tot_recall = 0
#             for batch in data_loader:
#                 predictions = model(batch['features'])
#                 batch_precision, batch_recall = self(predictions['stitch_tags'], batch['ground_truth']['stitches'], batch['name'])
#                 tot_precision += batch_precision
#                 tot_recall += batch_recall

#         return tot_precision / len(data_loader), tot_recall / len(data_loader)
"""


class PanelChamferLoss(): 
    def __init__(self): 
        pass

    def __call__(self, predicted_outline, gt_outline):
        #batch x points x 2
        predicted_outline = predicted_outline.cuda()
        gt_outline = gt_outline.cuda()
        
        from metrics.chamfer_dist import ChamferDistanceL2
        chamfer = ChamferDistanceL2(redue=False)
        dist = chamfer(predicted_outline, gt_outline)

        return dist.mean(dim=1).cpu()

class PanelSetChamferLoss(): 

    def __init__(self) -> None:
        pass

    def __call__(self, pred_panels: torch.FloatTensor, gt_panels: torch.FloatTensor, pred_panel_nums: torch.IntTensor, gt_panel_nums: torch.IntTensor):
        """
        Args:
            pred_panels [batch x max_n_panels x points x 2]: predicted panel outlines, the valid pred panels are at the beginning of the tensor (number of valid panels is stored in pred_panel_nums)
            gt_panels [batch x max_n_panels x points x 2]: predicted panel outlines, the valid GT panels are at the beginning of the tensor (number of valid panels is stored in gt_panel_nums)
            pred_panel_nums [batch]: number of pred panels in each pattern
            gt_panels [batch]: number of GT panels in each pattern
            """
        device = pred_panels.device
        pred_panels = pred_panels.to(device)
        gt_panels = gt_panels.to(device)

        concat_pred_panels = pred_panels.reshape(pred_panels.shape[0], -1, 2)

        overall_chamfer = torch.zeros(gt_panels.shape[0])
        for i in range(gt_panels.shape[1]):
            dist = torch.cdist(concat_pred_panels, gt_panels[:,i, :,:])
            dist = dist.reshape(pred_panels.shape[0], pred_panels.shape[1], pred_panels.shape[2], -1)
            min_dist0, _ = torch.min(dist, dim=3)
            min_dist1, _ = torch.min(dist, dim=2)

            minimum = min_dist0.mean(dim=2) + min_dist1.mean(dim=2)
            increment, _ = torch.min(minimum, dim=1)
            overall_chamfer += increment

        return overall_chamfer


class NumbersInPanelsAccuracies():
    """
        Evaluate in how many cases the number of panels in patterns and number of edges in panels were detected correctly
    """
    def __init__(self, panel_info_config: PanelConfig, standardize_config: StatsConfig):
        """
            Requesting data stats to recognize padding correctly
            Should be a dictionary with {'shift': <>, 'scale': <>} keys containing stats for panel outlines
        """
        self.data_stats = {
            'shift': torch.tensor(standardize_config.shift),
            'scale': torch.tensor(standardize_config.scale),
        }
        self.max_panel_len = panel_info_config.max_panel_len
        self.pad_vector = eval_pad_vector(self.data_stats)
        self.empty_panel_template = self.pad_vector.repeat(self.max_panel_len, 1)
        self.panel_loop_threshold = torch.tensor([3, 3]) / torch.Tensor(self.data_stats['scale'])[:2]  # 3 cm per coordinate is a tolerable error

    def __call__(self, predicted_outlines: torch.Tensor, gt_num_edges, gt_panel_nums, pattern_names=None):
        """
         Evaluate on the batch of panel outlines predictoins 
        """

        batch_size = predicted_outlines.shape[0]
        max_num_panels = predicted_outlines.shape[1]
        max_num_edges = predicted_outlines.shape[2]
        if self.empty_panel_template.device != predicted_outlines.device:
            self.empty_panel_template = self.empty_panel_template.to(predicted_outlines.device)
            self.panel_loop_threshold = self.panel_loop_threshold.to(predicted_outlines.device)

        correct_num_panels = 0.
        num_edges_accuracies = 0.

        correct_pattern_mask = torch.zeros(batch_size, dtype=torch.bool)
        valid_panel_num = torch.zeros(batch_size, dtype=torch.int32)
        num_edges_in_correct = 0
        out_patterns = []
        valid_edge_num = []

        for pattern_idx in range(batch_size):
            # assuming all empty panels are at the end of the pattern, if any
            predicted_num_panels = 0
            correct_num_edges = 0.
            panel_list = []
            edge_num = []
            for panel_id in range(max_num_panels):
                # predicted_bool_matrix = torch.isclose(
                #     predicted_outlines[pattern_idx][panel_id], 
                #     self.empty_panel_template, atol=0.07)  # this value is adjusted to have similar effect to what is used in core.py
                all_edges = predicted_outlines[pattern_idx][panel_id]
                predicted_bool_matrix = torch.all(torch.isclose(all_edges[:, [0, 1]], self.empty_panel_template[:, [0, 1]], atol=0.07), dim=-1)
                all_edges = all_edges[~predicted_bool_matrix]  # remove padded edges
                # check is the num of edges matches
                predicted_num_edges = len(all_edges)  # only non-padded rows
                if predicted_num_edges < 2:
                    continue

                # check if edge loop closes
                edge_coords = all_edges[:, :2]
                loop_distance = edge_coords.sum(dim=0)    # empty edges are about zero anyway
                if (torch.abs(loop_distance) > self.panel_loop_threshold).any() and predicted_num_edges < max_num_edges:
                    # if the edge loop doesn't return to origin, 
                    # it basically means the need to create extra edge to force the closing
                        last_edge = torch.cat([-loop_distance, torch.zeros_like(edge_coords[0])]).reshape(1, 4)
                        all_edges = torch.cat([all_edges, last_edge], dim=0)  # add a copy of the first edge to close the loop
                        predicted_num_edges += 1
                else:
                    all_edges[-1, :2] -= loop_distance  # move the last edge to the origin to close the loop
            
                if predicted_num_edges < 3:
                    # 0, 1, 2 edges are not enough to form a panel
                    #  -> assuming this is an empty panel & moving on
                    continue
                if predicted_num_edges < max_num_edges:
                    all_edges = F.pad(all_edges, (0, 0, 0, max_num_edges - predicted_num_edges))  # pad with zeros
                panel_list.append(all_edges)
                edge_num.append(predicted_num_edges)
                # othervise, we have a real panel
                predicted_num_panels += 1

                panel_correct = (predicted_num_edges == gt_num_edges[pattern_idx * max_num_panels + panel_id])
                correct_num_edges += panel_correct

                if pattern_names is not None and not panel_correct:  # pattern len predicted wrongly
                    log.warn('{}::panel {}:: {} edges instead of {}'.format(
                        pattern_names[pattern_idx], panel_id,
                        predicted_num_edges, gt_num_edges[pattern_idx * max_num_panels + panel_id]))

            panel_list = torch.stack(panel_list, dim=0)
            edge_num = torch.tensor(edge_num)
            assert panel_list.shape[1] == max_num_edges and panel_list.shape[0] == predicted_num_panels
            if predicted_num_panels < max_num_panels:
                panel_list = F.pad(panel_list, (0, 0, 0, 0, 0, max_num_panels - predicted_num_panels))  # pad with zeros
                edge_num = F.pad(edge_num, (0, max_num_panels - predicted_num_panels))  # pad with zeros
            out_patterns.append(panel_list)
            valid_edge_num.append(edge_num)
            valid_panel_num[pattern_idx] = predicted_num_panels
            # update num panels stats
            correct_len = (predicted_num_panels == gt_panel_nums[pattern_idx])
            correct_pattern_mask[pattern_idx] = correct_len
            correct_num_panels += correct_len

            if pattern_names is not None and not correct_len:  # pattern len predicted wrongly
                log.warn('{}::{} panels instead of {}'.format(
                    pattern_names[pattern_idx], predicted_num_panels, gt_panel_nums[pattern_idx]))

            # update num edges stats (averaged per panel)
            num_edges_accuracies += correct_num_edges / gt_panel_nums[pattern_idx]
            if correct_len:
                num_edges_in_correct += correct_num_edges / gt_panel_nums[pattern_idx]
                
        valid_edge_num = torch.stack(valid_edge_num, dim=0)
        out_patterns = torch.stack(out_patterns, dim=0)
        # average by batch
        return (
            correct_num_panels / batch_size, 
            num_edges_accuracies / batch_size, 
            correct_pattern_mask,   # which patterns in a batch have correct number of panels? 
            num_edges_in_correct / correct_pattern_mask.sum(),  # edges for correct patterns
            out_patterns, 
            valid_panel_num, 
            valid_edge_num
        )
    

class PanelVertsL2():
    """
        Aims to evaluate the quality of panel shape prediction independently from loss evaluation
        * Convers panels edge lists to vertex representation (including curvature coordinates)
        * and evaluated MSE on them
    """
    def __init__(self, panel_info_config: PanelConfig, standardize_config: StatsConfig):
        """Info for evaluating padding vector if data statistical info is applied to it.
            * if standardization/normalization transform is applied to padding, 'data_stats' should be provided
                'data_stats' format: {'shift': <torch.tenzor>, 'scale': <torch.tensor>} 
        """
        self.data_stats = {
            'shift': torch.tensor(standardize_config.shift),
            'scale': torch.tensor(standardize_config.scale),
        }
        self.max_panel_len = panel_info_config.max_panel_len
        self.empty_panel_template = torch.zeros((panel_info_config.max_panel_len, len(self.data_stats['shift'])))
    
    def __call__(self, predicted_outlines, gt_outlines, gt_num_edges, correct_mask=None):
        """
            Evaluate on the batch of panel outlines predictoins 
            * per_panel_leading_edges -- specifies where is the start of the edge loop for GT outlines 
                that is well-matched to the predicted outlines. If not given, the default GT orientation is used
        """
        num_panels = predicted_outlines.shape[1]

        # flatten input into list of panels
        predicted_outlines = predicted_outlines.view(-1, predicted_outlines.shape[-2], predicted_outlines.shape[-1])
        gt_outlines = gt_outlines.view(-1, gt_outlines.shape[-2], gt_outlines.shape[-1])

        # devices
        if self.empty_panel_template.device != predicted_outlines.device:
            self.empty_panel_template = self.empty_panel_template.to(predicted_outlines.device)
        for key in self.data_stats:
            if self.data_stats[key].device != predicted_outlines.device:
                self.data_stats[key] = self.data_stats[key].to(predicted_outlines.device)

        # un-std
        predicted_outlines = predicted_outlines * self.data_stats['scale'] + self.data_stats['shift']
        gt_outlines = gt_outlines * self.data_stats['scale'] + self.data_stats['shift']

        # per-panel evaluation
        panel_errors = []
        correct_panel_errors = []
        # panel_mask = correct_mask.t().repeat(num_panels) if correct_mask is not None else None
        panel_mask = torch.repeat_interleave(correct_mask, num_panels) if correct_mask is not None else None

        # panel_mask = panel_mask.view(-1) if correct_mask is not None else None

        for panel_idx in range(len(predicted_outlines)):
            prediced_panel = predicted_outlines[panel_idx]
            gt_panel = gt_outlines[panel_idx]

            # unpad both panels using correct gt info -- for simplicity of comparison
            num_edges = gt_num_edges[panel_idx]
            if num_edges < 3:  # empty panel -- skip comparison
                continue
            prediced_panel = prediced_panel[:num_edges, :]  
            gt_panel = gt_panel[:num_edges, :]

            # average squred error per vertex (not per coordinate!!) hence internal sum
            panel_errors.append(
                torch.mean(torch.sqrt(((self._to_verts(gt_panel) - self._to_verts(prediced_panel)) ** 2).sum(dim=1)))
            )

            if panel_mask is not None and panel_mask[panel_idx]:
                correct_panel_errors.append(panel_errors[-1])
        
        # mean of errors per panel
        if panel_mask is not None and len(correct_panel_errors):
            return sum(panel_errors) / len(panel_errors), sum(correct_panel_errors) / len(correct_panel_errors)
        else:
            return sum(panel_errors) / len(panel_errors), None

    def _to_verts(self, panel_edges):
        """Convert normalized panel edges into the vertex representation"""

        vert_list = [torch.tensor([0, 0]).to(panel_edges.device)]  # always starts at zero
        # edge: first two elements are the 2D vector coordinates, next two elements are curvature coordinates
        for edge in panel_edges:
            next_vertex = vert_list[-1] + edge[:2]
            edge_perp = torch.tensor([-edge[1], edge[0]]).to(panel_edges.device)

            # NOTE: on non-curvy edges, the curvature vertex in panel space will be on the previous vertex
            #       it might result in some error amplification, but we could not find optimal yet simple solution
            next_curvature = vert_list[-1] + edge[2] * edge[:2]  # X curvature coordinate
            next_curvature = next_curvature + edge[3] * edge_perp  # Y curvature coordinate

            vert_list.append(next_curvature)
            vert_list.append(next_vertex)

        vertices = torch.stack(vert_list)

        # align with the center
        vertices = vertices - torch.mean(vertices, axis=0)  # shift to average coordinate

        return vertices


class UniversalL2():
    """
        Evaluate L2 on the provided (un-standardized) data -- useful for 3D placement
    """
    def __init__(self, standardize_config: StatsConfig):
        """Info for un-doing the shift&scale of the data 
        """
        self.data_stats = {
            'shift': torch.tensor(standardize_config.shift),
            'scale': torch.tensor(standardize_config.scale),
        }
    
    def __call__(self, predicted: torch.Tensor, gt, correct_mask=None):
        """
         Evaluate on the batch of predictions 
         Used for rotation/translation evaluations which have input shape
         (#batch, #panels, #feature)
        """
        num_panels = predicted.shape[1]
        correct_mask = torch.repeat_interleave(correct_mask, num_panels) if correct_mask is not None else None

        # flatten input 
        predicted = predicted.view(-1, predicted.shape[-1])
        gt = gt.view(-1, gt.shape[-1])

        # devices
        for key in self.data_stats:
            if self.data_stats[key].device != predicted.device:
                self.data_stats[key] = self.data_stats[key].to(predicted.device)

        # un-std
        predicted = predicted * self.data_stats['scale'] + self.data_stats['shift']
        gt = gt * self.data_stats['scale'] + self.data_stats['shift']

        L2_norms = torch.sqrt(((gt - predicted) ** 2).sum(dim=1))

        if correct_mask is not None and len(gt[correct_mask]):
            correct_L2_norms = torch.mean(torch.sqrt(((gt[correct_mask] - predicted[correct_mask]) ** 2).sum(dim=1)))
        else:
            correct_L2_norms = None

        return torch.mean(L2_norms), correct_L2_norms

    

if __name__ == "__main__":
    # debug

    stitch_eval = PatternStitchPrecisionRecall()

    tags = torch.FloatTensor(
        [[
            [
                [0, 0, 0],
                [1.2, 3., 0],
                [0, 0, 0]
            ],
            [
                [0, 3., 0],
                [0, 0, 0],
                [1.2, 3., 0],
            ]
        ]]
    )
    stitches = torch.IntTensor([
        [
            [1, 5]
        ]
    ]).transpose(0, 1)

    print(stitch_eval(tags, stitches))
