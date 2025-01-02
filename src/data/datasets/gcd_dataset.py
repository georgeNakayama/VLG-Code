import numpy as np
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset
from typing import List, Dict, Tuple, Union, Optional, Literal
import cv2
import random
import json
# My
from data.patterns.gcd_pattern.pattern_converter import NNSewingPattern
from data.datasets.garmentcodedata.panel_classes import PanelClasses
from data.datasets.utils import (SHORT_QUESTION_LIST, 
                                 ANSWER_LIST, 
                                 DEFAULT_PLACEHOLDER_TOKEN, 
                                 DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST, 
                                 SPECULATIVE_TEXT_SHORT_QUESTION_LIST, 
                                 SHORT_QUESTION_WITH_TEXT_LIST,
                                 EDITING_QUESTION_LIST
                                 )

## incorperating the changes from maria's three dataset classes into a new
## dataset class. this also includes features from sewformer, for interoperability
class GarmentCodeData(Dataset):  
    def __init__(
        self, 
        root_dir: str, 
        editing_dir: str, 
        caption_dir: str, 
        sampling_rate: List[int],
        datalist_file: str,
        editing_flip_prob: float,
        split_file: str,
        body_type: Literal['default_body'],
        panel_classification: Optional[str] = None,
        split: Literal["train", "val"] = "train"
        ): 

        self.editing_dir = editing_dir
        self.editing_flip_prob = editing_flip_prob
        self.caption_dir = caption_dir
        self.sampling_rate=sampling_rate
        self.panel_classification = panel_classification
        
        #################################
        # init from the basedataset class
        self.root_path = Path(root_dir)

        self.datapoints_names = []
        self.panel_classes = []
        
        self.short_question_list = SHORT_QUESTION_LIST
        self.descriptive_text_question_list = DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST
        self.speculative_text_question_list = SPECULATIVE_TEXT_SHORT_QUESTION_LIST
        self.text_image_question_list = SHORT_QUESTION_WITH_TEXT_LIST
        self.editing_question_list = EDITING_QUESTION_LIST
        self.answer_list = ANSWER_LIST
        self.split = split
        self.datapoints_names = json.load(open(split_file, "r"))[split]
                
        self.panel_classifier = PanelClasses(classes_file=panel_classification)
        self.panel_classes = self.panel_classifier.classes

        print("The panel classes in this dataset are :", self.panel_classes)
        

        self.gt_cached = {}
        self.gt_caching = True

    # added from maria 
    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints_names)  
    
    def _parepare_image(self, image_paths):
        """Fetch the image for the given index"""
        image_path = random.choice(image_paths)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_path
    
    def get_mode_names(self):
        return ['image', 'description','occasion', 'text_image', 'editing']

    # added from maria 
    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data. 
        Does not support list indexing"""
        
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        datapoint_name = self.datapoints_names[idx]
        data_name = datapoint_name.split('/')[-1]
        image_paths = [
            os.path.join(self.root_path, datapoint_name, f'{data_name}_render_back.png'),
            os.path.join(self.root_path, datapoint_name, f'{data_name}_render_front.png')
        ]
        if data_name in self.gt_cached:
            gt_pattern, edited_pattern, editing_captions, captions = self.gt_cached[data_name]
        else:
            spec_file = os.path.join(self.root_path, datapoint_name, f'{data_name}_specification_shifted.json')
            gt_pattern = NNSewingPattern(spec_file, panel_classifier=self.panel_classifier, template_name=data_name)
            gt_pattern.name = data_name
            
            editing_spec_file = os.path.join(self.editing_dir, data_name, f'edited_specification.json')
            editing_caption_json = os.path.join(self.editing_dir, data_name, f'editing_caption.json')
            if (not os.path.exists(editing_spec_file)) or (not os.path.exists(editing_caption_json)):
                edited_pattern = None
                editing_captions = None
            else:
                edited_pattern = NNSewingPattern(editing_spec_file, panel_classifier=self.panel_classifier, template_name=data_name)
                edited_pattern.name = data_name
                editing_captions = json.load(open(editing_caption_json, 'r'))
            
            caption_json = os.path.join(self.caption_dir, data_name, f'captions.json')
            if os.path.exists(caption_json):
                captions = json.load(open(caption_json, 'r'))
            else:
                captions = None
            
            self.gt_cached[data_name] = (gt_pattern, edited_pattern, editing_captions, captions)
            
            
        image = torch.zeros((3, 224, 224))
        image_path = ''
        sample_type = np.random.choice(5, p=self.sampling_rate)
        if sample_type == 4 and edited_pattern is None:
            sample_type = 0  # no editing if there is no edited pattern
        if (sample_type in [1, 2, 3]) and captions is None:
            sample_type = 0  # no text if there is no caption
        if sample_type == 0:
            # image_only
            image, image_path = self._parepare_image(image_paths)
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.short_question_list)
                questions.append([{"type": "image"}, {"type": "text", "text": question_template}])
                answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
                answers.append([{"type": "text", "text": answer_template}])
            out_pattern = [gt_pattern]
        elif sample_type == 1:
            # descriptive text_only
            descriptive_text = captions['description']
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.descriptive_text_question_list).format(sent=descriptive_text)
                questions.append([{"type": "image"}, {"type": "text", "text": question_template}])
                answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
                answers.append([{"type": "text", "text": answer_template}])
            out_pattern = [gt_pattern]
        elif sample_type == 2:
            # speculative text_only
            speculative_text = captions['occasion']
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.speculative_text_question_list).format(sent=speculative_text)
                questions.append([{"type": "text", "text": question_template}])
                answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
                answers.append([{"type": "text", "text": answer_template}])
            out_pattern = [gt_pattern]
        elif sample_type == 3:
            # image_text
            descriptive_text = captions['description']
            image, image_path = self._parepare_image(image_paths)
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.text_image_question_list).format(sent=descriptive_text)
                questions.append([{"type": "image"}, {"type": "text", "text": question_template}])
                answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
                answers.append([{"type": "text", "text": answer_template}])
            out_pattern = [gt_pattern]
        elif sample_type == 4:
            # garment_editing
            if random.random() > self.editing_flip_prob:
                before_pattern = gt_pattern
                after_pattern = edited_pattern
                editing_text = editing_captions['editing_description_forward']
            else:
                before_pattern = edited_pattern
                after_pattern = gt_pattern
                editing_text = editing_captions['editing_description_reverse']
                
            before_pattern.name = "before_" + before_pattern.name
            after_pattern.name = "after_" + after_pattern.name
            out_pattern = [before_pattern, after_pattern]
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.editing_question_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN, sent=editing_text)
                questions.append([{"type": "text", "text": question_template}])
                answer_template = random.choice(self.answer_list).format(pattern=DEFAULT_PLACEHOLDER_TOKEN)
                answers.append([{"type": "text", "text": answer_template}])

        # dialog = [{"role":"system","content":[{"type": "text", "text": system_prompt}]}]
        dialog = []
        # question = [{"role":"system","content":[{"type": "text", "text": system_prompt}]}]
        question = []

        for i in range(len(questions)):
            dialog.append({"role": "user", "content": questions[i]})
            dialog.append({"role": "assistant", "content": answers[i]})
            question.append({"role": "user", "content": questions[i]})

        return (
            image_path,
            image,
            dialog,
            question,
            out_pattern,
            sample_type,
        ) 
        
    def evaluate_patterns(self, pred_patterns: List[NNSewingPattern], gt_patterns: List[NNSewingPattern]):
        return self.garment_tokenizer.evaluate_patterns(pred_patterns, gt_patterns)