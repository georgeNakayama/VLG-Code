import dataclasses
import enum
import json
import os
import random
from typing import Optional, Literal

import cv2
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset

from data.patterns.gcd_pattern.pattern_converter import NNSewingPattern
from data.patterns.gcd_pattern.panel_classes import PanelClasses
from .utils import (
    SHORT_QUESTION_LIST,
    ANSWER_LIST,
    DEFAULT_PLACEHOLDER_TOKEN,
    DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST,
    SPECULATIVE_TEXT_SHORT_QUESTION_LIST,
    SHORT_QUESTION_WITH_TEXT_LIST,
    EDITING_QUESTION_LIST,
)


class SampleType(enum.Enum):
    IMAGE = 0
    DESC_TEXT = 1
    SPEC_TEXT = 2
    IMAGE_AND_TEXT = 3
    EDIT = 4

    def __int__(self):
        return self.value


@dataclasses.dataclass
class GroundTruthPattern:
    default_pattern: NNSewingPattern
    random_pattern: Optional[NNSewingPattern]

## incorperating the changes from maria's three dataset classes into a new
## dataset class. this also includes features from sewformer, for interoperability
class GarmentCodeData(Dataset):
    def __init__(
        self,
        root_dir: str,
        editing_dir: str,
        caption_dir: str,
        sampling_rate: list[int],
        editing_flip_prob: float,
        split_file: str,
        panel_classification: Optional[str] = None,
        body_type: Literal["default_body", "random_body"] = "default_body",
        split: Literal["train", "val"] = "train",
    ):

        self.editing_dir = editing_dir
        self.editing_flip_prob = editing_flip_prob
        self.caption_dir = caption_dir
        self.sampling_rate = sampling_rate
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
        self.body_type = body_type

        print("The panel classes in this dataset are :", self.panel_classes)

        self.gt_cached = {}
        self.gt_caching = True

    # added from maria
    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints_names)

    def _prepare_image(self, image_paths):
        """Fetch the image for the given index"""
        image_path = random.choice(image_paths)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_path

    def get_mode_names(self):
        return ["image", "description", "occasion", "text_image", "editing"]

    # added from maria
    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data.
        Does not support list indexing"""

        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        datapoint_name = self.datapoints_names[idx]
        data_name = datapoint_name.split("/")[-1]
        image_paths = [
            os.path.join(
                self.root_path, datapoint_name, f"{data_name}_render_back.png"
            ),
            os.path.join(
                self.root_path, datapoint_name, f"{data_name}_render_front.png"
            ),
        ]
        if data_name in self.gt_cached:
            gt_pattern, edited_pattern, editing_captions, captions = self.gt_cached[
                data_name
            ]
        else:
            default_spec_file = os.path.join(
                self.root_path,
                datapoint_name,
                f"{data_name}_specification_shifted.json",
            )
            default_gt_pattern = NNSewingPattern(
                default_spec_file,
                panel_classifier=self.panel_classifier,
                template_name=data_name,
            )
            default_gt_pattern.name = data_name

            random_spec_file = str(default_spec_file).replace(
                "default_body", "random_body"
            )
            if not os.path.exists(random_spec_file):
                random_gt_pattern = None
            else:
                random_gt_pattern = NNSewingPattern(
                    random_spec_file,
                    panel_classifier=self.panel_classifier,
                    template_name=data_name,
                )
                random_gt_pattern.name = data_name

            gt_pattern = GroundTruthPattern(default_gt_pattern, random_gt_pattern)

            editing_spec_file = os.path.join(
                self.editing_dir, data_name, f"edited_specification.json"
            )
            editing_caption_json = os.path.join(
                self.editing_dir, data_name, f"editing_caption.json"
            )
            if (not os.path.exists(editing_spec_file)) or (
                not os.path.exists(editing_caption_json)
            ):
                edited_pattern = None
                editing_captions = None
            else:
                edited_pattern = NNSewingPattern(
                    editing_spec_file,
                    panel_classifier=self.panel_classifier,
                    template_name=data_name,
                )
                edited_pattern.name = data_name
                editing_captions = json.load(open(editing_caption_json, "r"))

            caption_json = os.path.join(self.caption_dir, data_name, f"captions.json")
            if os.path.exists(caption_json):
                captions = json.load(open(caption_json, "r"))
            else:
                captions = None

            self.gt_cached[data_name] = (
                gt_pattern,
                edited_pattern,
                editing_captions,
                captions,
            )

        use_random_body = self.body_type == "random_body" and bool(
            random.choice([True, False])
        )
        if use_random_body and gt_pattern.random_pattern is None:
            use_random_body = False

        if use_random_body:
            image_paths = [
                str(path).replace("default_body", "random_body") for path in image_paths
            ]

        image = torch.zeros((3, 800, 800))
        image_path = ""
        sample_type = np.random.choice(list(SampleType), p=self.sampling_rate)
        if sample_type == SampleType.EDIT and edited_pattern is None:
            sample_type = SampleType.IMAGE  # no editing if there is no edited pattern
        if (
            sample_type
            in [SampleType.DESC_TEXT, SampleType.SPEC_TEXT, SampleType.IMAGE_AND_TEXT]
        ) and captions is None:
            sample_type = SampleType.IMAGE  # no text if there is no caption
        if sample_type == SampleType.IMAGE:
            # image_only
            image, image_path = self._prepare_image(image_paths)
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.short_question_list)
                questions.append(
                    [{"type": "image"}, {"type": "text", "text": question_template}]
                )
                # answer_template = random.choice(self.answer_list).format(
                #     pattern=DEFAULT_PLACEHOLDER_TOKEN
                # )
                answer_template=DEFAULT_PLACEHOLDER_TOKEN
                answers.append([{"type": "text", "text": answer_template}])
            out_pattern = [
                (
                    gt_pattern.random_pattern
                    if use_random_body
                    else gt_pattern.default_pattern
                )
            ]
        elif sample_type == SampleType.DESC_TEXT:
            # descriptive text_only
            descriptive_text = captions["description"]
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(
                    self.descriptive_text_question_list
                ).format(sent=descriptive_text)
                questions.append(
                    [{"type": "image"}, {"type": "text", "text": question_template}]
                )
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append([{"type": "text", "text": answer_template}])
            out_pattern = [gt_pattern.default_pattern]
        elif sample_type == SampleType.SPEC_TEXT:
            # speculative text_only
            speculative_text = captions["occasion"]
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(
                    self.speculative_text_question_list
                ).format(sent=speculative_text)
                questions.append(
                    [{"type": "image"}, {"type": "text", "text": question_template}]
                )
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append([{"type": "text", "text": answer_template}])
            out_pattern = [gt_pattern.default_pattern]
        elif sample_type == SampleType.IMAGE_AND_TEXT:
            # image_text
            descriptive_text = captions["description"]
            image, image_path = self._prepare_image(image_paths)
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.text_image_question_list).format(
                    sent=descriptive_text
                )
                questions.append(
                    [{"type": "image"}, {"type": "text", "text": question_template}]
                )
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append([{"type": "text", "text": answer_template}])
            out_pattern = [
                (
                    gt_pattern.random_pattern
                    if use_random_body
                    else gt_pattern.default_pattern
                )
            ]
        elif sample_type == SampleType.EDIT:
            # garment_editing
            if random.random() > self.editing_flip_prob:
                before_pattern = gt_pattern.default_pattern
                after_pattern = edited_pattern
                editing_text = editing_captions["editing_description_forward"]
            else:
                before_pattern = edited_pattern
                after_pattern = gt_pattern.default_pattern
                editing_text = editing_captions["editing_description_reverse"]

            before_pattern.name = "before_" + before_pattern.name
            after_pattern.name = "after_" + after_pattern.name
            out_pattern = [before_pattern, after_pattern]
            # questions and answers
            questions = []
            answers = []
            for i in range(1):
                question_template = random.choice(self.editing_question_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN, sent=editing_text
                )
                questions.append(
                    [{"type": "image"}, {"type": "text", "text": question_template}]
                )
                answer_template = random.choice(self.answer_list).format(
                    pattern=DEFAULT_PLACEHOLDER_TOKEN
                )
                answers.append([{"type": "text", "text": answer_template}])

        dialog = []
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
            int(sample_type),
        )

    def evaluate_patterns(
        self, pred_patterns: list[NNSewingPattern], gt_patterns: list[NNSewingPattern]
    ):
        return self.garment_tokenizer.evaluate_patterns(pred_patterns, gt_patterns)
