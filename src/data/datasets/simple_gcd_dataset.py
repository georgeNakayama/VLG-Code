import enum

import json
from pathlib import Path
import random
import typing

import cv2
import numpy as np
import torch

from data.patterns.gcd_pattern.pattern_converter import NNSewingPattern
from data.patterns.gcd_pattern.panel_classes import PanelClasses
from .utils import (
    ANSWER_LIST,
    DEFAULT_PLACEHOLDER_TOKEN,
    DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST,
    SHORT_QUESTION_LIST,
    SHORT_QUESTION_WITH_TEXT_LIST,
)


class SampleType(enum.Enum):
    IMAGE = 0
    TEXT = 1
    BOTH = 3

    def __int__(self):
        return self.value


class SimpleGarmentCodeData(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: Path | str,
        caption_dir: Path | str,
        split_file: Path | str,
        sampling_rate: list[int],
        panel_classification: str | None = None,
        split: typing.Literal["train", "val"] = "train",
        cache_gt: bool = True
    ):

        self.root_path = Path(root_dir)
        self.caption_dir = Path(caption_dir)
        assert self.root_path.exists()
        assert self.caption_dir.exists()

        self.sampling_rate = sampling_rate
        assert len(self.sampling_rate) == len(SampleType)

        self.panel_classification = panel_classification

        self.datapoints_names = []
        self.panel_classes = []

        split_file = Path(split_file)
        assert split_file.exists()
        self.datapoints_names = json.load(open(split_file, "r"))[split]

        self.panel_classifier = PanelClasses(classes_file=panel_classification)
        self.panel_classes = self.panel_classifier.classes

        self.split = split

        self.gt_cached = {}
        self.gt_caching = cache_gt

    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints_names)

    def _prepare_image(self, image_paths: list[Path | str]):
        """Fetch the image for the given index"""
        image_path = random.choice(image_paths)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_path

    def get_mode_names(self):
        return ["image", "text", "both"]

    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data.
        Does not support list indexing"""

        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        datapoint_name = self.datapoints_names[idx]

        # Get the rand_id
        data_name = datapoint_name.split("/")[-1]

        image_paths = [
            self.root_path/datapoint_name/f"{data_name}_render_back.png",
            self.root_path/datapoint_name/f"{data_name}_render_front.png"
        ]

        if not data_name in self.gt_cached:
            spec_file = self.root_path/datapoint_name/f"{data_name}_specification_shifted.json"

            gt_pattern = NNSewingPattern(
                spec_file,
                panel_classifier=self.panel_classifier,
                template_name=data_name,
            )
            gt_pattern.name = data_name

            caption_json: Path = self.caption_dir/data_name/f"captions.json"

            assert caption_json.exists()
            captions = json.load(open(caption_json, "r"))

            self.gt_cached[data_name] = (
                gt_pattern,
                captions,
            )

        gt_pattern, captions = self.gt_cached[data_name]

        sample_type = np.random.choice(list(SampleType), p=self.sampling_rate)

        if sample_type != SampleType.TEXT:
            image, image_path = self._prepare_image(image_paths)
        else:
            image = torch.zeros((3, 800, 800))
            image_path = ""

        if sample_type == SampleType.IMAGE:
            question_choices = SHORT_QUESTION_LIST
        elif sample_type == SampleType.TEXT:
            question_choices = DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST
        else:
            question_choices = SHORT_QUESTION_WITH_TEXT_LIST

        question_template = random.choice(question_choices)

        if sample_type != SampleType.IMAGE:
            text = captions["description"]
            question_template = question_template.format(sent=text)

        question = [{"type": "image"}, {"type": "text", "text": question_template}]
        
        answer_template = random.choice(ANSWER_LIST).format(
            pattern=DEFAULT_PLACEHOLDER_TOKEN
        )
        answer = [{"type": "text", "text": answer_template}] 

        dialog = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        prompt = [{"role": "user", "content": question}]
        out_pattern = [gt_pattern]

        return (
            image_path,
            image,
            dialog,
            prompt, 
            out_pattern,
            int(sample_type),
        )

    def evaluate_patterns(
        self, pred_patterns: list[NNSewingPattern], gt_patterns: list[NNSewingPattern]
    ):
        return self.garment_tokenizer.evaluate_patterns(pred_patterns, gt_patterns)
