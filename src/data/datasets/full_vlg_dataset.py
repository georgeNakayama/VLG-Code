import enum

import json
from pathlib import Path
import random
import typing

import cv2
import numpy as np
import pandas as pd
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
    BOTH = 2
    SYNTH = 3
    REASON = 4
    EDIT = 5

    def __int__(self):
        return self.value


class FullDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: Path | str,
        caption_dir: Path | str,
        editing_dir: Path | str,
        split_file: Path | str,
        synth_root_dir: Path | str,
        sampling_rate: list[int],
        panel_classification: str | None = None,
        edit_prompts_file: str | None = None,
        extra_parquet_file: str | None = None,
        split: typing.Literal["train", "val"] = "train",
        cache_gt: bool = True,
    ):

        self.root_path = Path(root_dir)
        self.caption_dir = Path(caption_dir)
        self.editing_dir = Path(editing_dir)
        self.synth_root = Path(synth_root_dir)
        assert self.root_path.exists()
        assert self.caption_dir.exists()
        assert self.editing_dir.exists()
        assert self.synth_root.exists()

        self.sampling_rate = sampling_rate
        assert len(self.sampling_rate) == len(SampleType)

        self.datapoints_names = []

        split_file = Path(split_file)
        assert split_file.exists()
        self.datapoints_names = json.load(open(split_file, "r"))[split]

        self.panel_classifier = PanelClasses(classes_file=panel_classification)
        self.edit_prompts = json.load(open(edit_prompts_file, "r"))

        self.extra = pd.read_parquet(extra_parquet_file)
        self.extra = self.extra.set_index("idx")
        # Filter datapoints to only keep those with populated image_path in extra data
        self.datapoints_names = [
            dp for dp in self.datapoints_names
            if isinstance(self.extra.loc[Path(dp).name, "synth_paths"], (list, np.ndarray))
            and not pd.isna(self.extra.loc[Path(dp).name, "synth_paths"]).all()
        ]

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
        return ["image", "text", "both", "synth", "reason", "edit"]

    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data.
        Does not support list indexing"""

        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        datapoint_name = self.datapoints_names[idx]

        # Get the rand_id
        data_name = datapoint_name.split("/")[-1]

        image_paths = [
            self.root_path / datapoint_name / f"{data_name}_render_back.png",
            self.root_path / datapoint_name / f"{data_name}_render_front.png",
        ]

        if not data_name in self.gt_cached:
            spec_file = (
                self.root_path
                / datapoint_name
                / f"{data_name}_specification_shifted.json"
            )

            gt_pattern = NNSewingPattern(
                spec_file,
                panel_classifier=self.panel_classifier,
                template_name=data_name,
            )
            gt_pattern.name = data_name
            editing_spec_file = (
                self.editing_dir / data_name / f"edited_specification.json"
            )
            if not editing_spec_file.exists():
                edited_pattern = None
            else:
                edited_pattern = NNSewingPattern(
                    editing_spec_file,
                    panel_classifier=self.panel_classifier,
                    template_name=data_name,
                )
                edited_pattern.name = data_name

            caption_json: Path = self.caption_dir / data_name / f"captions.json"

            assert caption_json.exists()
            captions = json.load(open(caption_json, "r"))

            self.gt_cached[data_name] = (
                gt_pattern,
                edited_pattern,
                captions,
            )

        gt_pattern, edited_pattern, captions = self.gt_cached[data_name]

        extra_modalities = self.extra.loc[data_name]

        sample_type = np.random.choice(list(SampleType), p=self.sampling_rate)

        if sample_type == SampleType.REASON:
            if extra_modalities["reasoning"] is None:
                sample_type = SampleType.TEXT

        if sample_type == SampleType.SYNTH:
            if extra_modalities["synth_paths"] is None:
                sample_type = SampleType.IMAGE

        if sample_type == SampleType.EDIT:
            if (
                extra_modalities["editing_instructions"] is None
                or extra_modalities["editing_instructions"] not in self.edit_prompts
                or edited_pattern is None
            ):
                sample_type = SampleType.IMAGE
        
        if sample_type == SampleType.IMAGE:
            image, image_path = self._prepare_image(image_paths)
            question_template = random.choice(SHORT_QUESTION_LIST)
        elif sample_type == SampleType.TEXT:
            image = torch.zeros((3, 800, 800))
            image_path = ""
            question_template = random.choice(DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST)
            question_template = question_template.format(sent=captions["description"])
        elif sample_type == SampleType.BOTH:
            image, image_path = self._prepare_image(image_paths)
            question_template = random.choice(SHORT_QUESTION_WITH_TEXT_LIST)
            question_template = question_template.format(sent=captions["description"])
        elif sample_type == SampleType.REASON:
            image = torch.zeros((3, 800, 800))
            image_path = ""
            question_template = random.choice(extra_modalities["reasoning"])
        elif sample_type == SampleType.SYNTH:
            image_paths = [self.synth_root / file for file in extra_modalities["synth_paths"]]
            image, image_path = self._prepare_image(image_paths)
            question_template = random.choice(SHORT_QUESTION_LIST)
        elif sample_type == SampleType.EDIT:
            image, image_path = self._prepare_image(image_paths)
            question_template = random.choice(extra_modalities["editing_instructions"])

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
        out_pattern = (
            [gt_pattern] if not sample_type == SampleType.EDIT else [edited_pattern]
        )

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


if __name__ == "__main__":
    # Define paths to your existing dataset files.
    root_dir = "/scratch/m000051/garment_gang/data/garmentcodedatav2"
    caption_dir = "/scratch/m000051/garment_gang/data/long-caption-processed"
    editing_dir = "/projects/m000051/data/vlg/gcd_editing"
    split_file = "../assets/garmentcodedatav2_datasplit.json"
    synth_root_dir = "/scratch/m000051/garment_gang/data/synth_images"
    panel_classification = "../assets/panel_classes_garmentcodedata.json"
    edit_prompts_file = "../assets/editing.json"
    extra_parquet_file = "/scratch/m000051/garment_gang/data/extras.parquet"

    # Define sampling rate for different modes
    sampling_rate = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0]

    # Instantiate the dataset
    dataset = FullDataset(
        root_dir=root_dir,
        caption_dir=caption_dir, 
        editing_dir=editing_dir,
        split_file=split_file,
        synth_root_dir=synth_root_dir,
        panel_classification=panel_classification,
        edit_prompts_file=edit_prompts_file,
        extra_parquet_file=extra_parquet_file,
        sampling_rate=sampling_rate,
        split="train"
    )

    print("Dataset length:", len(dataset))

    import tqdm
    # Sample an entry (e.g., the first one).
    for i in tqdm.tqdm(range(10000)):
        sample = dataset[i]
        image_key, image, dialog, prompt, out_pattern, sample_type = sample

    print("Image key:", image_key)
    print("Image shape:", image.shape if image is not None else None)
    print("Dialog:", dialog)
    print("Prompt:", prompt)
    print("Output pattern:", out_pattern)
    print("Sample type:", sample_type)
