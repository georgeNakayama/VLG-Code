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
        # Apply random zoom to the image if it's not a zero tensor (TEXT or REASON types)
        # if not isinstance(image, torch.Tensor):
                
        #     # Random zoom factor between 0.8 (zoom out) and 1.2 (zoom in)
        #     zoom_factor = random.uniform(0.5, 2)
            
        #     # Calculate new size while maintaining aspect ratio
        #     h, w = image.shape[:2] 
        #     new_h = int(h * zoom_factor)
        #     new_w = int(w * zoom_factor)
            
        #     # Resize image using cv2
        #     image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
        #     # Center crop or pad to original size
        #     if zoom_factor > 1:  # Zoomed in - need to crop
        #         start_h = (new_h - h) // 2
        #         start_w = (new_w - w) // 2
        #         image = image[start_h:start_h+h, start_w:start_w+w]
        #     else:  # Zoomed out - need to pad
        #         pad_h = (h - new_h) // 2
        #         pad_w = (w - new_w) // 2
        #         pad_h_top = pad_h
        #         pad_h_bottom = pad_h if (h-new_h)%2==0 else pad_h+1
        #         pad_w_left = pad_w  
        #         pad_w_right = pad_w if (w-new_w)%2==0 else pad_w+1
                
        #         image = np.pad(
        #             image,
        #             ((pad_h_top, pad_h_bottom), 
        #              (pad_w_left, pad_w_right),
        #              (0, 0)),
        #             mode='constant',
        #             constant_values=0
        #         )
        # Apply random translation to the image
        # if not isinstance(image, torch.Tensor):
        #     h, w = image.shape[:2]
            
        #     # Random translation in x and y direction (-20% to +20% of image size)
        #     tx = int(random.uniform(-0.4, 0.4) * w)
        #     ty = int(random.uniform(-0.4, 0.4) * h)
            
        #     # Create translation matrix
        #     translation_matrix = np.float32([[1, 0, tx], [0, 1, ty]])
            
        #     # Apply translation
        #     translated = cv2.warpAffine(image, translation_matrix, (w, h))
            
        #     # Create white background
        #     white_background = np.ones_like(image) * 255
            
        #     # Copy valid parts of translated image onto white background
        #     image = white_background.copy()
            
        #     # Calculate valid regions after translation
        #     y_start = max(0, -ty)
        #     y_end = min(h, h - ty)
        #     x_start = max(0, -tx) 
        #     x_end = min(w, w - tx)
            
        #     # Source regions in translated image
        #     src_y_start = max(0, ty)
        #     src_y_end = min(h, h + ty)
        #     src_x_start = max(0, tx)
        #     src_x_end = min(w, w + tx)
            
        #     # Copy valid region
        #     image[y_start:y_end, x_start:x_end] = translated[src_y_start:src_y_end, src_x_start:src_x_end]
        # Create a random bright background color (ensuring brightness > 128)
        # if not isinstance(image, torch.Tensor):
        #     random_color = [
        #         random.randint(128, 255),  # R
        #         random.randint(128, 255),  # G 
        #         random.randint(128, 255)   # B
        #     ]
        #     # Only change white pixels to random color
        #     white_mask = np.all(image == 255, axis=-1)
        #     image[white_mask] = random_color

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
