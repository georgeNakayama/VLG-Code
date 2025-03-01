import json
import os
from pathlib import Path
import random
import tempfile

import cv2
import torch

from data.patterns.gcd_pattern.pattern_converter import NNSewingPattern
from data.patterns.gcd_pattern.panel_classes import PanelClasses
from data.datasets.utils import ANSWER_LIST, DEFAULT_PLACEHOLDER_TOKEN


class VLGBenchmark(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path: Path | str,
        panel_classification: str | None = None,
        split = None,
    ):

        data_path = Path(data_path)
        assert data_path.exists()

        self.data_points = [data_path / f for f in os.listdir(data_path) if os.path.isdir(data_path / f)]

        self.panel_classifier = PanelClasses(classes_file=panel_classification)


    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.data_points)

    def _prepare_image(self, image_path: str | Path):
        """Fetch the image for the given index"""
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image, image_path
    
    def _get_sewing_pattern(self, pattern: str, data_name: str) -> NNSewingPattern:
        """Expects the pattern as undecoded json."""

        with tempfile.NamedTemporaryFile(suffix=".json", delete=True) as temp_file:
            json.dumps(pattern, temp_file)  # Write JSON to the file

            gt_pattern = NNSewingPattern(
                temp_file.name,
                panel_classifier=self.panel_classifier,
                template_name=data_name,
            )
            gt_pattern.name = data_name
            return gt_pattern

    def get_mode_names(self):
        return ["benchmark"]

    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data.
        Does not support list indexing"""

        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()

        data_point_root = self.data_points[idx]
        input_spec_file = data_point_root / "input.json"
        assert input_spec_file.exists()
        input_spec = json.load(open(input_spec_file, "r"))


        # Get the rand_id
        gt_pattern = NNSewingPattern(
                data_point_root / input_spec["pattern"],
                panel_classifier=self.panel_classifier,
                template_name=input_spec["idx"],
            )
        gt_pattern.name = input_spec["idx"]

        if input_spec["image_path"]:
            image_path = data_point_root / input_spec["image_path"]
        else:
            image_path = None

        if image_path and image_path.exists():
            image, image_path = self._prepare_image(image_path)
        else:
            image = torch.zeros((3, 800, 800))
            image_path = ""

        question = [{"type": "image"}, {"type": "text", "text": input_spec["prompt"]}]
        
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
            0,
        )

    def evaluate_patterns(
        self, pred_patterns: list[NNSewingPattern], gt_patterns: list[NNSewingPattern]
    ):
        return self.garment_tokenizer.evaluate_patterns(pred_patterns, gt_patterns)
