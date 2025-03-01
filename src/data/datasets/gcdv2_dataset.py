import enum
import io
import json
import random
import typing
import tarfile

import numpy as np
from PIL import Image
import pyarrow.dataset as ds
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


class SimpleGarmentCodeData(torch.utils.data.Dataset):
    def __init__(
        self,
        tar_file: str,         # Local tar file (webdataset) containing images
        parquet_file: str,     # Path to the parquet file containing JSON data
        split_file: str,       # Local JSON file containing the datapoint names per split
        sampling_rate: list[int],
        panel_classification: str | None = None,
        split: typing.Literal["train", "val"] = "train",
        cache_gt: bool = True
    ):
        # Open the tar file for images.
        self.tar = tarfile.open(tar_file, "r:*")
        # Build an index mapping file keys to tarfile member objects.
        self.tar_members = {member.name: member for member in self.tar.getmembers()}

        # Open the parquet file as a PyArrow dataset (lazy loading).
        self.dataset = ds.dataset(parquet_file, format="parquet")

        # Load datapoints names from the provided split file.
        with open(split_file, "r") as f:
            splits = json.load(f)
        self.datapoints_names = splits[split]

        self.sampling_rate = sampling_rate
        assert len(self.sampling_rate) == len(SampleType)

        self.panel_classifier = PanelClasses(classes_file=panel_classification)
        self.split = split

        # Cache for already loaded ground-truth data.
        self.gt_cached = {}
        self.gt_caching = cache_gt

    def __len__(self):
        """Return the number of entries in the dataset."""
        return len(self.datapoints_names)

    def _prepare_image(self, image_keys: list[str]):
        """
        Randomly pick one of the candidate image keys (paths in the tar file),
        extract it, and decode it into an image.
        """
        key = random.choice(image_keys)
        if key not in self.tar_members:
            raise Exception(f"Key {key} not found in tar file")
        member = self.tar_members[key]
        fileobj = self.tar.extractfile(member)
        if fileobj is None:
            raise Exception(f"Failed to extract file for key {key}")
        data = fileobj.read()
        image = Image.open(io.BytesIO(data))
        image = np.array(image)
        return image, key

    def get_mode_names(self):
        return ["image", "text", "both", "synth", "reason", "edit"]

    def __getitem__(self, idx):
        """
        Retrieve the data sample corresponding to the given index.
        This includes extracting the image from the local tar file and
        lazily fetching the JSON data (specification and captions) from the parquet file.
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datapoint_name = self.datapoints_names[idx]

        # Build keys corresponding to the images in the tar file.
        image_keys = [
            f"{datapoint_name}.png",
            f"{datapoint_name}.png"
        ]

        # Retrieve ground truth data lazily using PyArrow if not cached.
        if datapoint_name not in self.gt_cached:
            # Filter the dataset for the row with the matching "data_name".
            filtered_table = self.dataset.to_table(filter=(ds.field("idx") == datapoint_name))
            if filtered_table.num_rows == 0:
                raise ValueError(f"Data for {datapoint_name} not found in parquet file")
            # Convert the first (and assumed only) row to a dict.
            row = {col: filtered_table.column(col)[0].as_py() for col in filtered_table.column_names}

            pattern = row["pattern"]
            text = row["text"]

            gt_pattern = NNSewingPattern(
                pattern,
                panel_classifier=self.panel_classifier,
                template_name=datapoint_name,
            )
            gt_pattern.name = datapoint_name

            self.gt_cached[datapoint_name] = (gt_pattern, text)

        gt_pattern, text = self.gt_cached[datapoint_name]

        # Randomly select a sample type (IMAGE, TEXT, BOTH) using the given sampling rate.
        sample_type = np.random.choice(list(SampleType), p=self.sampling_rate)

        if sample_type != SampleType.TEXT:
            image, image_key = self._prepare_image(image_keys)
        else:
            image = torch.zeros((3, 800, 800))
            image_key = ""

        # Choose a question template based on the sample type.
        if sample_type == SampleType.IMAGE:
            question_choices = SHORT_QUESTION_LIST
        elif sample_type == SampleType.TEXT:
            question_choices = DESCRIPTIVE_TEXT_SHORT_QUESTION_LIST
        else:
            question_choices = SHORT_QUESTION_WITH_TEXT_LIST

        question_template = random.choice(question_choices)
        if sample_type != SampleType.IMAGE:
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
            image_key,
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
    tar_file = "/projects/m000051/data/vlg/gcdv2-images.tar"           # Local tar file containing images.
    parquet_file = "/projects/m000051/data/vlg/gcdv2-labels.parquet"   # Parquet file with JSON metadata.
    split_file = "/projects/m000051/data/vlg/gcdv2-datasplit.json"     # JSON file specifying train/val splits.
    panel_classification = "/scratch/m000051/garment_gang/AIpparel-Code/assets/panel_classes_garmentcodedata.json"     # Panel classification file (if needed).

    # Define a sampling rate (e.g., equal probability for IMAGE, TEXT, BOTH).
    sampling_rate = [0.33, 0.33, 0.34]

    # Instantiate the dataset.
    dataset = SimpleGarmentCodeData(
        tar_file=tar_file,
        parquet_file=parquet_file,
        split_file=split_file,
        sampling_rate=sampling_rate,
        panel_classification=panel_classification,
        split="train",
        cache_gt=True
    )

    print("Dataset length:", len(dataset))

    # Sample an entry (e.g., the first one).
    sample = dataset[0]
    image_key, image, dialog, prompt, out_pattern, sample_type = sample

    print("Image key:", image_key)
    print("Image shape:", image.shape if image is not None else None)
    print("Dialog:", dialog)
    print("Prompt:", prompt)
    print("Output pattern:", out_pattern)
    print("Sample type:", sample_type)
