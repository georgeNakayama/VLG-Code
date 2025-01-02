import glob
import os
import random
from typing import List
import cv2
import numpy as np
import torch
from .dataset_garmentcodedata_token_qva import GarmentCodeDatasetQVA
from .vqa_dataset import VQADataset

class HybridDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        garment_dataset: GarmentCodeDatasetQVA, 
        vqa_dataset: VQADataset,
        sample_rate: List[int]=[7, 3],
    ):
        self.garment_dataset = garment_dataset
        self.vqa_dataset = vqa_dataset
        self.all_datasets = [self.garment_dataset, self.vqa_dataset]
        self.sample_rate=sample_rate


    def __len__(self):
        return len (self.garment_dataset) + len(self.vqa_dataset)

    def __getitem__(self, idx):
        # get current cuda device
        ind = np.random.choice([0, 1], p=self.sample_rate)
        data = self.all_datasets[ind]
        inference = False
        return *data[0], inference


