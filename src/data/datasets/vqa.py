import json
import os

import cv2
import numpy as np
import torch
import torch.utils.data

class LLaVAInstruct(torch.utils.data.Dataset):  
    def __init__(
        self, 
        data_file_path: str,
        image_root_path: str,
        ): 

        self.data_file_path = data_file_path
        self.image_root_path = image_root_path
        #################################
        self.datapoints = json.load(open(self.data_file_path, "r"))

        self.gt_cached = {}
        self.gt_caching = True

    def __len__(self):
        """Number of entries in the dataset"""
        return len(self.datapoints)  
    
    def _parepare_image(self, image_path: str):
        """Fetch the image for the given index"""
        # assert os.path.exists(image_path), f"The path is not there {image_path}"
           # Check if the file exists
        if not os.path.exists(image_path):
            print(f"Image path does not exist: {image_path}")
            # Create a dummy image with COCO dimensions (640x480, RGB)
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            image[:, :] = [128, 128, 128]  # Set to a neutral gray color
        else:
            # Load the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image from path: {image_path}")
                # Create a dummy image with COCO dimensions
                image = np.zeros((480, 640, 3), dtype=np.uint8)
                image[:, :] = [128, 128, 128]  # Set to a neutral gray color
            else:
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return image
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    
    def _is_complete_datapoint(self, datapoint: dict):
        assert "id" in datapoint
        assert "image" in datapoint
        assert "conversations" in datapoint
    
    def _is_correct_conversation(self, conversation: list[dict]):
        assert len(conversation) % 2 == 0
        for i, conversation_part in enumerate(conversation):
            if i % 2 == 0:
                assert conversation_part["from"] == "human"
            else:
                assert not conversation_part["from"] == "human"
    
    # added from maria 
    def __getitem__(self, idx):
        """Called when indexing: read the corresponding data. 
        Does not support list indexing"""
        
        if torch.is_tensor(idx):  # allow indexing by tensors
            idx = idx.tolist()
        
        datapoint = self.datapoints[idx]
        self._is_complete_datapoint(datapoint)

        data_name = datapoint["id"]
        image_path = os.path.join(self.image_root_path, datapoint["image"])
        image = self._parepare_image(image_path)
        if data_name in self.gt_cached:
            conversation = self.gt_cached[data_name]
        else:
            conversation = datapoint["conversations"]
            self.gt_cached[data_name] = conversation
            
        # dialog = [{"role":"system","content":[{"type": "text", "text": system_prompt}]}]
        dialog = []
        # question = [{"role":"system","content":[{"type": "text", "text": system_prompt}]}]
        question = []

        self._is_correct_conversation(conversation)
        for conversation_part in conversation[:-2]:
            if conversation_part["from"] == "human":
                dialog.append({"role": "user", "content": conversation_part["value"]})
            else:
                dialog.append({"role": "assistant", "content": conversation_part["value"]})

        question.append({"role": "user", "content": f"<image> {conversation[-2]['value']}"})
        ground_truth = [conversation[-1]["value"]]

        return (
            image_path,
            image,
            dialog,
            question,
            ground_truth,
            -1
        ) 
