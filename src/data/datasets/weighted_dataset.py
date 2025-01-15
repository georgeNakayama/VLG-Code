import random

from torch.utils.data import Dataset

class BatchWeightedDataset(Dataset):
    def __init__(self, dataset_a, dataset_b, weight_a, weight_b, batch_size):
        self.dataset_a = dataset_a
        self.dataset_b = dataset_b
        self.weight_a = weight_a
        self.weight_b = weight_b
        self.batch_size = batch_size

        # Compute probabilities for batch-level selection
        self.prob_a = self.weight_a / (self.weight_a + self.weight_b)
        self.prob_b = self.weight_b / (self.weight_a + self.weight_b)

        # Store dataset lengths
        self.len_a = len(dataset_a)
        self.len_b = len(dataset_b)
        self.total_batches = (self.len_a // self.batch_size) + (self.len_b // self.batch_size)

    def __len__(self):
        return self.total_batches
    
    def __getattr__(self, name):
        if hasattr(self.dataset_a, name):
           return getattr(self.dataset_a, name)
        elif hasattr(self.dataset_b, name):
           return getattr(self.dataset_b, name)
        else:
           raise Exception("None of the datasets implements this function!")

    def __getitem__(self, idx):
        # Decide which dataset to sample from for the batch
        if random.random() < self.prob_a:
            dataset = self.dataset_a
            len_dataset = self.len_a
        else:
            dataset = self.dataset_b
            len_dataset = self.len_b

        # Calculate batch start index
        start_idx = (idx * self.batch_size) % len_dataset
        end_idx = start_idx + self.batch_size

        # Return batch from the chosen dataset
        return [dataset[i % len_dataset] for i in range(start_idx, end_idx)]