from typing import Optional
import torch
import albumentations as A
from abc import abstractmethod
import cv2
import pandas as pd
import os
from pytorch_lightning import LightningDataModule
from transformers import AutoTokenizer
from torch.utils.data import random_split, DataLoader

class ImageRetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, artifact_dir: str, tokenizer=None, target_size: Optional[int] = None, max_length: int = 200, lazy_loading: bool = False):
        super().__init__()
        self.artifact_dir = artifact_dir
        self.target_size = target_size
        self.image_files, self.captions = self.fetch_dataset()
        self.lazy_loading = lazy_loading
        self.images = self.image_files if lazy_loading else [cv2.imread(f) for f in self.image_files]
        self.tokenizer = tokenizer
        self.tokenized_captions = tokenizer(list(self.captions), padding=True, truncation=True, max_length=max_length, return_tensors="pt")
        self.transforms = A.Compose([
            A.Resize(target_size, target_size, always_apply=True),
            A.Normalize(max_pixel_value=255.0, always_apply=True),
        ])

    @abstractmethod
    def fetch_dataset(self):
        pass

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, index):
        item = {key: values[index] for key, values in self.tokenized_captions.items()}
        image = cv2.imread(self.image_files[index])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = self.transforms(image=image)["image"]
        item["image"] = torch.tensor(image).permute(2, 0, 1).float()
        item["caption"] = self.captions[index]
        return item

class Flickr8kDataset(ImageRetrievalDataset):
    def fetch_dataset(self):
        annotations = pd.read_csv(os.path.join(self.artifact_dir, "captions.txt"))
        image_files = [os.path.join(self.artifact_dir, "Images", img) for img in annotations["image"]]
        captions = annotations["caption"].tolist()
        return image_files, captions

class ImageRetrievalDataModule(LightningDataModule):
    def __init__(self, artifact_dir, dataset_name, val_split=0.2, tokenizer_alias=None, target_size=224, max_length=100, lazy_loading=False, train_batch_size=16, val_batch_size=16, num_workers=4):
        super().__init__()
        self.artifact_dir = artifact_dir
        self.dataset_name = dataset_name
        self.val_split = val_split
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_alias)
        self.target_size = target_size
        self.max_length = max_length
        self.lazy_loading = lazy_loading
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.num_workers = num_workers

    @staticmethod
    def split_data(dataset: ImageRetrievalDataset, val_split: float):
        train_length = int((1 - val_split) * len(dataset))
        val_length = len(dataset) - train_length
        return random_split(dataset, [train_length, val_length])

    def setup(self, stage: Optional[str] = None):
        dataset = Flickr8kDataset(self.artifact_dir, self.tokenizer, self.target_size, self.max_length, self.lazy_loading)
        self.train_dataset, self.val_dataset = self.split_data(dataset, self.val_split)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.train_batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers > 0)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers, persistent_workers=self.num_workers > 0)
