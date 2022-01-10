import os
import logging
import json
from PIL import Image
import base64
from io import BytesIO
from dataclasses import dataclass

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torch.utils.data.distributed import DistributedSampler

from clip import _tokenizer
from clip.clip import tokenize

class JsonlDataset(Dataset):
    def __init__(self, jsonl_filename, img_filename, split="val", max_txt_length=24):
        assert os.path.exists(jsonl_filename), "The annotation datafile {} not exists!".format(jsonl_filename)
        assert os.path.exists(img_filename), "The image npz datafile {} not exists!".format(img_filename)
        
        logging.debug(f'Loading jsonl data from {jsonl_filename}.')
        self.samples = []
        with open(jsonl_filename, "r") as fin:
            for line in fin:
                obj = json.loads(line.strip())
                query_id = obj['query_id']
                query = obj['query_text']
                for target in obj['item_ids']:
                    self.samples.append((query_id, query, target))
        logging.debug(f'Finished loading jsonl data from {jsonl_filename}.')
        
        logging.debug(f'Loading image npzfile from {img_filename}.')
        self.imgs = np.load(img_filename, "r")
        logging.debug(f'Finished loading image npzfile from {img_filename}.')

        self.split = split
        self.max_txt_length = max_txt_length

    def _read_img_tensor_from_npzfile(self, img_id):
        img_array = self.imgs[str(img_id)]
        return torch.from_numpy(img_array)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        query_id, query, img_id = self.samples[idx]
        image = self._read_img_tensor_from_npzfile(img_id)
        text = tokenize([str(query)], context_length=self.max_txt_length)[0]
        eos_index = text.numpy().tolist().index(_tokenizer.vocab['[SEP]'])
        return image, text, eos_index


@dataclass
class DataInfo:
    dataloader: DataLoader
    sampler: DistributedSampler


def get_dataset(args, is_train, max_txt_length=24):
    input_filename = args.train_data if is_train else args.val_data
    img_filename = args.train_img if is_train else args.val_img
    dataset = JsonlDataset(
        input_filename,
        img_filename,
        split="train" if is_train else "val",
        max_txt_length=max_txt_length)
    num_samples = len(dataset)
    sampler = DistributedSampler(dataset) if args.distributed and is_train else None
    shuffle = is_train and sampler is None

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        sampler=sampler,
        drop_last=is_train,
    )
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)

    return DataInfo(dataloader, sampler)
    

def get_data(args, max_txt_length=24):
    data = {}

    if args.train_data:
        data["train"] = get_dataset(
            args, is_train=True, max_txt_length=max_txt_length)
    if args.val_data:
        data["val"] = get_dataset(
            args, is_train=False, max_txt_length=max_txt_length)

    return data
