# -*- coding: utf-8 -*-
'''
This script performs transformation on images and dumps the output arrays as npzfiles, which saves time for training.
'''

import argparse
import os
import torch
from PIL import Image
import base64
from io import BytesIO
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize, RandomResizedCrop
from tqdm import tqdm
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../Multimodal_Retrieval/", help="the directory which stores the image tsvfiles")
    parser.add_argument("--image_resolution", type=int, default=224, help="the resolution of transformed images, default to 224*224")
    return parser.parse_args()

def _convert_to_rgb(image):
    return image.convert('RGB')

def build_transform(resolution):
    normalize = Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    return Compose([
            Resize(resolution, interpolation=Image.BICUBIC),
            CenterCrop(resolution),
            _convert_to_rgb,
            ToTensor(),
            normalize,
        ])

if __name__ == "__main__":
    args = parse_args()
    transform = build_transform(args.image_resolution)
    train_path = os.path.join(args.data_dir, "train_imgs.tsv")
    val_path = os.path.join(args.data_dir, "valid_imgs.tsv")
    test_path = os.path.join(args.data_dir, "test_imgs.tsv")
    for path, split in zip((train_path, val_path, test_path), ("train", "valid", "test")):
        assert os.path.exists(path), "the {} filepath {} not exists!".format(split, path)
        print("begin to transform {} split".format(split))
        image_dict = {}
        with open(path, "r") as fin:
            for line in tqdm(fin):
                img_id, b64 = line.strip().split("\t")
                image = Image.open(BytesIO(base64.urlsafe_b64decode(b64)))
                image_array = transform(image).numpy()
                image_dict[img_id] = image_array
        output_path = "{}.{}.npz".format(path[:-4], args.image_resolution)
        np.savez(output_path, **image_dict)
        print("finished transforming {} images for {} split, the output is saved at {}".format(len(image_dict), split, output_path))
    print("done!")