from torchvision.datasets import CelebA
from torchvision import transforms
from sklearn.metrics import f1_score, accuracy_score
import torch
from torch.utils.data import random_split

from torchvision import models
from torch import nn
import os

from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

import argparse
import pickle
import json

import pandas as pd
import numpy as np
from models import ResNetClassifier

from divexp import *

if __name__ == "__main__":
    ds = CelebA(root='./data', split='all', download=True, transform=transforms.ToTensor())

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--device", type=str, default="cuda")
    argParser.add_argument("--checkpoint", type=str)
    argParser.add_argument("--minsup", type=float, default=0.01)

    args = argParser.parse_args()

    ckpt_dir = "models-ckpt"
    pt_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pt")
    json_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.json")
    ds_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pkl")

    model = torch.load(pt_filename)
    model.eval()

    with open(json_filename, "r") as f:
        config = json.load(f)
    
    with open(ds_filename, "rb") as f:
        ds = pickle.load(f)
        train_set, test_set, test_sets = ds["train"], ds["test"], ds["test_chunks"]
    
    attr_id = config["attr_id"]
    attr_names = train_set.dataset.attr_names
    col_names = np.array([ attr.strip() for attr in attr_names if attr.strip() ])
    metadata_mask = np.arange(len(col_names)) != attr_id
    metadata_names = col_names[metadata_mask]

    df_train = pd.DataFrame(data=train_set.dataset.attr[np.array(train_set.indices)][:, metadata_mask], columns=metadata_names)
    matches = compute_matches(df_train.astype(bool), minsup=args.minsup, n_proc=36)

    matches_ts_list = []
    df_tests = []
    for ts in tqdm(test_sets):
        md = ts.dataset.dataset.attr[np.array(ts.dataset.indices)][np.array(ts.indices)][:, metadata_mask].numpy()
        df_test = pd.DataFrame(md.astype(bool), columns=metadata_names)
        matches_ts = compute_matches(df_test, fi=matches.fi)
        matches_ts_list.append(matches_ts)
        df_tests.append(df_test)
    
    # save pickle
    ckpt_dir = "models-ckpt"
    matches_filename = os.path.join(ckpt_dir, f"matches-{args.checkpoint}.pkl")

    with open(matches_filename, "wb") as f:
        pickle.dump({
            "matches_train": matches,
            "matches_batches": matches_ts_list,
            "metadata_train": df_train,
            "metadata_batches": df_tests,
            "col_names": col_names,
            "metadata_mask": metadata_mask,
            "metadata_names": metadata_names,
        }, f)