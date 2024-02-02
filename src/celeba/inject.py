"""
This file takes as input a checkpoint for which a model exists (models.py), and the subgroups (M) for the training set (precompute.py).
It produces a test session (i.e. sequence of batches) where one randomly chosen subgorup is injected with noise.

This is repeated N times (n-targets) for each of a pool of supports (see "supports"). 

"""

from sklearn.metrics import f1_score, accuracy_score

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


import os
from tqdm import tqdm

import torch

import argparse
import pickle
import json

import pandas as pd
import numpy as np

import sys
sys.path.append("..")
from divexp import *
from config import *

import random

def closest_odd(n):
    n = int(n)
    if n % 2 == 0:
        return n + 1
    else:
        return n

def parser():

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--checkpoint", type=str)
    argParser.add_argument("--n-targets", type=int, default=10)
    
    # parameters for "injection"
    argParser.add_argument("--start-noise", type=int, default=10) # add noise after 10 batches
    argParser.add_argument("--transitory", type=int, default=10) # duration of the transitory
    argParser.add_argument("--frac-noise", type=float, default=1.0) # % of points to add noise to (transitory from 0 to this)d
    
    argParser.add_argument("--metric", type=str, choices=["accuracy"], default="accuracy")
    argParser.add_argument("--seed", type=int, default=42)
    argParser.add_argument("--device", type=str, default="cuda")

    args = argParser.parse_args()
    return args

def mock_parser():
    
        class MockArgs:
            def __init__(self):
                self.checkpoint = "resnet50"
                self.n_targets = 10
                self.start_noise = 10
                self.transitory = 10
                self.frac_noise = 1.0
                self.metric = "accuracy"
                self.seed = 42
                self.device = "cuda"
        
        args = MockArgs()
        return args

if __name__ == "__main__":

    args = parser()


    # model_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pt")
    pt_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pt")
    ds_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pkl")
    matches_filename = os.path.join(ckpt_dir, f"matches-{args.checkpoint}.pkl")
    json_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.json")

    with open(json_filename, "r") as f:
        config = json.load(f)

    model = torch.load(pt_filename).to(args.device)
    
    with open(ds_filename, "rb") as f:
        ds = pickle.load(f)
        train_set, test_set, _ = ds["train"], ds["test"], ds["test_chunks"]
    
    with open(matches_filename, "rb") as f:
        matches_obj = pickle.load(f)
        df_train = matches_obj["metadata_train"]
        df_tests = matches_obj["metadata_batches"]
        matches = matches_obj["matches_train"]
        matches_ts_list = matches_obj["matches_batches"]
        col_names = matches_obj["col_names"]
        metadata_mask = matches_obj["metadata_mask"]
        metadata_names = matches_obj["metadata_names"]
    attr_id = config["attr_id"]


    sampling = "uniform"
    n_batches = 30

    if sampling == "uniform":
        # sample from all possible supports (0.01 => 1.0 with 20 buckets)
        vmin = 0.01
        vmax = 1.0
        n_buckets = 20

        supports = np.logspace(np.log10(vmin), np.log10(vmax), n_buckets)

        nums = {
            (supports[i], supports[i+1]): args.n_targets for i in range(len(supports)-1)
        }
    elif sampling == "precomputed":
        nums = {(0.3793, 0.4833): 13,
                (0.0336, 0.0428): 1,
                (0.0264, 0.0336): 3,
                (0.0162, 0.0207): 1,
                (0.4833, 0.6158): 20,
                (0.2336, 0.2976): 2,
                (0.1438, 0.1833): 1,
                (0.01, 0.0127): 0,
                (0.0695, 0.0886): 2,
                (0.1129, 0.1438): 4,
                (0.0207, 0.0264): 2,
                (0.2976, 0.3793): 7,
                (0.0886, 0.1129): 5,
                (0.0428, 0.0546): 4,
                (0.6158, 0.7848): 31,
                (0.1833, 0.2336): 6,
                (0.0127, 0.0162): 3,
                (0.7848, 1.0): 33,
                (0.0546, 0.0695): 2}

    all_subgroups = []
    for (from_sup, to_sup), count in nums.items():
        valid_subgrp = matches.fi[(matches.fi.support >= from_sup) & (matches.fi.support < to_sup)]
        print(from_sup, "to", to_sup, "=>", len(valid_subgrp))
        all_subgroups.extend([ (fname, from_sup, to_sup) for fname in valid_subgrp.itemsets.sample(n=min(len(valid_subgrp),count)) ])
    random.shuffle(all_subgroups)

    with tqdm(all_subgroups) as pbar:
        for target_sg, from_sup, to_sup in pbar:

            outfile = os.path.join(ckpt_dir, "sup-wise", f"{args.checkpoint}-noise-{args.frac_noise:.2f}-support-{from_sup:.4f}-{to_sup:.4f}-target-{'-'.join(list(map(str,target_sg)))}.pkl")
            if os.path.exists(outfile):
                print("Skipping", outfile)
                continue
                
            pbar.set_description(f"Support {from_sup:.4f} to {to_sup:.4f} -- {target_sg}")

            accuracies = []
            f1 = []
            y_trues = []
            y_preds = []
            divs = []
            altered = []
            matches_ts_list = []
            
            test_sets = random_split(test_set, [1/n_batches] * n_batches)

            # noise!
            blurs = [None] * args.start_noise + [
                transforms.GaussianBlur(closest_odd(i), sigma=i) if i > 0 else lambda x : x # identity if sigma=0
                for i in np.linspace(1, 40, n_batches - args.start_noise)
            ]
            for ts, blur in zip(test_sets, blurs):
                # compute matches
                md = ts.dataset.dataset.attr[np.array(ts.dataset.indices)][np.array(ts.indices)][:, metadata_mask].numpy()
                df_test = pd.DataFrame(md.astype(bool), columns=metadata_names)
                matches_ts = compute_matches(df_test, fi=matches.fi)
                matches_ts_list.append(matches_ts)

                in_target = (md[:, list(target_sg)] == 1).all(axis=1).nonzero()[0]
                mask = torch.zeros(len(md)).bool()
                picked = np.random.choice(in_target, size=int(round(len(in_target) * args.frac_noise)), replace=False)
                if blur is not None:
                    mask[picked] = True

                x, attr = next(iter(DataLoader(ts, shuffle=False, batch_size=len(ts), num_workers=4)))
                y_true = attr[:, attr_id]

                if mask.sum():
                    # only > 0 if blur != None
                    x[mask] = blur(x[mask])
                print("Affected", mask.sum(), "out of", len(mask))

                y_pred = (model(x.to(args.device))>0).cpu().numpy().flatten()
                
                altered.append(mask)

                y_trues.append(y_true)
                y_preds.append(y_pred)

                divs.append(div_explorer(Matches(matches=matches_ts.matches.astype(int), fi=matches.fi), y_true, y_pred, [args.metric]))

                accuracies.append(accuracy_score(y_true, y_pred))
                f1.append(f1_score(y_true, y_pred))
        
            # store results
            with open(outfile, "wb") as f:
                pickle.dump({
                    "subgroup": target_sg,
                    "batches": test_sets,
                    "accuracies": accuracies,
                    "f1": f1,
                    "divs": divs,
                    "y_trues": y_trues,
                    "y_preds": y_preds,
                    "blurs": blurs,
                    "altered": altered,
                    "matches_batches": matches_ts_list
                }, f)
