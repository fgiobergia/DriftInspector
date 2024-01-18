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
import matplotlib.pyplot as plt
from torch.utils.data import TensorDataset

def closest_odd(n):
    n = int(n)
    if n % 2 == 0:
        return n + 1
    else:
        return n

if __name__ == "__main__":
    ds = CelebA(root='./data', split='all', download=True, transform=transforms.ToTensor())

    argParser = argparse.ArgumentParser()
    argParser.add_argument("--device", type=str, default="cuda")
    argParser.add_argument("--checkpoint", type=str)
    argParser.add_argument("--n-targets", type=int, default=100)
    argParser.add_argument("--start-noise", type=int, default=10) # add noise after 10 batches
    argParser.add_argument("--frac-noise", type=float, default=1.0) # % of points to add noise to
    argParser.add_argument("--batch-size", type=int, default=1024) # % of points to add noise to
    argParser.add_argument("--metric", type=str, choices=["accuracy"], default="accuracy") # add noise after 10 batches

    args = argParser.parse_args()

    ckpt_dir = "models-ckpt"
    pt_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pt")
    json_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.json")
    ds_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pkl")
    matches_filename = os.path.join(ckpt_dir, f"matches-{args.checkpoint}.pkl")

    model = torch.load(pt_filename).to(args.device)
    model.eval()

    with open(json_filename, "r") as f:
        config = json.load(f)
    
    with open(ds_filename, "rb") as f:
        ds = pickle.load(f)
        train_set, test_set, test_sets = ds["train"], ds["test"], ds["test_chunks"]
    
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
    output_dir = os.path.join(ckpt_dir, f"{args.checkpoint}-{args.metric}-noise-{args.frac_noise:.2f}")
    os.makedirs(output_dir, exist_ok=True)

    for target_sg in matches.fi.itemsets.sample(n=args.n_targets):
        sg = tuple(target_sg)
        outfile = os.path.join(output_dir, f"target-{'-'.join(map(str, sorted(sg)))}.pkl")
        print("Target", sg, "output", outfile)

        n_batches = len(test_sets)
        blurs = [None] * args.start_noise + [
            transforms.GaussianBlur(closest_odd(i), sigma=i) if i > 0 else lambda x : x # identity if sigma=0
            for i in np.linspace(1, 40, n_batches - args.start_noise)
        ]

        accuracies = []
        f1 = []
        y_trues = []
        y_preds = []
        divs = []

        samples = [None] * len(test_sets)

        for pos, (ts, blur, matches_ts) in enumerate(zip(test_sets, blurs, matches_ts_list)):
            # get predictions
            model.eval()
            
            with torch.no_grad():
                y_true = []
                y_pred = []
                metadata = []

                dl = DataLoader(ts, batch_size=args.batch_size, shuffle=False, num_workers=4)
                md = np.vstack([ y[:, metadata_mask].numpy() for _, y in dl ])
                in_target = (md[:, sg] == 1).all(axis=1).nonzero()[0]
                mask = torch.zeros(len(md)).bool()
                picked = np.random.choice(in_target, size=int(round(len(in_target) * args.frac_noise)), replace=False)
                mask[picked] = True
                dl_mask = DataLoader(TensorDataset(mask), batch_size=args.batch_size, shuffle=False, num_workers=4)
                tot_target = len(in_target)


                num_altered = 0
                for (x, y), (mask_batch,) in tqdm(zip(dl, dl_mask), total=len(dl)):
                    md = y[:, metadata_mask].numpy().astype(int)

                    # apply blur, if required
                    # if blur is not None:
                    #     # only add noise to points in target subgroups
                    #     # mask = (md[:, sg] == 1).all(axis=1)
                    #     if mask.any():
                    #         num_altered += mask.sum()
                    #         x[mask] = blur(x[mask])
                    if blur is not None and mask_batch.any():
                        num_altered += mask_batch.sum().item()
                        x[mask_batch] = blur(x[mask_batch])
                    
                    if mask_batch.any() and samples[pos] is None:
                        samples[pos] = x[mask_batch][0].cpu().numpy().transpose(1, 2, 0)

                    out = model(x.to(args.device))
                    preds = (out > 0).cpu().numpy().astype(int)
                    y_pred.extend(preds)
                    y_true.extend(y[:, attr_id].numpy().astype(int))
                    metadata.extend(md)
                
                y_true = np.hstack(y_true)
                y_pred = np.hstack(y_pred)
                metadata = np.vstack(metadata)

                y_trues.append(y_true)
                y_preds.append(y_pred)

                divs.append(div_explorer(matches_ts, y_true, y_pred, [args.metric]))

                accuracies.append(accuracy_score(y_true, y_pred))
                f1.append(f1_score(y_true, y_pred))
                print("Altered", num_altered, "out of", tot_target, "accuracy", accuracies[-1], "f1", f1[-1])
        
        # store results

        fig, ax = plt.subplots(1, len(samples), figsize=(len(samples) * 3, 3))
        for i, sample in enumerate(samples):
            ax[i].imshow(sample)
            ax[i].axis('off')
        fig.savefig(os.path.join(output_dir, f"target-{'-'.join(map(str, sorted(sg)))}.png"))
        
        with open(outfile, "wb") as f:
            pickle.dump({
                "subgroup": sg,
                "accuracies": accuracies,
                "f1": f1,
                "divs": divs,
                "y_trues": y_trues,
                "y_preds": y_preds,
                "blurs": blurs,
            }, f)
