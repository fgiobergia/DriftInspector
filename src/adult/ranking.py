from glob import glob
import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
import pandas as pd
from scipy.sparse import csr_array
from functools import reduce
from sklearn.metrics._ranking import _ndcg_sample_scores
import config
import sys
sys.path.append("../")

from sklearn.metrics import ndcg_score
from detect import detect_singlebatch, detect_multibatch, _get_altered_in_window
from utils import get_support_bucket

from distill import Result

from scipy.stats import spearmanr
from tqdm import tqdm

exp_type = "adult"
n_samples = 50
N = None
win_size = 5
table = {}
compute_ndcg =  True
compute_corr = False

sup_thresholds = {
    "adult": 1.0,
    "celeba": 0.5
}


for exp_type in ["adult", "celeba"]:
    if exp_type == "adult":
        noise = 0.5
        checkpoint = "xgb-adult"
        ckpt_dir = "/data2/fgiobergia/drift-experiments/"
    else:
        ckpt_dir = os.path.join(config.ckpt_dir, "sup-wise")
        noise = 1.0
        checkpoint = "resnet50"
    
    with open(f"{checkpoint}-results-v2.pkl", "rb") as f:
        results = pickle.load(f)

    data = []
    for r in results:
        if r.metric != "accuracy" or r.gt == "neg" or r.support[0] > sup_thresholds[exp_type]:
            continue
        d = {
            "support": r.support[0],
            "window": r.window,
            "result": r
        }
        data.append(d)

    df_results = pd.DataFrame(data=data)

    results_subset = df_results.groupby(["support", "window"]).apply(lambda gb: gb.sample(n=min(len(gb), n_samples))).result.tolist()

    GT = {}
    tstats = {}
    deltas = {}
    supports = set()


    for r in results_subset:
        if r.gt != "pos" or r.metric != "accuracy" or r.window != win_size:
            continue
        if r.support not in GT: # assume that , if not in GT, also not in tstats and deltas
            GT[r.support] = []
            tstats[r.support] = []
            deltas[r.support] = []
        GT[r.support].append(r.altered)
        tstats[r.support].append(r.tstat)
        deltas[r.support].append(r.delta)
        supports.add(r.support)
    supports = sorted(list(supports))

    GT = [ np.vstack(GT[sup]) for sup in supports ]
    tstats = [ np.vstack(tstats[sup]) for sup in supports ]
    deltas = [ np.vstack(deltas[sup]) for sup in supports ]

    all_GT = np.vstack([ GT[i] for i in range(len(supports)) ])
    all_delta = np.vstack([ deltas[i] for i in range(len(supports)) ])
    all_tstat = np.vstack([ tstats[i] for i in range(len(supports)) ])
    all_GT.shape, all_delta.shape, all_tstat.shape

    table[exp_type] = {}

    for method in ["delta", "tstat", "random"]:
        if method == "delta":
            y_score = -all_delta
        elif method == "tstat":
            y_score = all_tstat
        elif method == "random":
            y_score = np.random.random(all_GT.shape)
        else:
            raise ValueError("Invalid method")
        
        # various nDCG metrics
        if compute_ndcg:
            for k in [None, 10, 100]:
                scores = _ndcg_sample_scores(all_GT[:N], y_score[:N], k=k)
                key = "nDCG" if k is None else f"nDCG@{k}"
                table[exp_type][(key, method)] = f"{scores.mean():.4f} ± {scores.std():.4f}"
        
        if compute_corr:
            p = ((((all_GT[:N] - all_GT[:N].mean(axis=1, keepdims=True)) * (y_score[:N] - y_score[:N].mean(axis=1, keepdims=True))).mean(axis=1)) / (all_GT[:N].std(axis=1) * y_score[:N].std(axis=1)))
            table[exp_type][("Pearson", method)] = f"{p.mean():.4f} ± {p.std():.4f}"
            print(f"{exp_type} {method} Pearson: {p.mean():.4f} ± {p.std():.4f}")

            s = np.diagonal(spearmanr(all_GT[:N], y_score[:N], axis=1).statistic, offset=all_GT[:N].shape[0])
            table[exp_type][("Spearman", method)] = f"{s.mean():.4f} ± {s.std():.4f}"
            print(f"{exp_type} {method} Spearman: {s.mean():.4f} ± {s.std():.4f}")

df = pd.DataFrame(table).sort_index(level=(0,1))
print(df.to_latex())
