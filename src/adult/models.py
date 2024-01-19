import argparse
import os
import numpy as np
import pickle
import random
import pandas as pd

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import xgboost as xgb

from config import *


def random_split(df, ratios):
    df = df.sample(frac=1).reset_index(drop=True) # shuffle
    splits = []
    start = 0
    for ratio in ratios:
        split = df.iloc[start:start+int(len(df)*ratio)].reset_index(drop=True)
        start += int(len(df)*ratio)
        splits.append(split)
    return splits
    
def load_adult_df():
    columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation","relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","target"]
    categ = ["workclass","education","marital-status","occupation","relationship","race","sex","native-country"]
    num = [ c for c in columns if c not in categ and c != "target" ]
    df = pd.concat([pd.read_csv(os.path.join(data_dir, "adult.data"), header=None, index_col=False), pd.read_csv(os.path.join(data_dir, "adult.test"), header=None, index_col=False, skiprows=1)], axis=0)
    df.columns = columns
    df["target"] = df["target"].apply(lambda x: 1 if x in [" >50K", ">50K."] else 0)
    return df, categ, num

if __name__ == "__main__":
    # if called, train a model

    argParser = argparse.ArgumentParser()

    argParser.add_argument("--checkpoint", type=str)

    args = argParser.parse_args()

    df, categ, num = load_adult_df()


    # split `ds` into a train & test -- the test set is then further split into 30 chunks
    # NOTE: we should probably store these sets somewhere
    # (they should be deterministic, but still)
    random.seed(42)
    np.random.seed(42)

    test_size = 0.5
    n_chunks = 30
    df_train, *df_tests = random_split(df, [ 1- test_size] + [test_size / n_chunks] * n_chunks)

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categ)])

    y_train = df_train["target"]
    X_train = df_train.drop(columns=["target"])
    X_train = preprocessor.fit_transform(X_train)

    xgb = xgb.XGBClassifier()
    xgb.fit(X_train, y_train)

    # save model

    model_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.pkl")
    ds_filename = os.path.join(ckpt_dir, f"{args.checkpoint}.dataset.pkl")

    with open(model_filename, "wb") as f:
        pickle.dump(xgb, f)
    
    with open(ds_filename, "wb") as f:
        pickle.dump({
            "train": df_train,
            "test_chunks": df_tests,
            "numerical": num,
            "categorical": categ,
            "transform": preprocessor
        }, f)