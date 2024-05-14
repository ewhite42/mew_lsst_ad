import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
import os

subspace = ["a*", "i-z", "a", "sini", "e"]
subspace_err = []
colorspace = ["a*", "i-z"]



def loader(
        file = "1Mssobjects.csv", # file path
        size = None, # Number of rows
        cols = ["a*", "i-z", "a", "sini", "e"] # ensure values are present
):
 
    df = pd.read_csv(f"{os.path.dirname(os.path.realpath(__file__))}/1Mssobjects.csv", index_col=0)

    df["u-g"] = df["uh"] - df["gh"]
    df["u-r"] = df["uh"] - df["rh"]
    df["u-i"] = df["uh"] - df["ih"]
    df["u-z"] = df["uh"] - df["zh"]
    df["u-y"] = df["uh"] - df["yh"]


    df["g-r"] = df["gh"] - df["rh"]
    df["g-i"] = df["gh"] - df["ih"]
    df["g-z"] = df["gh"] - df["zh"]
    df["g-y"] = df["gh"] - df["yh"]



    df["r-i"] = df["rh"] - df["ih"]
    df["r-z"] = df["rh"] - df["zh"]
    df["r-y"] = df["rh"] - df["yh"]


    df["i-z"] = df["ih"] - df["zh"]
    df["i-y"] = df["ih"] - df["zh"]

    df["z-y"] = df["zh"] - df["yh"]

    # Need to calculate errors for colors?


    df["a*"] = .89*(df["g-r"]) + .45*(df["r-i"]) -.57

    df["sini"] = np.sin(df["incl"] * np.pi/180)
    df["a"] = df["q"]/(1-df["e"])

    df = df.dropna(subset=cols)
    if size:
        return df.iloc[:size, :]

    else:
        return df.dropna(subset=subspace)

def preprocessed_subspace():
    return loader()[subspace]

def normalised_subspace():
    return scaler.fit_transform(preprocessed_subspace().to_numpy())

scaler = MinMaxScaler()
scaler.fit(preprocessed_subspace().to_numpy())


