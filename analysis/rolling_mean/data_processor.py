import os
import sys

import numpy as np
import pandas as pd
import tqdm

DATA_PATH = os.path.join("data", "csv")
OUTPUT_PATH = os.path.join("outputs", "rolling_mean", "data.csv")

BETA = 20

COLUMNS = [
            "AF3",
            "F7",
            "F3",
            "FC5",
            "T7",
            "P7",
            "O1",
            "O2",
            "P8",
            "T8",
            "FC6",
            "F4",
            "F8",
            "AF4",
            "label"
          ]


def getFileNames(path):
    for root, dirs, files in os.walk(path):
        # One level deep .csv files
        return files


def prepareDataFromCsv(fname):
    df = pd.read_csv(fname, header=None)
    df = df.loc[1:, 2:]

    return df.astype(np.float32).rolling(window=BETA).mean().loc[BETA:BETA+10]


def extractLabel(fname):
    name, _ = os.path.splitext(fname)
    return name[-5:-2]


def runAll():
    global OUTPUT_PATH, DATA_PATH, BETA
    if len(sys.argv) > 1:
        OUTPUT_PATH = sys.argv[1]

    if len(sys.argv) > 2:
        DATA_PATH = sys.argv[2]

    filenames = getFileNames(DATA_PATH)

    data = pd.DataFrame(columns=COLUMNS)

    print("Collecting data from:", DATA_PATH)
    for i, f in tqdm.tqdm(enumerate(filenames)):
        X = prepareDataFromCsv(os.path.join(DATA_PATH, f))
        y = extractLabel(f)
        X['label'] = y
        X = X.rename(columns={i + 2: COLUMNS[i] for i in range(14)})
        data = data.append(X)

    print(data.columns)

    print("Saving to:", OUTPUT_PATH)
    data.to_csv(OUTPUT_PATH, index=False)
