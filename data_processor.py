import os
import sys

import numpy as np
import pandas as pd
import tqdm

DATA_PATH = os.path.join("data", "csv")
OUTPUT_PATH = os.path.join("outputs", "data.csv")

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

    return df.astype(np.float32).mean()


def extractLabel(fname):
    name, _ = os.path.splitext(fname)
    return name[-5:-2]


if __name__ == "__main__":
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
        data.loc[i] = X.tolist() + [y]

    print(data.columns)

    print("Saving to:", OUTPUT_PATH)
    data.to_csv(OUTPUT_PATH, index=False)
