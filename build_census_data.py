"""
Rebuilds a censusData.csv equivalent from the raw UCI Adult files
(data_capstone/adult.data + adult.test), since the original Codecademy/Canvas
-provided cleaned CSV wasn't available.

Known differences from the original censusData.csv used in this capstone:
  - 'sex' is renamed to 'sex_selfID' and the label column to 'income_binary',
    matching the column names the notebook expects.
  - '?' missing markers are converted to real NaN.
  - adult.test's leading garbage line and trailing '.' on labels are stripped,
    and adult.data + adult.test are concatenated into one pool (the notebook
    does its own train_test_split, so a single combined pool matches its flow).
  - The original write-up reports 162 missing 'age' values for median
    imputation practice. The raw UCI data has zero missing 'age' values (that
    gap doesn't exist upstream), so that specific imputation step will be a
    no-op here. Everything else in the notebook is unaffected.
"""
import os
import pandas as pd
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data_capstone")

COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country",
    "income",
]

def load(path, skiprows=0):
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skiprows=skiprows,
        skipinitialspace=True,
        na_values="?",
    )
    df["income"] = df["income"].str.rstrip(".")
    return df

train = load(os.path.join(DATA_DIR, "adult.data"))
test = load(os.path.join(DATA_DIR, "adult.test"), skiprows=1)

df = pd.concat([train, test], ignore_index=True)
df = df.dropna(how="all")  # drops the trailing blank line in adult.data
df = df.rename(columns={"sex": "sex_selfID", "income": "income_binary"})

out_path = os.path.join(DATA_DIR, "censusData_reconstructed_from_uci.csv")
df.to_csv(out_path, index=False)
print(f"Wrote {len(df)} rows to {out_path}")
print(df["income_binary"].value_counts())
print("Missing age values:", df["age"].isnull().sum())
