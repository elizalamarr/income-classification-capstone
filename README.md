# Income Classification Capstone

Binary classification on the [UCI Census Income (Adult) dataset](https://archive.ics.uci.edu/dataset/2/adult):
predicting whether an individual's income is `<=50K` or `>50K` from demographic and
employment features. Compares a Logistic Regression baseline (with `GridSearchCV`
hyperparameter tuning) against a small feed-forward neural network (Keras).

## Results

| Metric | Logistic Regression | Neural Network |
|---|---|---|
| Accuracy | 0.8529 | 0.8555 |
| F1 Score | 0.6596 | 0.6675 |

Full EDA, preprocessing rationale, model reflections, and ethical considerations are in
the notebook itself.

## About the data

The capstone was originally assigned with a pre-cleaned `censusData.csv` (provided
separately, not part of the standard UCI download). That file wasn't available when
this was set up outside the original course environment, so [`build_census_data.py`](build_census_data.py)
reconstructs an equivalent from the raw UCI files (`data_capstone/adult.data` +
`adult.test`): renames `sex`→`sex_selfID`, builds `income_binary`, and converts `?` to
proper missing values.

Two known differences from the original assignment data:
- The original had 162 seeded-missing `age` values (for a median-imputation exercise);
  the raw UCI data has none, so that step is a no-op here.
- The original's `sex_selfID` column used `Female`/`Non-Female` category labels; this
  reconstruction keeps the raw `Male`/`Female` labels since the exact relabeling wasn't
  known.

Both are why the metrics above differ slightly from earlier runs of this notebook.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/). Dependencies are pinned deliberately:
`scikit-learn==1.1.3` (the notebook uses `OneHotEncoder(sparse=False)` and
`get_feature_names()`, both removed in scikit-learn 1.2+) and `numpy<2` (needed for
that scikit-learn version's compiled wheels).

```bash
git clone https://github.com/elizalamarr/income-classification-capstone.git
cd income-classification-capstone
uv sync
```

## Running it

**Interactive (Jupyter Lab):**
```bash
uv run jupyter lab
```
Open `census_data_capstone.ipynb` and select the "Income Classification Capstone (uv)"
kernel.

**From the terminal, no browser (re-executes every cell and saves the outputs back into the notebook):**
```bash
uv run jupyter nbconvert --to notebook --execute --inplace census_data_capstone.ipynb
```

## Repo structure

```
census_data_capstone.ipynb   # the capstone notebook
build_census_data.py         # rebuilds the census CSV from raw UCI files
data_capstone/                # raw UCI files + the reconstructed CSV
pyproject.toml / uv.lock      # pinned dependencies
```
