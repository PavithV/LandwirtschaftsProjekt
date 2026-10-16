# Current ML Pipeline

This folder contains the current automated workflow for the project.

The pipeline is started manually from the repository root. It performs two steps:

1. Data preparation
2. Exploratory data analysis (EDA)

## Run the pipeline

The runner is located at `pipeline/_run_pipeline.py`.

## Step 1: Data preparation

File: `pipeline/src/data_prep.py`

The pipeline reads:

```text
data/train.csv
```

It then:

- Finds spectral-band columns whose names match `X<wavelength>` and sorts them by wavelength.
- Detects and removes spectral columns that are completely empty.
- Converts spectral values to numeric and linearly interpolates missing bands within each row.
- Uses spectral-column medians only as a fallback for completely empty rows.
- Calculates `spectral_sum`.
- Adds spectral statistics, vegetation indices, red-edge, senescence, water,
  and wavelength-aware gradient features used in Milestone 2.
- Encodes `Month` as numeric, sine, and cosine features.
- One-hot encodes categorical `AEZ` values.
- Removes the CSV index and identifier columns (`Unnamed: 0`, `H1`, and `id`).
- Saves the result to:

```text
data/processed/cleaned_data.csv
```

The input and output folders are created automatically when needed.

The resulting file is fully numeric apart from the target columns (`Crop` and
`Stage`) and contains no missing numeric feature values. All transformations
are row-local or fitted to the input data before downstream modelling.

## Step 2: EDA graphs

File: `pipeline/src/eda.py`

The cleaned data is used to create eight graphs in:

```text
output/eda_plots/
```

The graphs show:

1. Crop class distribution
2. Spectral-sum distribution by crop
3. Crop/stage count heatmap
4. Mean spectral signature for each crop
5. Spectral variance by crop
6. Correlation between sampled spectral bands
7. A two-dimensional PCA projection
8. Spectral signatures across growth stages for each crop

The EDA code identifies usable spectral bands again by removing bands that are 100% missing and sorting the remaining bands by wavelength.

For the PCA graph, missing spectral values are median-imputed and the values are standardized before PCA is applied. This PCA is only for visualization.

## Current workflow overview

```text
data/train.csv
       |
       v
Data preparation
       |
       v
data/processed/cleaned_data.csv
       |
       v
EDA graphs
       |
       v
output/eda_plots/
```

## What is currently not implemented

The current pipeline is a working baseline, but it is not yet the complete Milestone 2 workflow. It currently does not:

- Train a `Stage` classifier.
- Compare independent, multi-output, or hierarchical target strategies.
- Train or compare multiple model types.
- Perform cross-validation or hyperparameter search.
- Use PCA in the actual model pipeline.
- Report Macro F1.
- Evaluate predictions on `data/test.csv`.
- Generate predictions for the unlabeled `data/test.csv` file.
- Persist a formal feature schema or configuration file for applying the exact
  one-hot category set to a separately processed inference file.

Model training, evaluation, and inference are outside the scope of this
pipeline. The processed CSV is the handoff artifact for downstream modelling.

## Important distinction

The separate `data/test.csv` file is not used by `_run_pipeline.py`.

## Main files

```text
pipeline/
|-- _run_pipeline.py       # Starts data preparation and EDA
|-- src/
|   |-- data_prep.py       # Cleans data and creates engineered features
|   |-- eda.py             # Creates EDA graphs
|   |-- ml_train.py        # Standalone model-training helper
|   |-- ml_eval.py         # Standalone model-evaluation helper
|-- output/
    |-- eda_plots/         # Generated EDA images
```
