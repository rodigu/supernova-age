- [supernova-age](#supernova-age)
  - [Overview](#overview)
  - [Data processing](#data-processing)
  - [Clustering algorithm](#clustering-algorithm)
  - [Clustering evaluation](#clustering-evaluation)

# supernova-age

## Overview

You need an empty directory `out` and a directory `data` that contains the raw simulated SN data.
The order is:

```python
r_read.py
```

```python
clustering.py
```

```python
cluster_eval_all.py
```

Cluster eval all will print each `df` it reads as it runs.

## Data processing

Running `r_read.py` as is will produce a file `output_1_test_typed.csv` inside the `out` folder.
It expects data from inside a `data` folder containig the simulations of the SNe.

Altering the constant list `BAND_CHOICE` will change which bands are kept after processing.
Altering the constant int `DAY_RANGE` will affect the range of days for averageing band readings.
E.g. for `DAY_RANGE=3`, we average band readings done over the span of 3 days.

## Clustering algorithm

Running `clustering.py` will generate folders for each clustering algorithm. Inside those directories, it will also create one dir for band difference clustering and another one for band clustering. Inside of those, there will be 3 to 4 folders labeled by the input used to create the resulting csvs that are inside them. The 3 csvs inside represent each of the SN types (II, Ia, Ibc).
Note that you might get slightly different results due to the randomized nature of the clustering algorithms.

## Clustering evaluation

Running `cluster_eval_all.py` will make use of the generated csvs and create 3 plots for each csv that are used in our evaluation process.
