- [supernova-age](#supernova-age)
  - [Overview](#overview)
  - [Order to excecute](#order-to-excecute)
  - [Data processing](#data-processing)
  - [Clustering algorithm](#clustering-algorithm)
  - [Clustering evaluation](#clustering-evaluation)

# supernova-age

## Overview

Our data was provided courtesy of Patrick D. Aleo and can be downloaded at: https://uofi.box.com/s/l5nw5lrpwqnezm8ptu7u3j79os9rqjdo

After downloading `YSE_DR1_SIMS_60k_SNR4_grizXY_SIMLIB_FLUXCALERR_COR_220427.tar.gz`, create a directory called `data` and extract the contents to it.
You also need to create an empty folder called `out`.

Before running `r_read.py`, your directory should look like this:

```
|   clustering.py
|   cluster_eval_all.py
|   README.md
|   r_read.py
|
+---data
|   +---PALEO_YSE_ZTF_MODEL01
|   +---PALEO_YSE_ZTF_MODEL12
|   +---PALEO_YSE_ZTF_MODEL20
|   +---PALEO_YSE_ZTF_MODEL33
|   \---PALEO_YSE_ZTF_MODEL37
\---out
```

After running `r_read.py`, `output_1_typed.csv` will be created inside the `out` directory.
`r_read.py` takes around 11 hours to run on a M1 MacBook Pro 2020.

## Order to excecute

The order is:

```python
python r_read.py
```

```python
python clustering.py
```

```python
python cluster_eval_all.py
```

Cluster eval all will print each `df` it reads as it runs.

If you already have access to `output_1_typed.csv`, you wont need to run `r_read.py`
That file can be downloaded from: https://raw.githubusercontent.com/rodigu/supernova-age/main/output_1_typed.csv

## Data processing

Running `r_read.py` as is will produce a file `output_1_test_typed.csv` inside the `out` folder.
It expects data from inside a `data` folder containig the simulations of the SNe.

Altering the constant list `BAND_CHOICE` will change which bands are kept after processing.
Altering the constant int `DAY_RANGE` will affect the range of days for averageing band readings.
E.g. for `DAY_RANGE=3`, we average band readings done over the span of 3 days.

## Clustering algorithm

Running `clustering.py` will generate folders for each clustering algorithm. Inside those directories, it will also create one dir for band difference clustering and another one for band clustering. Inside of those, there will be 3 to 4 folders labeled by the input used to create the resulting csvs that are inside them. The 3 csvs inside represent each of the SN types (II, Ia, Ibc).
Note that you might get slightly different results due to the randomized nature of the clustering algorithms.

You need `output_1_test_typed.csv` inside the `out` folder to run `clustering.py`.

## Clustering evaluation

Running `cluster_eval_all.py` will make use of the generated csvs and create 3 plots for each csv that are used in our evaluation process.
