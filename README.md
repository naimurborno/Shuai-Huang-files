# Project README

This repository contains code to prepare a custom dataset and train a model that expects features, cluster indices, and binary labels in a specific format.

---

## 1. Dataset Format (Input)

Your **raw dataset must be stored in a single directory** and contain the following `.mat` files:

* **`label.mat`**
  Contains binary labels for the samples.

* **`s_x_feature.mat`**
  Contains the feature matrix.

* **`s_x_cluster_index.mat`**
  Contains the cluster index matrix.

No other structure is expected at this stage. Just place these files together in one folder.

---

## 2. Dataset Preparation

Before training, the dataset must be converted into the format expected by the model.

Run the dataset preparation script:

```bash
python Prepare-dataset.py \
  --data_dir "<path to folder containing the .mat files>" \
  --dest_dir "<path to output directory>"
```

### Output Structure

After running the script, the destination directory will contain:

```
<dest_dir>/
│
├── Cluster_index_mat/
│   └── *.mat        # Cluster index files
│
├── Features/
│   └── *.npy        # Feature matrices
│
└── label.mat        # Binary labels
```

This converted structure is **mandatory** for training.

---

## 3. Training Configuration

Before starting training, edit the `train_config.py` file.

Set the following paths:

```python
config["label_data_dir"] = "<path to label.mat>"
config["feature_data_dir"] = "<path to Features folder>"
config["cluster_data_dir"] = "<path to Cluster_index_mat folder>"
```

Also configure the required hyperparameters in the same file:

* Number of epochs
* Learning rate
* Stride
* Kernel sizes

Make sure these values match your dataset and experimental setup.

---

## 4. Training the Model

Once the dataset is prepared and `train_config.py` is properly configured, start training by running:

```bash
python train.py
```

The training script will automatically load the processed data and begin training using the provided configuration.

---

## 5. Notes

* Do **not** skip the dataset preparation step. The training code will not work with raw `.mat` files.
* Ensure all paths in `train_config.py` are correct and absolute if possible.
* The model assumes **binary classification**.

---

If something breaks, it is almost certainly due to an incorrect directory path or a missing file. Double-check those first.
