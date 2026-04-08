# `generate_splits.py`
This utility generates canonical, strictly deterministic dataset split manifests for **CIFAR-10** and **MUFAC**. It is specifically designed for evaluating Machine Unlearning algorithms by defining exact, reproducible partitions.

The script creates one mixed-label unlearning task per dataset with the following mathematically validated subsets:
* **Train:** The canonical training split.
* **Forget:** A deterministic, proportionally allocated subset of `train` designated for unlearning.
* **Retrain:** The exact complement of the forget set (`train \ forget`), used for training gold-standard models from scratch.
* **Val / Test:** Held-out evaluation sets.

## Prerequisites & Data Structure
Make sure to run `download_data.ipynb` from the data folder first before this.

By default, the script looks for a `data` directory one level above the script's location. Ensure your raw datasets follow this structure before running:

```text
[data_root]/
├── cifar-10/
│   └── cifar-10-python.tar.gz          # Fallback if torchvision is unavailable
└── MUFAC/
    ├── custom_train_dataset.csv
    ├── custom_val_dataset.csv
    ├── custom_test_dataset.csv
    ├── fixed_val_dataset_positive/
    ├── fixed_val_dataset_negative/
    ├── fixed_test_dataset_positive/
    └── fixed_test_dataset_negative/
```

## Usage Examples

### Basic Generation
Generate the default 10-class splits for both CIFAR-10 and MUFAC. Outputs will be saved in the directory where the script resides.
```bash
python generate_splits.py
```

### Specify Output and Data Directories
Point to a custom raw data location and save the generated manifests to a specific folder.
```bash
python generate_splits.py --data-root /path/to/raw/data --out-root ./generated_manifests
```

### Target a Specific Dataset
Generate splits only for CIFAR-10.
```bash
python generate_splits.py --datasets cifar10
```

### Custom Forget Strategies
Modify the size and scope of the unlearning task. For example, forget 5% of the data, restricted to the top 2 most frequent classes.
```bash
python generate_splits.py --forget-percentage 5.0 --forget-top-k-classes 2
```

### Reproducibility Verification (Audit Mode)
Regenerate all artifacts into a temporary sandbox and perform a byte-for-byte comparison against your existing output directory. This ensures your currently saved manifests are uncorrupted and standard-compliant.
```bash
python generate_splits.py --verify-referenced-files
```

---

## Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--data-root` | Path | `../data` | Root directory containing the raw `cifar-10` and `MUFAC` folders. |
| `--out-root` | Path | `.` (Script dir) | Output directory for the generated CSV and JSON manifests. |
| `--datasets` | List | `cifar10 mufac` | Space-separated list of datasets to process. Choices: `cifar10`, `mufac`. |
| `--verify-only` | Flag | `False` | Audits existing manifests by regenerating and diffing them instead of writing new files. |
| `--forget-percentage`| Float | *Dynamic* | Percentage of the canonical train split to assign to the forget set. Defaults to `100 / number_of_labels`. |
| `--forget-top-k-classes`| Int | `None` | Restricts the forget set to draw only from the *k* most frequent labels. |

---

## Output Structure

Running the script successfully will populate your `--out-root` with the following structure:

```text
[out_root]/
├── cifar10/
│   ├── samples.csv                     # Master catalog of all images and labels
│   └── tasks/
│       └── forget_mixed.json           # The unlearning task manifest
├── mufac/
│   ├── samples.csv
│   └── tasks/
│       └── forget_mixed.json
└── Report/
    └── figures/
        ├── cifar10_forget_mixed_histograms.svg   # Visual distribution of splits
        └── mufac_forget_mixed_histograms.svg
```

### Understanding the JSON Manifest
The generated `.json` files inside the `tasks/` directory contain arrays of exact `sample_id` strings mapping to the data rows in `samples.csv`. They also contain built-in sanity checks, including absolute item counts, label distribution dictionaries, and cryptographic fingerprints to guarantee dataset integrity.