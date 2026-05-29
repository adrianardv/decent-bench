# FEMNIST Experiment Setup

This folder contains the thesis-specific FEMNIST setup and inspection utilities.

## Folder Structure

- `inspect_femnist.py`: command-line inspection script.
- `smoke/`: smoke and feasibility scripts for local/Lambda test runs.
- `experiment0/`: hyperparameter tuning scripts and selected Experiment 0 hyperparameter summary.
- `src/inspection_helpers.py`: inspection/statistics helpers.
- `src/femnist_handler.py`: `FEMNISTDatasetHandler`, used to create decent-bench train/test datasets.
- `src/model.py`: FEMNIST CNN used by the experiment scripts.
- `experiment_log.md`: notes and decisions for the thesis chapter.
- `data/`: local dataset cache, ignored by git.
- `results/`: generated summaries, plots, and future experiment outputs, ignored by git.
- `checkpoints/`: benchmark checkpoints and saved metric outputs, ignored by git.

## Source Decision

FEMNIST originally comes from LEAF [Caldas et al., 2018]. LEAF preprocessing produces JSON files that are already
partitioned by user/writer, with each user containing `x` image vectors and `y` labels. This is the most faithful source
if the LEAF preprocessing pipeline has already been run.

The Hugging Face copy `flwrlabs/femnist`, distributed by Flower Labs, is often more convenient [Beutel et al., 2020]. It
keeps the important fields needed for the benchmark:

- `writer_id`, which gives the natural federated client partition.
- `character`, the 62-class label.
- 28 x 28 images.

The Hugging Face copy has one split, so this experiment code creates a deterministic per-writer train/test split. That is
similar in spirit to LEAF's sample-level train/test split.

## First Inspection Command

The Hugging Face source requires the optional `datasets` package:

```powershell
.venv\Scripts\python.exe -m pip install datasets
```

Then run:

```powershell
.venv\Scripts\python.exe experiments\femnist\inspect_femnist.py `
  --source huggingface `
  --candidate-clients 100 `
  --seed 20260524
```

The command writes CSV/JSON summaries and plots to `experiments/femnist/results/inspection/`.
The most important files are `all_clients_stats.csv` and `selected_clients_stats.csv`.
After the first download, add `--local-files-only` to use the cached dataset without contacting Hugging Face.
The default selected-client thresholds are `100` train samples and `20` test samples.

If LEAF JSON files have already been generated, use:

```powershell
.venv\Scripts\python.exe experiments\femnist\inspect_femnist.py `
  --source leaf-json `
  --leaf-train-dir path\to\leaf\data\femnist\data\train `
  --leaf-test-dir path\to\leaf\data\femnist\data\test `
  --candidate-clients 100 `
  --seed 20260524
```

## Using the Dataset Handler

```python
from experiments.femnist.src import FEMNISTDatasetHandler

train_dataset = FEMNISTDatasetHandler(split="train")
test_dataset = FEMNISTDatasetHandler(split="test")

train_partitions = train_dataset.get_partitions()
test_data = test_dataset.get_datapoints()
```

By default, the handler downloads/cache-loads `flwrlabs/femnist` and deterministically selects the clients using
`n_clients`, `min_train_samples`, `min_test_samples`, and `seed`. This means the same code can run on a remote GPU from a
clean clone without uploading the cached dataset or the inspection CSV files.

For a CNN, the default image layout is `(1, 28, 28)`. For an MLP, use `image_layout="flat"` to get vectors of shape
`(784,)`.

If the dataset is already cached and the machine should not contact Hugging Face, pass `local_files_only=True`. If a
fixed selected-client manifest should be used instead of deterministic reselection, pass `selected_clients_path`.

## Keeping the Same Clients Across Experiments

If two runs use the same FEMNIST handler settings, they select the same clients:

- `n_clients`
- `min_train_samples`
- `min_test_samples`
- `seed`
- `train_fraction`

The default values select the same 100 writers every time.
Dataset splitting and client selection use the decent-bench interoperability RNG layer.

For extra safety, or when the selected clients should be fixed explicitly from a saved file, pass the selected-client
manifest:

```python
FEMNISTDatasetHandler(
    split="train",
    selected_clients_path="experiments/femnist/results/inspection/selected_clients_stats.csv",
)
```

Use the same `selected_clients_path` for both train and test handlers.

## FEMNIST Benchmark Setup

All FEMNIST experiments should use the same selected writer set. The selected set is defined by:

- `n_clients = 100`
- `seed = 20260524`
- `train_fraction = 0.8`
- `min_train_samples = 100`
- `min_test_samples = 20`
- `clients_per_round = 20` / `selection_fraction = 0.2`
- `n_trials = 3`
- `iterations = 1000?`
- `state_snapshot_period = 100?`
- `checkpoint_step = None`
- `batch_size = 32`

These settings deterministically select the 100 writers that will be used throughout the FEMNIST benchmark experiments.
Do not change the seed or selection thresholds between experiments, otherwise the runs will no longer be directly
comparable.

The CNN model selected for all FEMNIST experiments is:

```text
Input: 1 x 28 x 28
Conv2d: 1 -> 32, kernel 5x5, padding 2, ReLU
MaxPool2d: 2x2
Conv2d: 32 -> 64, kernel 5x5, padding 2, ReLU
MaxPool2d: 2x2
Flatten: 64 * 7 * 7
Dense: 256, ReLU
Output Dense: 62 logits
Loss: torch.nn.CrossEntropyLoss
```

The model outputs logits, not softmax probabilities. `torch.nn.CrossEntropyLoss` applies the softmax/log-softmax
operation internally.

## References

The BibTeX entries used for this dataset setup are stored in `references.bib`.

- Caldas et al. (2018), LEAF: A Benchmark for Federated Settings.
- Beutel et al. (2020), Flower: A Friendly Federated Learning Research Framework.
