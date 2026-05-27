# FEMNIST Experiment Memory

This document records the practical decisions made while preparing the FEMNIST thesis experiments.

## Dataset Setup

- Dataset source: `flwrlabs/femnist` from Hugging Face, which keeps the LEAF-style `writer_id` field.
- Original benchmark reference: LEAF FEMNIST.
- Federated partition: natural writer/client split.
- Number of available writers in the Hugging Face copy: `3597`.
- Number of classes: `62` (`0-9`, `A-Z`, `a-z`).
- Train/test split: deterministic per-writer `80/20` split because the Hugging Face copy has one split.
- Seed: `20260524`.
- Client selection: deterministic selection of `100` writers.
- Minimum selected-client requirements: at least `100` train samples and `20` test samples.
- Selected subset size after locking the final 100-client setting:
  - train samples: `19,554`
  - test samples: `4,886`
  - minimum train samples/client: `100`
  - minimum test samples/client: `25`
  - median train samples/client: `142`
  - median test samples/client: `35.5`
  - selected classes covered: `62 / 62`
  - missing classes: none
- Data transforms:
  - convert image to grayscale
  - convert to `torch.float32`
  - scale pixels to `[0, 1]`
  - shape images as `(1, 28, 28)` for CNNs
  - labels as `torch.long`
  - no resize, no data augmentation, no runtime shuffle in the dataset handler

The dataset split and client selection now use the decent-bench interoperability RNG layer. Experiment 0 also calls
`iop.set_seed(20260524)` before building the benchmark problem, so model initialization and benchmark trial seeds are
reproducible.

## Client Count Decision

Current decision: use `100` clients for the required FEMNIST experiments.

Rationale:

- The selected subset still covers all `62` FEMNIST classes with seed `20260524`.
- It keeps the full experiment matrix feasible enough to run all selected algorithms in one decent-bench call.
- A 200-client all-algorithm run hit GPU memory limits on the A10 setup, while the 100-client setting completed.
- It is still a natural-writer FEMNIST benchmark, not an artificial partition.

Increasing the number of clients on a remote GPU may be possible, but it should only be done as an optional robustness
check after the required thesis benchmark is complete.

### Selected-Client Class Distribution

The plot `experiments/femnist/results/inspection/selected_client_class_distributions.png` shows the class histogram
for the fixed 100 selected writers. It should be interpreted as evidence of natural FEMNIST heterogeneity.

The selected clients are sorted by `writer_id` in `selected_clients_stats.csv`. In that order, the top half of the plot
contains writers with substantially more samples and a more balanced digit/uppercase/lowercase distribution. The bottom
half contains fewer samples and is much more digit-heavy.

Quantitatively, for the selected 100 clients:

| Plot group | Mean samples/client | Mean classes/client | Digit samples | Uppercase samples | Lowercase samples |
| --- | ---: | ---: | ---: | ---: | ---: |
| Top 50 writers in `writer_id` order | `325.5` | `56.5` | `32.8%` | `36.1%` | `31.2%` |
| Bottom 50 writers in `writer_id` order | `163.3` | `53.8` | `71.7%` | `14.4%` | `14.0%` |

This means the fixed FEMNIST subset has both quantity skew and label-distribution skew while still covering all 62
classes.

## Model

- Model: LEAF-lite CNN with convolution channels `32 -> 64`, `5x5` kernels, and dense layer size `256`.
- Parameter count: `871,102`.
- Output: `62` raw logits.
- Loss: `torch.nn.CrossEntropyLoss`.
- Metric activation: `ArgmaxActivation`.
- Batch size: `32`.
- Max batch size: `256`.
- Default device: CPU, but GPU is preferred for practical runs.
- Final FEMNIST comparison target:
  - clients: `100`
  - selected clients per round: `20 / 100`
  - selection scheme: `UniformSelection(fraction_selected_clients=0.2)`
  - trials: `3`
  - iterations: `???`
  - state snapshot period: `100` by default, `50` only for selected plotting runs
  - checkpoint step: `None`

The model intentionally has no final softmax layer. `CrossEntropyLoss` expects logits, and class predictions only need
`argmax(logits)`.

## Model Width Decision

Current model uses a LEAF-style convolutional frontend with `32 -> 64` channels and `5x5` kernels. This is chosen to
keep the FEMNIST model close to the model family used in the LEAF FEMNIST experiments, rather than tuning architecture
as another benchmark variable. The architecture itself is therefore treated as a fixed benchmark condition.

The common LEAF FEMNIST CNN uses:

- two `5x5` convolution layers,
- `32` then `64` channels,
- max pooling after each convolution,
- one fully connected layer with `2048` units,
- final softmax over labels `0-61`.

The thesis model keeps the LEAF convolutional frontend and the 62-class output, but reduces the dense layer to `256`
units. This gives a substantially smaller model while preserving the same type of visual feature extractor. The reason
for reducing the dense layer is computational: the full LEAF `2048`-unit dense layer would substantially increase model
parameters, communication payloads, stored snapshots, checkpoint sizes, and GPU memory pressure. The model outputs logits instead of an explicit softmax because `torch.nn.CrossEntropyLoss`
expects logits and applies the softmax/log-softmax internally.

The LEAF FEMNIST systems tutorial uses a 5% FEMNIST subsample with an `80/20` train/test split, trains the CNN for
`2000` rounds, and gives an example command with `3` clients per round and `10%` batch size:
https://leaf.cmu.edu/build/html/tutorials/femnist-md.html

## Pilot Runs

On local CPU smoke test - 2 clients, 1 iteration, FedAvg, completed successfully.
The local CPU is not practical for final-style FEMNIST pilots. A 20-client, 500-iteration FedAvg pilot reached
iteration `249` before timing out and produced a very large partial checkpoint. CPU should be reserved for smoke tests
and small debugging runs.

The local GPU environment works:

- PyTorch: `2.11.0+cu128`
- CUDA available: `True`
- GPU: `NVIDIA GeForce RTX 2050`

GPU smoke test:

- 2 clients, 1 iteration, FedAvg, completed successfully.

200-client GPU pilots, 1 trial, 300 iterations:

| Run | Training time | Loss at 0 | Loss at 100 | Loss at 300 | Server accuracy at 300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FedAvg, LR `0.01`, E=1 | about 6 min | `4.125` | `3.698` | `3.592` | `5.78%` |
| FedAvg, LR `0.05`, E=1 | about 5 min | `4.125` | `3.578` | `3.571` | `5.78%` |

The pilots plateaued early. For these FedAvg candidates, simply increasing to `1000` iterations is unlikely to fix the
low accuracy. The next useful step is hyperparameter tuning, especially local learning rate and number of local epochs.

Failed or memory-limited pilot runs:

| Hardware | Clients | Algorithms | Participation | Trials | Iterations | Snapshot period | Checkpoint step | Outcome |
| --- | ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| Local CPU, Intel i7-13620H | 200 | FedAvg | Full | 1 | 500 | not recorded | enabled | Reached iteration `249` before timing out; produced a large partial checkpoint. |
| Local GPU, RTX 2050 4 GB | 200 | All 10 algorithms | Full | 2 | 400 | `40` | `None` | CUDA out of memory after FedAvg reached `50%` overall progress / trial `1/2`; failed while deep-copying the network for the next trial/algorithm. |
| NVIDIA A10 24 GB (Lambda)| 200 | All 10 algorithms | Full | 2 | 400 | `40` | `200` | CUDA out of memory after FedAvg completed both trials in about `23-24` minutes; the process was using almost the full A10 memory during deep-copy/state retention. |
| NVIDIA A10 24 GB (Lambda)| 100 | All 10 algorithms | Full | 1 | 400 | `400` | `None` | Completed successfully; FedAvg and FedProx each finished in under `6` minutes. This motivated moving the locked subset from 200 to 100 clients. |

These failures were mostly memory/state-retention issues. The expensive part is the
combination of clients, algorithms, trials, stored snapshots, PyTorch model state, and decent-bench deep copies used to
preserve comparable benchmark states for metrics.


## Experiment 0 Design

Hyperparameter tuning.

Experiment 0 uses a validation split carved from the existing FEMNIST training split:

- tuning train: `80%` of the handler's train split,
- validation: `20%` of the handler's train split,

Final comparison experiments should train on the full handler train split and evaluate on the handler test split.
