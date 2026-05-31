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

## Model and configuration

- Model: LEAF-lite CNN with convolution channels `32 -> 64`, `5x5` kernels, and dense layer size `256`.
- Parameter count: `871,102`.
- Output: `62` raw logits.
- Loss: `torch.nn.CrossEntropyLoss`.
- Metric activation: `ArgmaxActivation`.
- Batch size: `32`.
- Max batch size: `256`.
- Default device: CPU, but GPU is preferred for practical runs.
- Main FEMNIST experiment configuration:

| Setting | Value |
| --- | --- |
| Number of clients | `100` |
| Minimum train samples/client | `100` |
| Minimum test samples/client | `20` |
| Number of classes | `62` |
| Train/test split | deterministic per-writer `80/20` |
| Client selection per round | `20 / 100` clients |
| Client selection scheme | `UniformSelection(fraction_selected_clients=0.2)` |
| Trials | `3` independent trials |
| Iterations | `1000` |
| Model | Conv32 -> Conv64 -> Dense256 -> 62 logits |
| Checkpoint step | `None` |
| Batch size | `32` |
| Algorithms | final selected subset after Experiment 0 tuning |
| Seed | `20260524` |


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
| Local GPU, RTX 2050 4 GB | 20 | All 10 algorithms via `smoke/smoke_run.py` | Full | 1 | 1000 | `50` | `None` | CUDA out of memory when FedAdagrad started. FedAvg, FedProx, SCAFFOLD, FedNova, FedAdam, and FedYogi completed first. After FedAvg, about `2 / 4` GB of dedicated GPU memory was already used; after FedProx, about `3.9 / 4` GB was used. FedAvg and FedProx took about `20` minutes each. SCAFFOLD took about `1h45m` and left dedicated GPU memory at `4 / 4` GB. |
| NVIDIA A10 24 GB (Lambda)| 200 | All 10 algorithms | Full | 2 | 400 | `40` | `200` | CUDA out of memory after FedAvg completed both trials in about `23-24` minutes; the process was using almost the full A10 memory during deep-copy/state retention. |
| NVIDIA A10 24 GB (Lambda)| 100 | All 10 algorithms | Full | 1 | 400 | `400` | `None` | Completed successfully; FedAvg and FedProx each finished in under `6` minutes. This motivated moving the locked subset from 200 to 100 clients. |
| NVIDIA A100 SXM4 40 GB (Lambda) | 100 | All 10 algorithms via `smoke/smoke_run.py` | Full | 1 | 800 | `100` | `None` | CUDA out of memory in the last algorithm, `FedPD`. This run used the selected 100-client FEMNIST setup, the LEAF-lite CNN, batch size `32`, and the same fixed shared smoke hyperparameters. It shows that even with fewer iterations and fewer snapshots, the full 10-algorithm benchmark in one `benchmark()` call is too memory-heavy for an A100 40 GB. |
| NVIDIA H100 80 GB HBM3 (Lambda) | 100 | 4 algorithms via `smoke/smoke_feasibility.py`: SCAFFOLD, FedNova, FedLT, FedDyn | Full | 3 | 1000 | `100` | `None` | Benchmark execution completed for all 4 algorithms. Each algorithm took about `30` minutes to complete 3 trials. GPU memory usage was about `60 / 80` GB after all 4 algorithms finished. This suggests the selected 100-client, 1000-iteration, 3-trial, full participation setting is feasible on H100 for a 4-algorithm benchmark, but with limited memory margin. |

These failures were mostly memory/state-retention issues. The expensive part is the
combination of clients, algorithms, trials, stored snapshots, PyTorch model state, and decent-bench deep copies used to
preserve comparable benchmark states for metrics.


## Experiment 0 Design

Hyperparameter tuning.

Experiment 0 uses a validation split carved from the existing FEMNIST training split:

- tuning train: `80%` of the handler's train split,
- validation: `20%` of the handler's train split,

This is a fixed hold-out validation setup, not cross-validation. Cross-validation was not selected because it would
multiply the cost of every federated hyperparameter candidate.

Final comparison experiments should train on the full handler train split and evaluate on the handler test split.

Current tuning protocol:

- run one algorithm family per process;
- use `n_trials = 1` for tuning to reduce runtime and memory;
- use `UniformSelection(fraction_selected_clients=0.2)` for all algorithms except `FedPD`;
- tune `FedPD` with full participation because it does not support partial participation;
- use `checkpoint_step = None`;
- do not use a checkpoint manager for candidate runs, so candidate tuning saves only CSV/JSON results;
- use `state_snapshot_period = iterations` for candidate runs because the candidates are not plotted;
- run a random/coarse search first;
- run a focused grid search around the best random/coarse candidate;
- for `FedOpt`, run a focused grid separately around the best random/coarse candidate for each FedOpt variant
  (`FedAdam`, `FedYogi`, and `FedAdagrad`), then select the best-performing variant;
- for `FedNova`, compare the plain variant, each optional mechanism alone, both momentum mechanisms together, and all
  three optional mechanisms together (`use_momentum`, `use_prox`, and `use_server_momentum`); for the proximal term,
  restrict `mu` to the values used in the FedNova paper: `{0.0005, 0.001, 0.005, 0.01}`;
- for `FedLT`, first tune `step_size`, `num_local_epochs`, and `rho` with `local_solver="gd"`, then compare `gd`,
  `adam`, and `nesterov` and default solver-specific parameters;
- optionally run the final best candidate for `2000` iterations with `state_snapshot_period = final_iterations / 10`
  to inspect whether performance plateaus before the end.

The final best-candidate curve is meant to help decide whether the later FEMNIST benchmark experiments should use
`1000` or `2000` iterations.

### Experiment 0 Selected Hyperparameters

The following hyperparameters have been selected from the completed and accepted Experiment 0 tuning runs. These are the
current candidates to carry forward into the main FEMNIST experiments. The same values are also stored in
`experiment0/selected_hyperparameters.json`.

| Algorithm | Selected variant | "Shared" hyperparameters | Algorithm-specific hyperparameters | Validation result |
| --- | --- | --- | --- | --- |
| FedAvg | FedAvg | `step_size = 0.1`, `num_local_epochs = 4` | none | Stable final curve; about `83%` server accuracy. |
| FedProx | FedProx | `step_size = 0.1`, `num_local_epochs = 4` | `mu = 0.025887619090591573` | Stable final curve; about `83%` server accuracy. |
| SCAFFOLD | SCAFFOLD | `step_size = 0.02441691061516309`, `num_local_epochs = 8` | `server_step_size = 1.0` | Stable final curve after focused rerun; about `82%` server accuracy. |
| FedNova | local + server momentum, no prox | `step_size = 0.015780201353739066`, `num_local_epochs = 3` | `use_momentum = True`, `use_server_momentum = True`, `use_prox = False`, `beta = 0.5`, `gamma = 0.9` | Stable final curve; about `84%` server accuracy. |
| FedOpt family | FedAdam | `step_size = 0.016454811464286817`, `num_local_epochs = 7` | `server_step_size = 0.005781649782731609`, `beta_1 = 0.9`, `beta_2 = 0.9`, `tau = 0.001` | FedAdam selected over FedYogi and FedAdagrad; about `83%` server accuracy. |
| FedLT | Adam local solver, no server regularizer | `step_size = 0.005`, `num_local_epochs = 8` | `rho = 1.0`, `local_solver = "adam"`, `solver_args = {"beta1": 0.5, "beta2": 0.999, "epsilon": 1e-8}` | Selected over GD and Nesterov; stable final curve; about `81%` server accuracy. A scaled L2 server regularizer was tested as a diagnostic but not selected as it did not improve the primary server-accuracy/loss criterion. |
| FedDyn | FedDyn | `step_size = 0.02760842017693185`, `num_local_epochs = 2` | `alpha = 1.0` | Stable final curve; about `83%` server accuracy. |
| FedPD | FedPD, full participation | `step_size = 0.03`, `num_local_epochs = 5` | `eta = 0.3`, `skip_probability = 0.2` | Selected from the focused 36-candidate grid after earlier FedPD candidates showed instability; stable final curve; about `82%` server accuracy and `80%` average client accuracy. FedPD is tuned with full participation because the current implementation does not support partial client participation. |

### FedNova Variant Choice

The FedNova tuning compares several internal FedNova variants instead of treating FedNova as only one fixed algorithm.
This is motivated by the FedNova paper/code, where different optional mechanisms are evaluated separately:

| Variant in this benchmark | Local momentum / `beta` | Proximal term / `mu` | Server momentum / `gamma` | Motivation |
| --- | --- | --- | --- | --- |
| `plain` | off | off | off | Vanilla FedNova baseline. |
| `momentum` | on | off | off | Local momentum FedNova variant. |
| `prox` | off | on | off | FedNova-Prox/proximal variant. |
| `server_momentum` | off | off | on | Server-momentum FedNova variant. |
| `both_momentums` | on | off | on | Hybrid local + server momentum variant. |
| `all_three` | on | on | on | Extra variant added here to test all three optional mechanisms together. |

For the proximal FedNova variants, `mu` is restricted to the paper/code values:

```text
{0.0005, 0.001, 0.005, 0.01}
```

I found evidence for the first five variants in the paper/code. I added `all_three` as an extra benchmark variant because
it is a natural combination to test once the framework already supports the three mechanisms.

The selected FedNova configuration uses both local momentum and server momentum, but not the proximal term. This agrees
with the FedNova paper's experimental discussion: in their CIFAR-10 experiments, the local-momentum FedNova variant is
reported as the best individual mechanism, and combining local and server momentum performs even better. The FEMNIST
tuning result here follows the same qualitative pattern, with the `both_momentums` variant outperforming the plain,
proximal-only, server-momentum-only, and all-three variants in the accepted run.

## Experiment 5 Design

Communication impairment robustness.

Experiment 5 benchmarks federated algorithms under controlled communication impairments:

- client availability impairments via `UniformActivationRate` and `MarkovChainActivation`;
- communication compression via `TopK` and `StochasticQuantization`;
- message loss via `UniformDropRate`;
- a combined availability + compression + drop condition.

Fixed setup:

- FEMNIST CNN model;
- `100` clients;
- `UniformSelection(fraction_selected_clients=0.2)`;
- `3` trials;
- `1500` iterations;
- state snapshots every `150` iterations;
- checkpoint step `None`;
- batch size `32`;
- seed `20260524`;
- no message noise.

Planned conditions:

| Condition key | Description |
| --- | --- |
| `clean_baseline` | Always active clients, no compression, no drops. |
| `activation_uniform_low` | Uniform client activation with low activation probability, no compression, no drops. |
| `activation_uniform_high` | Uniform client activation with high activation probability, no compression, no drops. |
| `activation_markov_sticky_online` | Markov-chain activation with one inactive-to-active / active-to-inactive combination, no compression, no drops. |
| `activation_markov_bursty_offline` | Markov-chain activation with another inactive-to-active / active-to-inactive combination, no compression, no drops. |
| `compression_topk_low` | Low-`k` `TopK` compression, always active clients, no drops. |
| `compression_topk_high` | Higher-`k` `TopK` compression, always active clients, no drops. |
| `compression_qsgd_low` | `StochasticQuantization` with fewer levels, always active clients, no drops. |
| `compression_qsgd_high` | `StochasticQuantization` with more levels, always active clients, no drops. |
| `drops_uniform_low` | Low-rate uniform message drops, always active clients, no compression. |
| `drops_uniform_high` | Higher-rate uniform message drops, always active clients, no compression. |
| `combined_uniform_topk_drops` | Uniform activation + `TopK` compression + uniform message drops. |

Outputs should be saved under:

```text
experiments/femnist/checkpoints/experiment5/<condition>/run_<timestamp>/
```

For each condition, the runner saves:

- per-algorithm checkpoints and raw metrics;
- combined `results/` tables and individual metric plots;
- `annotated_plots/` with one metric per figure and a grey impairment label box;
- compressed `metric_computation.pkl.zst` for later post-processing.

The saved metric object is intended for follow-up calculations such as percentage accuracy drop relative to
`clean_baseline` for each algorithm and condition.
