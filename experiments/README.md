# Federated Learning Experiments

This directory contains the experimental setups used to benchmark and compare federated learning algorithms with
`decent-bench`.

The goal is to provide fair comparisons by keeping the dataset partitions, model, loss, evaluation metrics,
communication conditions, and experiment budget consistent across algorithms within each task. Each dataset has its
own folder containing the dataset handler, model definition, experiment scripts, inspection figures, selected
hyperparameters, results, and documentation.

Together, the two tasks cover the main federated learning settings supported by these experiments:

- **FEMNIST simulates a cross-device setting**, with many relatively small writer-based clients and partial client
  participation.
- **Fed-ISIC2019 represents a cross-silo setting**, with a small number of institutional clients, larger local
  datasets, and normally full participation among active centers.

This allows `decent-bench` to benchmark the same federated algorithms in both cross-device and cross-silo scenarios.

## Experimental Tasks

### FEMNIST

[`femnist/`](femnist/) contains a standard research-oriented federated learning task based on handwritten character
classification.

FEMNIST provides a controlled cross-device-style benchmark with natural writer-based client partitions. It is useful
for studying federated optimization, client heterogeneity, partial participation, aggregation strategies, and
communication impairments in a widely used experimental setting.

See [`femnist/README.md`](femnist/README.md) for the dataset setup and experiment documentation.

### Fed-ISIC2019

[`fedisic2019/`](fedisic2019/) contains a more realistic cross-silo federated learning task using real-world
dermoscopic images from six medical data centers.

Fed-ISIC2019 introduces substantial class imbalance, quantity skew, label-distribution skew, and center-specific
feature variation. It is used to evaluate the algorithms under a realistic medical image-classification workload and
to study their robustness to communication and availability impairments.

See [`fedisic2019/README.md`](fedisic2019/README.md) for the dataset, preprocessing, model, loss, tuning procedure, and
results.

## Directory Overview

```text
experiments/
  README.md
  femnist/
    README.md
    src/
    experiment*/
    ...
  fedisic2019/
    README.md
    src/
    experiment*/
    ...
```

Dataset-specific assumptions and results should be documented inside the corresponding dataset folder. This keeps the
experiment collection organized as one folder per federated task while allowing both tasks to share the same
`decent-bench` benchmarking framework.
