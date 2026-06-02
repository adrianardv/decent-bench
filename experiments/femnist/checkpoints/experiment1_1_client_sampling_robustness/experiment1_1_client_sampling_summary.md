# Experiment 1.1 Client-Sampling Robustness Summary

This file summarizes the completed Experiment 1.1 runs saved under `experiments/femnist/checkpoints/experiment1_1_client_sampling_robustness/`.

## Client Overlap Between Seeds

| Seed A | Seed B | Shared clients | Different clients |
| --- | --- | ---: | ---: |
| 2001 | 20260524 | 4 / 100 | 96 / 100 |
| 2001 | 20260525 | 3 / 100 | 97 / 100 |
| 2001 | 20260526 | 2 / 100 | 98 / 100 |
| 2001 | 20260537 | 6 / 100 | 94 / 100 |
| 20260524 | 20260525 | 1 / 100 | 99 / 100 |
| 20260524 | 20260526 | 3 / 100 | 97 / 100 |
| 20260524 | 20260537 | 5 / 100 | 95 / 100 |
| 20260525 | 20260526 | 2 / 100 | 98 / 100 |
| 20260525 | 20260537 | 9 / 100 | 91 / 100 |
| 20260526 | 20260537 | 3 / 100 | 97 / 100 |

## Design Confirmation From Metadata

| Seed folder | Client-selection seed | Train/test split seed | Train samples | Test samples | Classes covered | Missing classes | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| 2001 | 2001 | 20260524 | 17902 | 4477 | 62 / 62 | none | complete |
| 20260524 | 20260524 | 20260524 | 19554 | 4886 | 62 / 62 | none | complete |
| 20260525 | 20260525 | 20260524 | 17817 | 4452 | 62 / 62 | none | complete |
| 20260526 | 20260526 | 20260524 | 18915 | 4732 | 62 / 62 | none | complete |
| 20260537 | 20260537 | 20260524 | 18289 | 4573 | 62 / 62 | none | complete |

## Summary Across Seed Runs

| Seed | Train | Test | Mean train/client | Median train/client | Mean classes/client | Digit | Upper | Lower |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2001 | 17902 | 4477 | 179.0 | 142.5 | 55.3 | 51.4% | 26.1% | 22.5% |
| 20260524 | 19554 | 4886 | 195.5 | 142.0 | 55.2 | 45.7% | 28.8% | 25.4% |
| 20260525 | 17817 | 4452 | 178.2 | 140.0 | 55.2 | 50.6% | 26.8% | 22.7% |
| 20260526 | 18915 | 4732 | 189.2 | 142.0 | 55.0 | 48.8% | 27.6% | 23.6% |
| 20260537 | 18289 | 4573 | 182.9 | 141.5 | 54.9 | 49.8% | 27.4% | 22.8% |

## Dataset Distribution And Accuracy By Seed

This table combines selected-client distribution statistics with the final server accuracy obtained by each federated algorithm on that selected subset.

| Seed | Mean train/client | Mean test/client | Train std | Train CV | Train min-max | Test min-max | Mean classes/client | Class min-max | Digit | Upper | Lower | FedAvg server acc. | FedProx server acc. | SCAFFOLD server acc. | FedNova server acc. | FedAdam server acc. | FedLT server acc. | FedDyn server acc. |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: | --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- | --- |
| 2001 | 179.0 | 44.8 | 63.4 | 0.35 | 102-315 | 26-79 | 55.3 | 31-62 | 51.4% | 26.1% | 22.5% | 82.40% ± 2.55% | 82.48% ± 1.72% | 81.32% ± 0.58% | 82.05% ± 0.53% | 81.83% ± 0.90% | 77.87% ± 2.33% | 81.69% ± 0.50% |
| 20260524 | 195.5 | 48.9 | 78.2 | 0.40 | 100-335 | 25-84 | 55.2 | 30-62 | 45.7% | 28.8% | 25.4% | 82.13% ± 3.14% | 83.59% ± 1.66% | 81.38% ± 0.51% | 82.62% ± 0.64% | 81.99% ± 1.55% | 80.11% ± 0.65% | 82.06% ± 0.43% |
| 20260525 | 178.2 | 44.5 | 68.2 | 0.38 | 101-357 | 25-89 | 55.2 | 40-62 | 50.6% | 26.8% | 22.7% | 82.07% ± 2.21% | 82.57% ± 2.22% | 81.84% ± 0.49% | 81.51% ± 0.58% | 81.72% ± 1.14% | 75.39% ± 1.47% | 81.90% ± 0.10% |
| 20260526 | 189.2 | 47.3 | 78.3 | 0.41 | 104-466 | 26-117 | 55.0 | 29-62 | 48.8% | 27.6% | 23.6% | 83.02% ± 1.67% | 84.13% ± 0.47% | 81.47% ± 0.64% | 83.84% ± 1.29% | 82.52% ± 1.70% | 81.03% ± 0.64% | 83.53% ± 1.36% |
| 20260537 | 182.9 | 45.7 | 66.2 | 0.36 | 110-338 | 28-85 | 54.9 | 35-62 | 49.8% | 27.4% | 22.8% | 82.83% ± 1.70% | 83.58% ± 0.39% | 82.15% ± 0.71% | 83.40% ± 0.47% | 81.83% ± 2.13% | 80.76% ± 1.14% | 82.83% ± 0.33% |

## Average Client Accuracy By Seed

| Seed | FedAvg avg acc. | FedProx avg acc. | SCAFFOLD avg acc. | FedNova avg acc. | FedAdam avg acc. | FedLT avg acc. | FedDyn avg acc. |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 2001 | 79.53% ± 0.28% | 79.47% ± 0.33% | 80.13% ± 0.16% | 81.35% ± 0.27% | 79.99% ± 0.21% | 73.15% ± 0.54% | 81.45% ± 0.32% |
| 20260524 | 79.58% ± 0.17% | 79.70% ± 0.30% | 80.47% ± 0.33% | 81.38% ± 0.17% | 80.03% ± 0.62% | 73.85% ± 0.34% | 81.71% ± 0.60% |
| 20260525 | 78.96% ± 0.30% | 79.08% ± 0.43% | 80.70% ± 0.22% | 81.10% ± 0.22% | 79.62% ± 0.40% | 74.27% ± 0.60% | 81.63% ± 0.25% |
| 20260526 | 80.10% ± 0.29% | 80.32% ± 0.60% | 80.34% ± 0.36% | 82.47% ± 0.89% | 80.64% ± 0.67% | 74.76% ± 0.17% | 83.13% ± 0.99% |
| 20260537 | 79.83% ± 0.12% | 79.66% ± 0.12% | 81.14% ± 0.18% | 82.17% ± 0.44% | 80.02% ± 0.35% | 74.53% ± 0.15% | 82.52% ± 0.19% |

## Conclusions

- The client-selection seed changed the selected writer set substantially: pairwise overlap is low, from 1 to 9 shared clients out of 100 for the planned thesis seeds, and 2 to 6 shared clients when comparing the extra `2001` seed against the planned seeds.
- The selected subsets remain comparable: each contains 100 unique clients, covers all 62 FEMNIST classes, and has a similar median number of train samples per client and mean number of classes per client.
- The original seed `20260524` has the largest total sample count and is slightly less digit-heavy than the other sampled subsets, but the differences are moderate rather than a change to a qualitatively different dataset.
- The clean-baseline algorithm conclusions are broadly stable across selected-client samples: FedAvg, FedProx, SCAFFOLD, FedNova, FedAdam, and FedDyn remain in a similar accuracy band, while FedLT remains lower in average client accuracy under this configuration.

