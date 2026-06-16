# Fed-ISIC2019 Experiments Log

## Dataset

- Dataset name: Fed-ISIC2019.
- Source used in this repository: Flower Labs Hugging Face dataset `flwrlabs/fed-isic2019`.
- Upstream recipe inspected: FLamby `flamby/datasets/fed_isic2019`.
- Task: 8-class dermoscopic image classification.
- Federated clients/sites: 6 centers.
- FLamby reported dataset size: 23,247 images with a fixed center-stratified train/test split.

## Classes

Fed-ISIC2019 is an 8-class classification task. The class index order used by the handler is:

| Index | Label | Meaning |
| --- | --- | --- |
| 0 | `MEL` | Melanoma |
| 1 | `NV` | Melanocytic nevus |
| 2 | `BCC` | Basal cell carcinoma |
| 3 | `AK` | Actinic keratosis / Bowen's disease-type intraepithelial carcinoma category used in ISIC-style lesion classification |
| 4 | `BKL` | Benign keratosis-like lesion, including solar lentigo, seborrheic keratosis, and lichen planus-like keratosis |
| 5 | `DF` | Dermatofibroma |
| 6 | `VASC` | Vascular lesion |
| 7 | `SCC` | Squamous cell carcinoma |

## Centers And Data Provenance

The six client/site IDs follow the FLamby center order. The source provenance below is derived from the FLamby Fed-ISIC2019 README and the HAM10000/ISIC source descriptions.

| Center ID | Center key | Samples | Provenance |
| --- | --- | ---: | --- |
| 0 | `BCN` | 12,413 | BCN_20000 Dataset: Department of Dermatology, Hospital Clinic de Barcelona. |
| 1 | `HAM_vidir_molemax` | 3,954 | HAM10000 Dataset subset from the ViDIR Group, Department of Dermatology, Medical University of Vienna; MoleMax acquisition source. |
| 2 | `HAM_vidir_modern` | 3,363 | HAM10000 Dataset subset from the ViDIR Group, Department of Dermatology, Medical University of Vienna; modern dermatoscopy acquisition source. |
| 3 | `HAM_rosendahl` | 2,259 | HAM10000 Dataset subset associated with Cliff Rosendahl's skin cancer practice source used in HAM10000. |
| 4 | `MSK` | 819 | MSK Dataset, used through the ISIC challenge sources and cited by FLamby as `(c) Anonymous`. |
| 5 | `HAM_vienna_dias` | 439 | HAM10000 Dataset subset from the ViDIR Group, Department of Dermatology, Medical University of Vienna; Vienna DIAS acquisition source. |

## Non-IID Characterization

Fed-ISIC2019 is a cross-silo dataset with natural center-based partitions, so the client data is non-IID in several ways:

- Quantity skew: the number of samples per client is very different. In the official train split, `BCN` has 9,930 samples while `HAM_vienna_dias` has 351 samples, a largest/smallest ratio of about `28.3x`. This is large for a six-client cross-silo setup and is the reason Experiment 2 compares uniform aggregation with data-size weighted aggregation.
- Label distribution skew: class proportions differ strongly across clients. For example, `HAM_vidir_molemax` is dominated by `NV` with 3,720 of 3,954 total samples, while several classes have zero or near-zero samples in some centers. `MSK` has no `BCC`, `AK`, `DF`, `VASC`, or `SCC` samples in the inspected full split. This means client updates can be biased toward different lesion-type mixtures.
- Global class imbalance: the dataset itself is imbalanced. Across the full inspected split, `NV` has 11,326 samples while `DF` has 239 samples, about `47.4x` more. In the train split, `NV` has 9,084 samples while `DF` has 184 samples, about `49.4x` more. This motivates balanced accuracy for evaluation and weighted focal loss for training.
- Feature/site skew: each client corresponds to a data source or acquisition site. Differences in hospitals, acquisition devices, acquisition protocols, patient populations, and image characteristics can create covariate shift across clients, even when labels overlap.

Overall, Fed-ISIC2019 should be treated as strongly non-IID. The largest directly measured skews are the client quantity skew (`28.3x` train largest/smallest center ratio) and the class imbalance (`49.4x` train largest/smallest class ratio). The center-by-class distribution plot in `experiments/fedisic2019/figures/client_class_distribution.png` visualizes the label skew.

## References And Citation

- FLamby Fed-ISIC2019 code and README: https://github.com/owkin/FLamby/tree/main/flamby/datasets/fed_isic2019
- Flower/Hugging Face dataset: https://huggingface.co/datasets/flwrlabs/fed-isic2019
- ISIC 2019 challenge data: https://challenge.isic-archive.com/landing/2019/
- HAM10000: Tschandl P., Rosendahl C. and Kittler H. The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. Scientific Data 5, 180161 (2018). https://doi.org/10.1038/sdata.2018.161
- ISIC challenge references listed by FLamby include Codella et al. 2017 and Combalia et al. 2019.

## Licence

- FLamby states that ISIC 2019 data is distributed under CC-BY-NC 4.0.
- HAM10000 metadata/data terms also apply; users should accept and follow the ISIC and HAM10000 terms before using the data.
- The code added here is repository code; the dataset itself is not vendored.

## Source Choice

I selected `flwrlabs/fed-isic2019` instead of directly downloading through FLamby because it fits the existing FEMNIST pattern: Hugging Face loading, cache control, and lightweight metadata inspection. It also avoids copying FLamby's dataset creation scripts into this repository. The implementation still follows FLamby's center split, runtime crops, balanced accuracy metric, EfficientNet-B0 model family, and weighted focal loss.

## Preprocessing

The Hugging Face dataset card exposes the image, center, and label columns and explicitly says that, to keep the same results as FLamby, users should apply the albumentations train/test transforms listed below. This handler does apply those runtime transforms.

What is done by this repository:

- load the RGB image from `flwrlabs/fed-isic2019`;
- apply the Flower/Hugging Face dataset-card transform recipe for train or test;
- crop images to 200 x 200 for batching;
- normalize;
- transpose to `C x H x W`;
- cast to `float32`;
- return integer labels as `torch.long`.

Albumentations is the image preprocessing and augmentation library used by the Flower/Hugging Face Fed-ISIC2019
recipe. The handler uses it because it provides the same computer-vision transforms used in the dataset-card example,
works directly on NumPy image arrays, and cleanly separates stochastic training augmentation from deterministic
evaluation preprocessing.

Why augmentation is useful here: Fed-ISIC2019 contains medical skin-lesion images from several acquisition centers.
The relevant class signal should be robust to nuisance variation such as lesion position in the crop, camera angle,
scale, local brightness/contrast, and small occlusions or artifacts. Random train-time augmentation exposes the model
to these variations and helps reduce overfitting to center-specific or image-specific artifacts. Evaluation is kept
deterministic so reported metrics are stable and comparable across algorithms.

The preprocessing also converts the dataset into the format expected by PyTorch classification models:

- images are converted to RGB arrays; grayscale inputs would be expanded to three channels and RGBA inputs would drop
  the alpha channel;
- Albumentations receives images as `H x W x C` NumPy arrays;
- after augmentation, images are transposed to PyTorch's `C x H x W` layout;
- images are cast to `float32`;
- labels are returned as integer class indices compatible with `torch.nn` classification losses.

What is assumed about the source:

- the source dataset is derived from FLamby and stores RGB images whose shorter edge is 224 px in the local copy inspected on 2026-06-13;
- FLamby's offline color-constancy preprocessing is treated as part of the packaged source preparation if Flower generated the dataset from FLamby's preprocessed images.

Important limitation: I did not independently verify from image pixels or source generation metadata that color constancy has been applied in the Hugging Face artifacts. The handler does not implement color constancy itself. If exact FLamby pixel-level reproduction is required, this should be verified against the FLamby-created `ISIC_2019_Training_Input_preprocessed` files or by using FLamby's dataset creation scripts directly.

Image-size check from `experiments/fedisic2019/inspect_fedisic2019.py` on 2026-06-13:

- `n_rows`: 23,247
- unique image sizes: 13
- minimum width: 224
- minimum height: 224
- minimum shorter edge: 224
- most common sizes: `224 x 224`, `298 x 224`, `337 x 224`, `334 x 224`, `336 x 224`
- conclusion: the images are not all square, but the shorter edge is 224 px, so FLamby's 200 x 200 random/center crop recipe is safe for this Hugging Face copy.

Training transform:

- `RandomScale(0.07)`
- `Rotate(50)`
- `RandomBrightnessContrast(0.15, 0.1)`
- `Flip(p=0.5)` when available; for albumentations 2.x this is approximated with a compatible horizontal/vertical flip wrapper because `Flip` was removed from the public API
- `Affine(shear=0.1)`
- `RandomCrop(200, 200)`
- `CoarseDropout` with 1 to 8 holes of size 16, using a compatibility wrapper for albumentations 1.x/2.x APIs
- `Normalize`
- transpose to `C x H x W` and cast to `float32`

Training transform purpose:

- `RandomScale(0.07)`: randomly zooms the image slightly in or out, making the model less sensitive to lesion scale.
- `Rotate(50)`: rotates images up to a wide angle range, which is appropriate because lesion diagnosis should not
  depend on camera orientation.
- `RandomBrightnessContrast(0.15, 0.1)`: perturbs illumination and contrast, helping robustness to acquisition and
  lighting differences across centers.
- `Flip(p=0.5)`: randomly mirrors images; lesion class should generally be invariant to horizontal or vertical
  orientation.
- `Affine(shear=0.1)`: applies a small geometric shear, adding mild viewpoint/geometry variation.
- `RandomCrop(200, 200)`: extracts a fixed-size crop for batching while adding random spatial variation during
  training.
- `CoarseDropout`: masks small square regions, encouraging the model not to rely too strongly on a tiny image artifact
  or local patch.
- `Normalize`: converts pixel intensities to normalized floating-point values using Albumentations defaults, matching
  the Flower/Hugging Face recipe and improving optimizer behavior.

Evaluation/ test transform:

- `CenterCrop(200, 200)`
- `Normalize`
- transpose to `C x H x W` and cast to `float32`

Evaluation transform purpose:

- `CenterCrop(200, 200)`: creates the same fixed spatial size as training without adding randomness.
- `Normalize`: applies the same normalization convention used for training.
- no random scale, rotation, brightness/contrast, flip, affine, random crop, or dropout is applied during evaluation,
  because validation/test metrics should measure the model on a stable deterministic version of each image.

Deviation from a pure Hugging Face/Flower pipeline: the dataset card demonstrates `with_transform(...)` on Flower partitions. `decent-bench` empirical-risk costs consume sequence-like datasets instead, so the Fed-ISIC partitions are lazy sequence objects that apply the same transform recipe on item access. If `PyTorchCost(load_dataset=True)` is enabled, those transformed tensors are materialized; the default experiment0 setting keeps `load_dataset=False` to avoid loading many GB of image tensors into memory. Since the algorithm repeatedly indexes the same lazy sequence, stochastic training augmentations are resampled when the partition is accessed rather than precomputed by the handler.

## Dataset Handler

Added `FedISICDatasetHandler` under `experiments/fedisic2019/src/fedisic_handler.py`.

Behavior:

- loads `train` or `test` from `flwrlabs/fed-isic2019`;
- detects the center and label columns from the dataset metadata;
- validates selected centers against the source metadata;
- returns one lazy partition per center/site;
- returns labels as integer `torch.long` tensors;
- returns images as normalized `torch.float32` tensors in `C x H x W` layout.

## Model

Main model: `FedISIC2019EfficientNet`.

- Uses `torchvision.models.efficientnet_b0`, already covered by this repository's dev dependency set.
- Replaces the final classifier layer with an output dimension equal to the number of classes detected from the dataset.
- Defaults to ImageNet pretrained weights to match the FLamby baseline intent. Use `--no-pretrained` for offline runs if torchvision weights are not cached.

Debug model: `FedISICSmallCNN`.

- Only intended for smoke tests and fast plumbing checks.
- It must not be treated as the main benchmark model.

## Loss

Main loss: weighted focal loss with `gamma=2.0`.

- The default alpha vector is copied from FLamby's Fed-ISIC2019 `BaselineLoss`:
  `[5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]`.
- `experiment0` defaults to these FLamby alpha values.
- Optional mode `--class-weight-mode computed` computes `N / n_c` class weights from the training split labels, matching the class-weight computation shown in FLamby's pooled benchmark script.

Weighted focal loss is a loss function used mainly for classification with class imbalance, especially when some
classes are rare and many examples are easy to classify. It extends cross-entropy loss by adding two ideas:

- Class weighting: gives more importance to under-represented or more important classes.
- Focal modulation: reduces the loss contribution from easy examples and focuses training on hard misclassified examples.

For multi-class classification, it is often written as:

```text
FL = - alpha_y * (1 - p_y)^gamma * log(p_y)
```

where `p_y` is the predicted probability of the true class `y`, and `alpha_y` is that class's weight. In practice,
weighted focal loss is useful for tasks such as fraud detection, medical diagnosis, object detection, rare-event
classification, and segmentation, where the majority class can otherwise dominate training.

For logits `z_i` and class target `y_i`, the implemented weighted focal loss is:

```text
log p_i = log_softmax(z_i)[y_i]
p_i = exp(log p_i)
L_i = - alpha[y_i] * (1 - p_i)^gamma * log p_i
L = mean_i L_i
```

`gamma=2.0` downweights examples that are already predicted with high confidence. `alpha[y_i]` upweights rare lesion classes, which is important for Fed-ISIC2019's class imbalance.

## Metrics

Added balanced accuracy to `decent-bench`:

- `BalancedAccuracy`
- `ServerBalancedAccuracy`

Both use `sklearn.metrics.balanced_accuracy_score`. Fed-ISIC2019 tuning uses `server balanced accuracy` as the primary selection metric and validation loss as the tie-breaker.

## Experiment 0

Added `experiments/fedisic2019/experiment0/experiment0.py`.

Setup:

- natural center partitions;
- full client participation;
- no client drops;
- no message noise;
- no compression;
- clients always active;
- batch size 64 by default;
- balanced accuracy for tuning.

Search strategy:

- reference candidates from FLamby-style values where available;
- FEMNIST selected values seed algorithms without FLamby Fed-ISIC references;
- random coarse search over compact ranges;
- focused grid around the best coarse candidates.

Initial reference selections are recorded in:

- `experiments/fedisic2019/selected_hyperparameters.json`

Full tuning writes run-specific best files under:

- `experiments/fedisic2019/checkpoints/experiment0/<algorithm>/<run>/exp0_best_hyperparameters.json`

## Experiment 0 Tuning Results

The finalized selected hyperparameters are stored in:

- `experiments/fedisic2019/selected_hyperparameters.json`

Selection criterion:

- primary metric: validation `server balanced accuracy`;
- tie-breaker: validation loss;
- tuning trials: `1`;
- tuning split: `80%` of the official train split for tuning train and `20%` for validation;
- clean baseline: full participation, always active clients, no drops, no noise, no compression.

Final selected values:

| Algorithm | Selected variant | Hyperparameters |
| --- | --- | --- |
| FedAvg | FedAvg | `step_size=0.02`, `num_local_epochs=4` |
| FedProx | FedProx | `step_size=0.016013056680630116`, `num_local_epochs=4`, `mu=0.00951105281010339` |
| SCAFFOLD | SCAFFOLD | `step_size=0.02`, `num_local_epochs=5`, `server_step_size=0.875` |
| FedNova | server momentum only | `step_size=0.007313389878312145`, `num_local_epochs=3`, `use_momentum=false`, `use_server_momentum=true`, `use_prox=false`, `gamma=0.5` |
| FedOpt family | FedAdam | `step_size=0.008554469418165058`, `num_local_epochs=4`, `server_step_size=0.0071790218875614565`, `beta_1=0.9`, `beta_2=0.9`, `tau=0.001` |
| FedLT | Adam local solver | `step_size=0.0008`, `num_local_epochs=3`, `rho=0.05`, `local_solver="adam"`, `solver_args={"beta1": 0.5, "beta2": 0.999, "epsilon": 1e-8}` |
| FedDyn | FedDyn green diagnostic candidate | `step_size=0.016013056680630116`, `num_local_epochs=3`, `alpha=0.33075447277711245` |
| FedPD | FedPD full participation | `step_size=0.03`, `num_local_epochs=5`, `eta=0.3`, `skip_probability=0.2` |

Additional tuning decisions:

- FedOpt is represented by one final `fedopt` entry selecting `FedAdam`; the earlier separate `fedadam`, `fedyogi`,
  and `fedadagrad` reference entries were removed from the final selected file.
- FedLT was refined after the first tuning run. The original conservative candidate was smooth but slow, the retuned
  faster candidate had stronger final performance but less stable early dynamics, and the final selected intermediate
  candidate (`step_size=0.0008`, `num_local_epochs=3`, `rho=0.05`) gave the best compromise between balanced accuracy,
  loss, and curve stability.
- FedDyn remained under-tuned under the available tuning budget. The first tuning run selected a very low
  learning-rate/alpha candidate whose curve stayed nearly flat until a late jump. A later diagnostic compared the
  better-looking alternatives, and the selected green candidate was the most plausible because it improved earlier and
  reduced loss more consistently. However, even the 1000-iteration green-candidate diagnostic still showed a long flat
  phase, unstable server accuracy, and only late improvement in server balanced accuracy. FedDyn results should
  therefore be interpreted cautiously as a likely under-tuned/sensitive baseline rather than as a fully optimized
  FedDyn configuration.

## Experiment 0 Retuning

Added `experiments/fedisic2019/experiment0/experiment0_retune.py`.

This is a compact follow-up to `experiment0` for algorithms whose final validation curves were suspicious or likely
under-tuned after the first full run:

- FedDyn: first run selected a very low learning rate/alpha and the final curve stayed nearly flat until a late jump.
- FedProx: first run selected a high learning rate/proximal coefficient and the final balanced accuracy curve jumped
  late rather than improving smoothly.
- FedLT: first run selected a low `rho` and low learning rate; retuning compares that current best against more
  standard `rho` values around `0.01`, `0.1`, and `1.0`.

The retuning script uses the same dataset split, model, weighted focal loss, metrics, clean communication assumptions,
and final-curve evaluation as `experiment0`, but replaces broad random/grid search with fixed candidate lists of at
most six candidates per algorithm. FedProx and FedLT include their current best `experiment0` candidate so the retune
can directly compare new candidates against the previous selection.

Retuning writes run-specific best files under:

- `experiments/fedisic2019/checkpoints/experiment0/<algorithm>/<run>/exp0_retune_best_hyperparameters.json`

## Experiment 2

Added `experiments/fedisic2019/experiment2/experiment2_aggregation_weighting.py`.

Purpose: compare uniform client aggregation against data-size weighted aggregation for each tuned Fed-ISIC2019
algorithm under a clean cross-silo baseline. Fed-ISIC2019 has substantially stronger quantity skew than the FEMNIST
subsets used in earlier experiments, so weighting by client sample count may have a larger effect.

Configuration:

- condition: clean baseline;
- official Fed-ISIC2019 train split for training;
- official Fed-ISIC2019 test split for evaluation;
- clients: all six natural centers;
- client activation: `AlwaysActive`;
- client selection scheme: `None`, so all active clients participate;
- communication: `NoDrops`, `NoNoise`, `NoCompression`;
- iterations: 2500;
- trials: 3;
- state snapshot period: 250;
- selected hyperparameters: `experiments/fedisic2019/selected_hyperparameters.json`;
- table metrics: default `decent-bench` metrics after availability filtering;
- plot metrics: default `decent-bench` plot metrics after availability filtering;
- plots are saved individually.

Aggregation variants:

- uniform aggregation: each received client upload has equal aggregation weight;
- data-size weighted aggregation: each received client upload is weighted by its local training sample count.

Quantity skew in the official Fed-ISIC2019 train split:

| Center | Train samples |
| --- | ---: |
| BCN | 9930 |
| HAM_vidir_molemax | 3163 |
| HAM_vidir_modern | 2691 |
| HAM_rosendahl | 1807 |
| MSK | 655 |
| HAM_vienna_dias | 351 |

The largest train center (`BCN`) has about `28.3x` more samples than the smallest train center
(`HAM_vienna_dias`).

Experiment 2 writes results under:

- `experiments/fedisic2019/checkpoints/experiment2/<algorithm>/<run>/`

## Experiment 5

Added `experiments/fedisic2019/experiment5/experiment5_communication_impairments.py`.

Purpose: evaluate robustness of the tuned Fed-ISIC2019 algorithms under cross-silo communication and availability
impairments. This experiment uses the official Fed-ISIC2019 train split for training and the official test split for
evaluation.

Configuration:

- iterations: 2500;
- trials: 3;
- state snapshot period: 250;
- batch size: 64;
- model/loss/data preprocessing: same as Experiment 0;
- selected hyperparameters: `experiments/fedisic2019/selected_hyperparameters.json`;
- client selection scheme: `None`, so all active clients participate;
- table metrics: default `decent-bench` metrics after availability filtering;
- plot metrics: default `decent-bench` plot metrics after availability filtering;
- plots are saved individually, plus condition-annotated versions under `annotated_plots`.

Conditions:

- `clean_baseline`: always active, full participation, no noise, no compression, no drops.
- `availability`: `UniformActivationRate(0.80)`.
- `compression`: `TopK(0.10)`.
- `drops`: `UniformDropRate(0.20)`.
- `noise`: `GaussianNoise(mean=0.0, std=0.001)`.
- `combination`: `UniformActivationRate(0.80)`, `TopK(0.10)`, `UniformDropRate(0.20)`, and
  `GaussianNoise(mean=0.0, std=0.001)`.

Cross-silo assumption: in cross-silo settings like Fed-ISIC2019, the default tendency is full participation because
there are few institutional clients and coordination is more controlled than in cross-device FL. Therefore this
experiment does not apply any explicit client selection scheme; if clients are active, they participate.

Availability assumption: in cross-silo FL, clients are generally assumed to be available and stable. Nevertheless,
temporary unavailability can occur due to maintenance windows, system failures, connectivity incidents, or local
operational constraints. For that reason, this experiment tests availability as an impairment, but uses a high
availability probability (`0.80`) because high availability is normally expected in cross-silo deployments.

Experiment 5 writes results under:

- `experiments/fedisic2019/checkpoints/experiment5/<condition>/<run>/`

## Reduced-Budget Pilot Experiments

Added:

- `experiments/fedisic2019/experiment5/experiment5_reduced_pilot.py`
- `experiments/fedisic2019/experiment2/experiment2_reduced_pilot.py`

Purpose: provide diagnostic results when the full-data Experiment 2 and Experiment 5 runs are too
slow to finish. These reduced pilots are useful for presentations and sanity checks, but they should be labelled as
reduced-budget pilot results, not as the final full Fed-ISIC2019 benchmark.

Shared reduced dataset configuration:

- official Fed-ISIC2019 train split sampled for training;
- official Fed-ISIC2019 test split sampled for evaluation;
- per-center sampling rate: `10%` of each center/split, rounded up with `ceil`;
- minimum requested samples per center/split: `100`;
- if a center/split has fewer than `100` samples available, all available samples are kept;
- sampling: deterministic stratified sampling within each center by class, using seed `20260524`;
- model: same ImageNet-pretrained EfficientNet-B0 with an 8-class classifier head;
- loss: same FLamby-weighted focal loss;
- batch size: `64`;
- iterations: `1000`;
- trials: `2`;
- state snapshot period: `200`.

The reduced dataset intentionally preserves quantity skew across centers.The current policy
keeps roughly `10%` of large centers and uses the `100`-sample minimum only for smaller center/split combinations.
With the current official split this gives `1961` training samples and `737` test samples.

Reduced subset selection rule:

- sampling is applied separately to the official train split and the official test split, so the official train/test
  semantics are preserved;
- within each split, sampling is done independently for each center/site;
- for each center/split, the requested sample count is `ceil(0.10 * n_center_split_samples)`;
- if that requested count is below `100`, the requested count is raised to `100`;
- if fewer than the requested number are available, all available samples are kept;
- samples are drawn without replacement using deterministic stratified sampling by class;
- the stratified sampler allocates class quotas approximately proportional to the center's original class
  distribution in that split, while preserving at least one example of each present class when the cap allows it;
- seed: `20260524`.

The comparison below uses train and test combined unless the table explicitly separates them.

Total sample count:

| Dataset | Train | Test | Total |
|:---|---:|---:|---:|
| Original full dataset | 18597 | 4650 | 23247 |
| Reduced pilot subset | 1961 | 737 | 2698 |

Samples per center:

| Center | Original samples | Reduced samples | Retained % |
|:---|---:|---:|---:|
| BCN | 12413 | 1242 | 10.0% |
| HAM_vidir_molemax | 3954 | 417 | 10.5% |
| HAM_vidir_modern | 3363 | 370 | 11.0% |
| HAM_rosendahl | 2259 | 281 | 12.4% |
| MSK | 819 | 200 | 24.4% |
| HAM_vienna_dias | 439 | 188 | 42.8% |

Samples and total-dataset percentage per class:

| Class | Original samples | Reduced samples | Original total % | Reduced total % |
|:---|---:|---:|---:|---:|
| MEL | 4185 | 484 | 18.00% | 17.94% |
| NV | 11326 | 1332 | 48.72% | 49.37% |
| BCC | 3323 | 342 | 14.29% | 12.68% |
| AK | 867 | 99 | 3.73% | 3.67% |
| BKL | 2426 | 299 | 10.44% | 11.08% |
| DF | 239 | 38 | 1.03% | 1.41% |
| VASC | 253 | 32 | 1.09% | 1.19% |
| SCC | 628 | 72 | 2.70% | 2.67% |

Percentage of each class inside each center, shown as `original -> reduced`:

| Center | MEL | NV | BCC | AK | BKL | DF | VASC | SCC |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| BCN | 23.0% -> 22.9% | 33.9% -> 33.5% | 22.6% -> 22.4% | 5.9% -> 6.0% | 9.2% -> 9.5% | 1.0% -> 1.0% | 0.9% -> 1.0% | 3.5% -> 3.8% |
| HAM_vidir_molemax | 0.6% -> 1.4% | 94.1% -> 91.8% | 0.1% -> 0.2% | 0.0% -> 0.0% | 3.1% -> 3.4% | 0.8% -> 1.4% | 1.4% -> 1.7% | 0.0% -> 0.0% |
| HAM_vidir_modern | 20.2% -> 19.7% | 54.5% -> 52.7% | 6.3% -> 6.2% | 0.6% -> 2.2% | 14.1% -> 13.8% | 1.5% -> 2.2% | 2.4% -> 2.7% | 0.3% -> 0.5% |
| HAM_rosendahl | 15.1% -> 14.6% | 35.5% -> 33.8% | 13.1% -> 13.5% | 4.8% -> 6.0% | 21.7% -> 20.3% | 1.3% -> 3.2% | 0.1% -> 0.4% | 8.2% -> 8.2% |
| MSK | 26.3% -> 27.5% | 50.7% -> 46.0% | 0.0% -> 0.0% | 0.0% -> 0.0% | 23.1% -> 26.5% | 0.0% -> 0.0% | 0.0% -> 0.0% | 0.0% -> 0.0% |
| HAM_vienna_dias | 15.3% -> 13.3% | 79.7% -> 80.3% | 1.1% -> 1.1% | 0.0% -> 0.0% | 2.3% -> 3.2% | 0.9% -> 1.1% | 0.7% -> 1.1% | 0.0% -> 0.0% |

Reduced-pilot inspection outputs generated under `experiments/fedisic2019/figures/reduced_pilot/`:

- `reduced_dataset_class_distribution.png`
- `reduced_dataset_client_class_distribution.png`
- `reduced_dataset_class_counts.csv`
- `reduced_dataset_center_class_counts.csv`
- `reduced_dataset_metadata.csv`

Reduced run metadata points to the shared inspection outputs above. The experiment scripts do not regenerate the
reduced-dataset figures inside every checkpoint directory, because the reduced subset is deterministic and the shared
figures already document it.

Reduced Experiment 5 configuration:

- conditions: `clean_baseline` and `combination`;
- algorithms: FedAvg, FedLT, FedAdam through the final selected `fedopt` entry, FedPD, and SCAFFOLD;
- clean baseline: always active, full participation, no noise, no compression, no drops;
- combination: `UniformActivationRate(0.80)`, `TopK(0.10)`, `UniformDropRate(0.20)`, and
  `GaussianNoise(mean=0.0, std=0.001)`;
- table metrics: default `decent-bench` metrics;
- plot metrics: default `decent-bench` plot metrics;
- condition-annotated plots are saved under `annotated_plots`.

Reduced Experiment 5 writes results under:

- `experiments/fedisic2019/checkpoints/experiment5_reduced_pilot/<condition>/<run>/`

Reduced Experiment 2 configuration:

- algorithm: FedAvg only;
- variants: uniform aggregation versus data-size weighted aggregation;
- condition: clean baseline with all clients active and full participation;
- table metrics: default `decent-bench` metrics;
- plot metrics: default `decent-bench` plot metrics.

Reduced Experiment 2 writes results under:

- `experiments/fedisic2019/checkpoints/experiment2_reduced_pilot/fedavg/<run>/`

## Results Obtained So Far

Result labels used below:

- Thesis relevant: suitable for the thesis only if clearly labelled with its scope and limitations.
- Diagnostic only: useful for engineering decisions, sanity checks, or explaining limitations, but not suitable as a
  headline thesis result.
- Not thesis relevant: should not be included as an experimental result in the thesis.

### Reduced Experiment 5: Clean Baseline vs Combined Impairments

Thesis relevance: thesis relevant as a reduced-budget pilot result. These results can support the presentation/thesis
claim that the Fed-ISIC2019 setup trains under clean communication and that combined communication impairments degrade
or destabilize several methods. They must not be presented as the final full-data Fed-ISIC2019 benchmark, because they
use the reduced stratified subset, `1000` iterations, and `2` trials.

Result paths:

- clean baseline:
  `experiments/fedisic2019/checkpoints/experiment5_reduced_pilot/clean_baseline/run_20260615_154644/`
- combined impairments:
  `experiments/fedisic2019/checkpoints/experiment5_reduced_pilot/combination/run_20260615_154716/`

Reference metric: server balanced accuracy.

Clean baseline final server metrics:

| Algorithm | Server balanced accuracy | Server accuracy |
| --- | ---: | ---: |
| FedAvg | 49.35% +/- 3.90% | 64.25% +/- 6.03% |
| FedLT | 50.16% +/- 6.55% | 64.38% +/- 12.93% |
| FedAdam | 50.76% +/- 16.48% | 64.72% +/- 3.45% |
| FedPD | 47.00% +/- 1.81% | 60.92% +/- 1.72% |
| SCAFFOLD | 50.43% +/- 8.61% | 64.11% +/- 4.31% |

Combined impairment final server metrics:

| Algorithm | Server balanced accuracy | Server accuracy |
| --- | ---: | ---: |
| FedAvg | 43.57% +/- 10.42% | 53.39% +/- 31.89% |
| FedLT | 10.98% +/- 1.73% | 24.76% +/- 204.30% |
| FedAdam | 12.47% +/- 0.43% | 49.93% +/- 1.72% |
| FedPD | 30.06% +/- 44.68% | 40.30% +/- 60.34% |
| SCAFFOLD | 11.64% +/- 0.01% | 18.05% +/- 37.93% |

Interpretation:

- Under the reduced clean baseline, all five tested algorithms train successfully and end in a narrow server balanced
  accuracy band of roughly `47%` to `51%`.
- Under the combined impairment condition, FedAvg is the most stable of the tested algorithms in this reduced pilot.
- FedPD has non-trivial balanced accuracy under the combined condition, but its uncertainty is very large, so it should
  not be interpreted as reliably better than the others from this pilot alone.
- FedLT, FedAdam, and SCAFFOLD collapse close to chance-level balanced accuracy under the combined condition.
- SCAFFOLD is stable in the clean reduced pilot but unstable under the combined impairment condition. The combined-run
  plots show very large client drift, so this is likely an impairment-interaction issue rather than a general
  implementation failure of SCAFFOLD.
- The loss plots under the combined condition are not a reliable ranking signal. Some methods show extreme loss values
  or degenerate-looking loss curves under impairment, while balanced accuracy remains the main reference metric for
  Fed-ISIC2019.

Recommended use:
- Include the clean-vs-combined reduced pilot only as a reduced-budget robustness pilot.
- State explicitly that it is preliminary and uses a smaller stratified subset for computational feasibility.
- Use server balanced accuracy plots as the main visual evidence.

### Reduced Experiment 2: Aggregation Weighting Pilot

Relevance: diagnostic only. This run is useful as a sanity check for uniform versus data-size weighted
aggregation on the reduced FedAvg pilot, but it should not be used as a thesis conclusion because it evaluates only one
algorithm and has high uncertainty.

Result path:

- `experiments/fedisic2019/checkpoints/experiment2_reduced_pilot/fedavg/run_20260615_154731/`

Final server metrics:

| Variant | Server balanced accuracy | Server accuracy |
| --- | ---: | ---: |
| FedAvg uniform | 50.72% +/- 19.97% | 63.98% +/- 0.86% |
| FedAvg data-size weighted | 47.49% +/- 13.01% | 62.21% +/- 23.27% |

Interpretation:

- The reduced pilot does not show a clear advantage for data-size weighted aggregation.
- The uncertainty is large, especially for server balanced accuracy.
- Because the run is reduced-budget and FedAvg-only, it should not be used as evidence that aggregation weighting does
  or does not matter on Fed-ISIC2019.

### Full Experiment 2: FedAvg Aggregation Weighting

This result should not be included in the thesis as an experimental conclusion.
It is a useful internal sanity check that the full-data FedAvg Experiment 2 run completed and that metrics can be
recovered from checkpoints, but it evaluates only FedAvg and does not show a meaningful difference between aggregation
variants.

Result path:

- `experiments/fedisic2019/checkpoints/experiment2/fedavg/run_20260615_131734/`

Final server metrics:

| Variant | Server balanced accuracy | Server accuracy |
| --- | ---: | ---: |
| FedAvg uniform | 74.98% +/- 0.78% | 79.68% +/- 1.02% |
| FedAvg data-size weighted | 74.78% +/- 4.71% | 79.95% +/- 3.90% |

Interpretation:

- Uniform and data-size weighted aggregation are essentially tied for FedAvg in this full-data run.
- The data-size weighted variant has a much larger margin of error for server balanced accuracy.
- Since this is only FedAvg and not the complete Experiment 2 algorithm set, it should not be used for the thesis
  comparison of aggregation weighting.

### Experiment 0 Diagnostic Curves

Thesis relevance: diagnostic only, except for the selected hyperparameters and the FedDyn limitation note. The
diagnostic curves explain why the final selected hyperparameters were chosen and why FedDyn should be interpreted
cautiously.

FedLT diagnostics:

| Candidate | Server balanced accuracy | Server accuracy | Average loss |
| --- | ---: | ---: | ---: |
| `step_size=0.0008`, `num_local_epochs=3`, `rho=0.05` | 69.03% | 71.23% | 0.3548 |
| `step_size=0.001`, `num_local_epochs=3`, `rho=0.05` | 67.87% | 71.31% | 0.4403 |

Interpretation: the `0.0008` FedLT candidate was kept because it gave slightly higher server balanced accuracy and a
lower loss than the otherwise similar `0.001` candidate. The result remains a tuning diagnostic rather than a thesis
benchmark result.

FedDyn diagnostics:

| Candidate | Server balanced accuracy | Server accuracy | Average loss |
| --- | ---: | ---: | ---: |
| `step_size=0.001`, `num_local_epochs=4`, `alpha=0.01` | 52.13% | 47.73% | 2.4062 |
| `step_size=0.002`, `num_local_epochs=4`, `alpha=0.01` | 50.48% | 48.86% | 2.6348 |
| `step_size=0.016013056680630116`, `num_local_epochs=3`, `alpha=0.33075447277711245` | 42.71% | 35.04% | 3.5584 |
| green 1000-iteration diagnostic with the final selected parameters | 48.23% | 42.35% | 2.2643 |

Interpretation: FedDyn remains under-tuned. The green candidate was kept as the most plausible selected option because
its curve improved earlier and its loss decreased more consistently, but its final behavior is still not competitive or
fully satisfactory. FedDyn should be described as an under-tuned or sensitive baseline in any thesis discussion.

### Results Not Yet Documented

Full Experiment 5 condition results.

## Inspection Figures

The inspection command writes the following paths:

- `experiments/fedisic2019/figures/example_grid.png`
- `experiments/fedisic2019/figures/samples_per_class.png`
- `experiments/fedisic2019/figures/samples_per_client.png`
- `experiments/fedisic2019/figures/client_class_distribution.png`

It also writes:

- `metadata.csv`
- `class_counts.csv`
- `client_counts.csv`
- `client_class_counts.csv`
- `image_size_counts.csv`
- `inspection_summary.json`


Inspect and generate figures:

```powershell
.\.venv\Scripts\python.exe experiments\fedisic2019\inspect_fedisic2019.py
```

Offline inspection using an existing Hugging Face cache:

```powershell
.\.venv\Scripts\python.exe experiments\fedisic2019\inspect_fedisic2019.py --local-files-only
```

Run Experiment 5 one condition at a time:

```powershell
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment5\experiment5_communication_impairments.py --condition clean_baseline
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment5\experiment5_communication_impairments.py --condition availability
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment5\experiment5_communication_impairments.py --condition compression
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment5\experiment5_communication_impairments.py --condition drops
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment5\experiment5_communication_impairments.py --condition noise
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment5\experiment5_communication_impairments.py --condition combination
```

Run Experiment 2 one algorithm at a time:

```powershell
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment2\experiment2_aggregation_weighting.py --algorithm fedavg
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment2\experiment2_aggregation_weighting.py --algorithm fedprox
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment2\experiment2_aggregation_weighting.py --algorithm scaffold
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment2\experiment2_aggregation_weighting.py --algorithm fednova
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment2\experiment2_aggregation_weighting.py --algorithm fedopt
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment2\experiment2_aggregation_weighting.py --algorithm fedlt
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment2\experiment2_aggregation_weighting.py --algorithm feddyn
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment2\experiment2_aggregation_weighting.py --algorithm fedpd
```

Run the reduced-budget pilot experiments:

```powershell
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment5\experiment5_reduced_pilot.py --condition clean_baseline
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment5\experiment5_reduced_pilot.py --condition combination
.\.venv\Scripts\python.exe experiments\fedisic2019\experiment2\experiment2_reduced_pilot.py
```

Use offline pretrained behavior only if torchvision weights are already cached; otherwise add `--no-pretrained`.

## Known Limitations

- Full EfficientNet-B0 Fed-ISIC2019 runs are expensive on CPU.
- The Hugging Face dataset must be available locally or downloadable from the current environment.
- The initial `selected_hyperparameters.json` is a reference starting point; empirical per-algorithm selections require running `experiment0`.
- The benchmark uses full client participation because Fed-ISIC2019 has only six centers and the first baseline should isolate algorithm behavior from client availability and communication impairments.
- Reduced-budget pilot results use capped stratified train/test subsets and should not be presented as the final
  full-data benchmark.
