# Solution Report: Zero-Order Fine-Tuning for CIFAR100

## 1. Reproducibility instructions
```bash
git clone https://github.com/PavelPodlesny/SMILES-2026-ZO-Limited-Resnet.git
cd ./SMILES-2026-ZO-Limited-Resnet
pip install -r requirements.txt
python validate.py \
	--data_dir ./data \
	--batch_size 64 \
	--n_batches 128 \
	--output results.json
```

## 2. Final solution description
### 2.1 Modified components
1. zo-optimizer: implemented Simultaneous Perturbation Stochastic Approximation (SPSA); the optimizer reads hyperparameters from optimizer_config.json
2. augmentation.py: tested various augmentation combinations; the most successful setup was:

   **Resize (224):** Standardize input dimensions.
   **Random Crop (224, padding=28):** Provided translation invariance.
   **Random Rotation (15°):** Introduced rotational robustness.
   **Random Horizontal Flip:** Effectively doubled the dataset variety.
   **Normalization:** Standard CIFAR100 statistics.

4. head_init.py: **Kaiming Uniform** initialization with zero bias, matching the ReLU activation characteristics of the ResNet backbone.

### 2.2 SPSA Hyperparameters
* **Perturbation Magnitude (c): 0.01.** Reducing c from the theoretical default (0.2) to 0.01 produced a substantial improvement. Smaller perturbations prevented the destruction of pretrained features and provided more stable gradient estimates.
* **Stability Constant (A): 12.8. (n_batches/10)** A non-zero A helped stabilize the initial learning rate, preventing volatile updates in the first few batches.
* **Distribution: Rademacher.** Using ±1 perturbations proved more efficient than Gaussian or Uniform distributions for the limited budget.
* **Steps:** 128 steps (Batch Size 64) utilized the full sample budget (8,192).

### 2.3 What contributed most:
Here and below, +X% accuracy means improvement over the initialized head.
1. SPSA hyperparameters selection (reducing c from 0.2 to 0.01) (+0.23% accuracy)
2. Augmentation (+0.12% accuracy)
3. default SPSA implementation (+0.09% accuracy)


Here is How the contributions were estimated:

default SPSA yields +0.09% accuracy

SPSA + hyperparameters tuning yields +0.32%

SPSA + hyperparameters tuning + augmentations yields +0.44%

## 3. Experiments and failed attempts
1. Xavier and orthogonal weight initialization: no improvement over the baseline head (1.21%)
2. WeightedRandomSampler: accuracy decreased (-0.12%)
3. Different combinations of augmentations: all spatial augmentations (crop, flip, rotation) were beneficial; ColorJitter wasn't beneficial (-0.13%)
4. (n_batches, batch_size) = (256,32), e.g. More steps but noisier updates; discarded due to a -0.31% accuracy drop.

## 5. Experimental Results
| Stage | Config Change | Validation Acc (Top-1) | Gain ($\Delta$) |
| :--- | :--- | :--- | :--- |
| **Baseline** | Initial Head | 1.21% | - |
| **Optimization** | $c=0.01$, $A=12.8$ | 1.53% | +0.32% |
| **Augmentation** | Spatial (Crop+Rot+Flip) | **1.65%** | **+0.44%** |

## 6. Key Takeaways
1. In fine-tuning, the weights are already "near" a good solution. Large perturbations (c=0.2) act as destructive noise rather than gradient probes.
2. For CIFAR100, translation and rotation augmentations provided a more significant boost than complex policies like AutoAugment in the low-data regime.
3. Balancing batch size (64) with the number of steps (128) ensured the SPSA estimator had enough samples to be accurate while still allowing for a sufficient number of parameter updates.
