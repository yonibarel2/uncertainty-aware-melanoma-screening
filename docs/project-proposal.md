# Uncertainty-Aware Skin Lesion Classification: Learning When to Defer to a Physician

> Verbatim transcription of the submitted project proposal.

**Deep Learning Course – Healthcare (Applied & Algorithmic)**
**By Elhar Brand and Yehonatan Barel**

## Problem Definition

Melanoma is the deadliest form of skin cancer, and early detection dramatically improves survival rates. Convolutional networks have achieved strong performance on dermoscopic image classification, but in a clinical setting an overconfident wrong prediction can be catastrophic. Standard classifiers often produce high softmax scores even for cases the model cannot reliably classify. This project asks: **Can uncertainty estimation reliably identify melanoma-screening cases that should be deferred to a physician?** We aim to build a melanoma-specific selective classifier that makes a screening prediction when confident and defers uncertain cases to a dermatologist.

## Main Challenges

1. **Class imbalance** – melanoma comprises roughly 11% of the dataset.
2. **Data leakage risk** – multiple images can originate from the same lesion, so image-level splits produce inflated results and lesion-level splitting is required.
3. **Estimating meaningful predictive uncertainty** that can distinguish reliable predictions from difficult cases.
4. **Evaluation must go beyond accuracy** and jointly assess calibration (ECE, Brier score) and selective-prediction behavior via risk-coverage and sensitivity-coverage curves.

## Data

HAM10000 contains 10,015 dermoscopic images derived from 7,470 unique lesions, labeled across 7 categories. Our primary task is binary melanoma screening (1,113 melanoma vs. 8,902 non-melanoma). All splits will be performed at the `lesion_id` level to prevent train–test leakage. As an extension, we will investigate the full 7-class problem, where the uncertainty structure becomes richer due to ambiguity between multiple related lesion types.

## Proposed Method

Our baseline is a fine-tuned ResNet-50 / EfficientNet-B0 (ImageNet-pretrained) trained with focal loss to address class imbalance. On top of this baseline we compare two uncertainty-estimation methods:

1. **Monte Carlo Dropout** (Gal & Ghahramani, 2016) – approximate Bayesian inference via T = 30 stochastic forward passes with dropout active at inference.
2. **Deep Ensembles** (Lakshminarayanan et al., 2017) – M = 5 independently trained models with different random seeds.

From each method we extract **predictive entropy** and **mutual information** as uncertainty measures.

## Related Work

Prior work has explored uncertainty-aware skin lesion classification, physician referral, and comparisons between uncertainty-estimation methods. Our project focuses on a controlled comparison of MC Dropout and Deep Ensembles on HAM10000 using lesion-level splitting and a clinically motivated, cost-sensitive selective-referral framework.

## Planned Contribution

- **(a)** A systematic comparison of two principal uncertainty families on HAM10000 with lesion-level splits, analyzing the accuracy–compute trade-off.
- **(b)** A cost-sensitive defer policy that incorporates the asymmetric costs of missed melanoma (a high-severity clinical error) versus referring a case to a dermatologist (an operational cost), and selects the referral threshold accordingly.
- **(c)** A failure analysis characterizing which lesion types drive high uncertainty.
- **(d)** Risk-coverage and sensitivity-coverage curves as the central evaluation, quantifying the melanoma-sensitivity gain achieved when the least-confident X% of cases are deferred to a physician.
