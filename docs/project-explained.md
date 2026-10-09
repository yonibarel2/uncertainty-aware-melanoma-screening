# Project Explained: What Are We Actually Doing

> Mirrored from the Claude Doc of the same name (2026-10-09). The Claude Doc is where we edit; this copy is for reference in the repo.

## Open Tasks

- [ ] Understand the project
- [ ] Build it
- [ ] Write the report (as written in the instructions)
- [ ] Prepare the presentation (as written in the instructions)

## CNN Basics (Only What We Need)

**What a CNN does:** a neural network for images. It slides small filters over the photo to find patterns: first edges and colors, then textures, then shapes like an irregular border. At the end it outputs a % (e.g. "92% melanoma").

**Pretrained + fine-tuning:** training a CNN from zero needs millions of photos, and we have 10k. So we take a CNN already trained on ImageNet (a famous public collection of 1.2M everyday photos), which already knows edges and textures, and continue training it on our skin photos. That retraining is our "training" part.

**Our two CNNs:**

- **ResNet-50:** 50 layers, ~25M weights. Well-known, strong baseline.
- **EfficientNet-B0:** ~5M weights, similar quality, much cheaper. Matters because Deep Ensembles train it 5 times.

**Dropout:** during training, some neurons are randomly switched off at each step, so the network doesn't rely on any single one. Normally it's off when predicting. MC Dropout keeps it on (more later).

## The Problem

### 1. CNNs are good on average

Melanoma is the deadliest form of skin cancer. **Early detection dramatically improves survival rates.**

CNNs (neural networks made for images) are already good at deciding if a skin photo shows melanoma or not:

The CNN outputs a % (e.g. "92% melanoma"). We turn it into a decision with a cutoff: above 50% means "melanoma" - so "good" means:

- Show it a **melanoma** photo and it usually gives a high melanoma %, so the decision is "melanoma" (e.g. 90 of 100).
- Show it a **healthy** photo and it usually gives a high healthy %, so the decision is "healthy" (e.g. 855 of 900).

On the group level it is good: it catches 90% of cancers and clears 95% of healthy people.

### 2. The dangerous mistake and a simple fix

**The bad case (false negative):** a real melanoma called "healthy" (e.g. 55% healthy), so the patient goes home. This is the most dangerous mistake.

**Suggested solution:** send home only if 95%+ healthy, else send to a doctor, so unsure cases get checked.

### 3. Why the simple fix fails

CNNs tend to be **overconfident**: they often give a high % (like 99%) even on hard photos they may get wrong.

**Example:** look at the 10 missed cancers and let's assume:

- **4 said "60% healthy":** the model was unsure, so we could flag them.
- **6 said "99% healthy":** many of the 855 correct answers also say ~99% healthy, so the % alone can't tell these 6 apart.

It is like a great doctor: right 95% of the time, but sometimes confidently wrong.

**Bottom line:** CNNs are good on the group level, but can still make a confident mistake on a single image, in both directions: missing a cancer, or a false alarm on a healthy person. Missing a cancer (the patient goes home sick) is far worse than a false alarm (an extra doctor visit).

### 4. Our research question

**The question:** can uncertainty estimation (methods that measure how unsure the model is, better than the plain %) catch the risky cases, including the hidden "99%" mistakes, and send them to a doctor (instead of trusting the model)?

**Note:** we don't prevent the mistake, because the model still makes it. We catch it and pass the case to a doctor.

**The goal:** we aim to build a **selective classifier**, a model that is allowed to not answer. Under the hood:

1. The model gives its % (e.g. "97% healthy").
2. We estimate how much to trust that %, in 2 checks:
   - Is the % far enough from 50%? If not, the model is openly unsure.
   - If yes, is that high % really trustworthy, or a hidden mistake?
3. We decide: trusted means "melanoma" or "healthy"; not trusted means send to a dermatologist (skin doctor).

So the final model has 3 outputs: melanoma, healthy, or "ask a doctor".

## Main Challenges

1. **Class imbalance:** only ~11% of photos are melanoma, so a lazy "always healthy" model looks good, and training may ignore melanomas. **Fix:** focal loss (later).
2. **Data leakage:** some spots have several photos. One in training and another in test means the test is cheating. **Fix:** split by spot (`lesion_id`), keeping all 10k photos.

***Note:*** *the proposal lists 2 more "challenges" that are not really challenges. #3 (trusting the high %) is our research question, see section 4 above. #4 (evaluation beyond accuracy) is how we measure success, its own section later.*

## Data

- **HAM10000:** a public dataset of 10,015 skin-spot photos of 7,470 spots, labeled into 7 lesion types.
- **Our task is binary:** melanoma (1,113) vs. everything else (8,902).
- **Split:** by spot (`lesion_id`), keeping all photos.
- **The 7-type problem:** future work, we won't do it.

## Proposed Method

### 1. The baseline

**The baseline:** the plain model we compare everything against.

- **Pretrained (by others):** a ResNet-50 / EfficientNet-B0 already trained on ImageNet.
- **Fine-tuned (by us):** we keep training it on our skin photos, using **focal loss**.

**Focal loss:** gives almost no penalty when the model is already right (98% on the correct answer) and a big penalty when it's wrong (30% on the correct answer), so hard photos (like missed melanomas) change the model's weights a lot, and easy ones barely change them.

**Why it fixes the imbalance:** most healthy photos are easy, so their penalties shrink to almost zero and the rare melanomas no longer get drowned out by them.

### 2. Uncertainty methods

**What we do:** keep the same trained CNN and check whether a high % is really trustworthy. The idea: don't ask the model once, ask it many times and see if it agrees with itself.

- **Agrees every time** (99%, 98%, 99%): trust it.
- **Answers jump around** (99%, 60%, 85%): hidden doubt, so doctor.

**Two ways to get many answers:**

1. **MC Dropout:** one model, asked 30 times with dropout left on, so each ask switches off a different random set of neurons. Cheap: 1 training run.
2. **Deep Ensembles:** the same CNN trained 5 times from different random starts, giving 5 different models, each asked once. More reliable, but 5 training runs.

We do both because comparing them is one of our contributions.

### 3. Two trust numbers

From each method we calculate 2 numbers that say how unsure the model is. They match our 2 checks:

| Number | What it measures | Our check |
| --- | --- | --- |
| **Predictive entropy** | How unsure the **average** answer is (average 50% = high, average 99% = low) | Is the % far from 50%? |
| **Mutual information** | How much the answers **disagree** with each other (99%, 60%, 85% = high) | Is the high % really trustworthy? |

We decide with **predictive entropy only**, our one trust score. It already includes mutual information (predictive entropy = mutual information + average single-answer doubt). MI is shown separately only for analysis.

## Related Work

Other researchers already tried our idea: a skin model that sends unsure cases to a doctor. We do it more carefully:

- We compare both methods **under the exact same conditions**.
- We avoid the photo-leak mistake (split by spot).
- We pick the threshold by **real costs** (later).

## Planned Contribution

The 4 contributions (a)–(d) are not 4 tasks. They are the **4 results we report**. Below is the whole project in order, and where each result comes from.

### Step 1: Split the photos

Split by spot into 3 groups, with ~11% melanoma in each:

| Group | Share | Used for |
| --- | --- | --- |
| Train | 70% | The model learns from these |
| Validation | 15% | Tuning choices: when to stop training, where to put the cutoff |
| Test | 15% | The final numbers, used once at the end |

**Rule:** never tune on the test set.

### Step 2: Train the models (train set)

**Baseline:** the CNN fine-tuned with focal loss.

- **MC Dropout:** uses this same model, with dropout left on when predicting.
- **Deep Ensemble:** the same CNN trained 5 times from different random starts.

### Step 3: Validation set → (b)

Every choice is made here, and frozen before we touch the test set:

1. **Stop training at the right time:** after each epoch (one full pass over the training photos), check validation AUC and keep the best. Stop if it doesn't improve for ~5 epochs.
2. **Tune settings, only if needed:** γ, α, learning rate (step size). Start with defaults; if results are weak, try 2–3 values and keep the best.
3. **Pick the defer cutoff at 1:100** (details below).
4. **Repeat the cutoff at 1:10 and 1:50:** "100" is a guess, so we show how the cutoff moves. The more you fear a miss, the more you refer.

**The cutoff in detail:**

No more training. Every photo already has a trust score, so we only choose where to draw the line: below it, send to a doctor.

- **Too strict:** doctors get flooded.
- **Too loose:** cancers get sent home.

**Our answer, with prices:** missed melanoma = 100, doctor visit = 1. Try many cutoffs and keep the cheapest (numbers made up):

| Send to doctor | Missed melanomas | Doctor visits | Total cost |
| --- | --- | --- | --- |
| 5% least trusted | 8 | 50 | 850 |
| 15% least trusted | 3 | 150 | **450** (best) |
| 40% least trusted | 1 | 400 | 500 |

This cost-based cutoff is our main original idea.

### Step 4: Final exam (test set) → (a) and (d)

Run the baseline, MC Dropout and the Ensemble on the same test photos and compare:

| Question | Measured by |
| --- | --- |
| Does it classify well? | sensitivity, AUC |
| Is its % honest? | ECE, Brier (calibration) |
| When we send the least-trusted cases to a doctor, how much better is the rest? | the 2 curves (d) |
| How expensive is it? | training + prediction time |

- **(a):** is the Ensemble's better quality worth 5× the training cost?
- **(d):** the curves. How much does catching melanoma improve as we send more of the least-trusted cases to a doctor? (details later)

### Step 5: Look at the mistakes (test set) → (c)

Which spot types does the model find hard or get wrong? For example: "most uncertain cases are moles that look like melanoma." This is the deep-insight part the lecturer wants.
