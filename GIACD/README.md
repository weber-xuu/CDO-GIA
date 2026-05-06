# CDO-GIA: Gradient Inversion Attack with Candidate Descent

 **This repository contains the official PyTorch implementation and experimental data for the paper** **"[CDO-GIA: A Robust Textual Gradient Inversion Attack against Federated
Language Models via Continuous-Discrete Optimization]"**.

  **We propose** **CDO-GIA**, a method to recover private text data from gradients by combining continuous optimization with a discrete candidate search strategy. This repository also includes implementations of baseline methods (LAMP, TAG, DLG) for comparison.

## 🛠️ Setup & Dependencies

**This project uses** **conda** **for environment management. You can install all necessary dependencies using the provided** **environment.yml** **file.**

**code**Bash

```
# Create the environment
conda env create -f environment.yml

# Activate the environment
# (Replace 'env_name' with the name specified inside environment.yml, usually at the top)
conda activate env_name
```

## 🚀 Running Experiments

**Below are the exact commands used to reproduce the experimental results on the** **CoLA** **dataset (Test split).**

### 1. Proposed Method: CDO-GIA

**To run our proposed Gradient Inversion Attack with Candidate Descent:**

**code**Bash

```
python attack_CDO-GIA.py \
  --dataset cola \
  --split test \
  --queue_size 30 \
  --iteration_size 50 \
  --rng_seed 301 \
  --loss cos \
  --n_inputs 100 \
  -b 1 \
  --coeff_perplexity 0 \
  --coeff_reg 1 \
  --lr 0.01 \
  --lr_decay 0.89 \
  --n_steps 2000
```

### 2. Baseline Methods

**We compare our method against three state-of-the-art baselines. Note that baselines are executed using** **attack_LAMP.py** **with specific configuration flags.**

#### LAMP (Language Model Priors)

**Comparison with the standard LAMP attack:**

**code**Bash

```
python attack_LAMP.py \
  --dataset cola \
  --rng_seed 301 \
  --split test \
  --swap_every 250 \
  --loss cos \
  --n_inputs 100 \
  -b 1 \
  --coeff_perplexity 0.2 \
  --coeff_reg 1 \
  --lr 0.01 \
  --lr_decay 0.89 \
  --n_steps 2000
```

#### TAG (Text Attack via Gradients)

**To run TAG, we use the** **--baseline** **flag and the** **tag** **loss function:**

**code**Bash

```
python attack_LAMP.py \
  --baseline \
  --dataset cola \
  --split test \
  --loss tag \
  --n_inputs 100 \
  --swap_every 0 \
  --rng_seed 301 \
  -b 1 \
  --lr 0.1 \
  --lr_decay 1 \
  --tag_factor 0.01 \
  --n_steps 2500
```

#### DLG (Deep Leakage from Gradients)

**To run the DLG baseline:**

**code**Bash

```
python attack_LAMP.py \
  --baseline \
  --dataset cola \
  --split test \
  --loss dlg \
  --n_inputs 100 \
  --swap_every 0 \
  --rng_seed 301 \
  -b 1 \
  --lr 0.1 \
  --lr_decay 1 \
  --n_steps 2500
```

## 📂 Output

**The reconstruction results and metrics will be saved in the** **/directory. The filenames include the timestamp and hyperparameters to ensure reproducibility.**
