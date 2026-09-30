# Interior Interpretability with Attention Rollout

This repository accompanies the paper 
[*Interior interpretability with attention rollout: contraction and propagation profiles in Transformers*](https://arxiv.org/abs/2607.22367).
It contains the code used to train tabular Transformer regressors for age prediction and to analyse how their self-attention patterns 
accumulate across layers. The project studies a metabolomic age-prediction cohort (Dataset 1, available subject to authorization) and a fully 
included synthetic tabular dataset (Dataset 2).

We apply the **attention rollout** construction as a diagnostic of attention-mediated propagation between feature tokens. 
It also includes PCA and SHAP GradientExplainer analyses, which provide complementary variance-based and 
prediction-attribution viewpoints.

## Motivation

Feature-attribution methods ask which input variables receive credit for a prediction. They do not directly describe 
how explicitly defined interactions inside a deep network compose from layer to layer. 
This work introduces *interior interpretability*: the analysis of operators that summarize selected internal interactions 
of a learned model.

For a tabular Transformer, each input variable is represented by its own token. The self-attention matrices describe 
token-to-token interactions at each layer, and attention rollout composes these matrices—while accounting for residual 
connections—into one feature-to-feature propagation operator. This provides a compact view of how attention-mediated 
propagation is organized across the network depth.

Attention rollout does **not** represent the complete Transformer computation: it omits value and output 
projections, normalization, MLP sublayers, and other nonlinear operations. Consequently, the resulting propagation 
scores must not be read as causal effects or as faithful attributions of the final prediction.

## Approach implemented

The attention rollout construction averages the attention heads in every Transformer layer, incorporates the residual 
connection, and composes the resulting matrices into an attention-rollout matrix. The column sums of this matrix give 
a propagation-based feature ranking: a larger score means that a feature accounts for more of the 
total attention-mediated mass in the rollout operator.

This ranking has a structural interpretation using classical Doeblin-Dobrushin contraction theory. 
When the rollout operator becomes strongly contractive, its rows approach a common propagation profile; the normalized
column sums identify that profile. 

![Attention rollout](rollout.png)

The empirical analysis compares three distinct quantities:

| Method | What it describes |
| --- | --- |
| PCA | Variance structure of the input data, independently of a predictor. |
| Attention rollout | Attention-mediated propagation between feature tokens across Transformer layers. |
| SHAP GradientExplainer | An expected-gradients approximation to prediction-attribution scores. |

These analyses are intentionally complementary; their feature rankings are not expected to agree in general.

## Repository contents

| File or directory | Purpose |
| --- | --- |
| `data_preprocessing_DS1.ipynb` | Preprocesses the restricted metabolomic Dataset 1. |
| `data_preprocessing_DS2.ipynb` | Preprocesses the included synthetic Dataset 2. |
| `train.ipynb` | Trains tabular Transformers with multiple depths and random seeds. |
| `attn_rollout.ipynb` | Extracts attention matrices and computes rollout operators. |
| `dobrushin.ipynb` | Evaluates rollout contraction and compares trained and untrained models. |
| `feature_importance.ipynb` | Computes rollout-based feature rankings. |
| `shap.ipynb` | Computes GradientExplainer approximations to SHAP scores. |
| `PCA_inspection.ipynb` | Computes PCA variance and feature-importance summaries. |
| `synthetic_data.csv` | Synthetic Dataset 2 used for the reproducible qualitative example. |
| `trained_models/`, `untrained_models/`, `Results/` | Saved checkpoints and experiment outputs. |

## Requirements

Use Python 3.10 or newer with Jupyter and the packages listed in `requirements.txt`:

```bash
pip install -r requirements.txt
pip install jupyter
```

The code uses PyTorch and can run on CPU; GPU acceleration is optional.

## Running the simulations

From this directory, launch Jupyter:

```bash
jupyter notebook
```

For the synthetic, fully reproducible workflow, run the notebooks in this order:

1. `data_preprocessing_DS2.ipynb`
2. `train.ipynb`
3. `attn_rollout.ipynb`
4. `dobrushin.ipynb`, `feature_importance.ipynb`, `shap.ipynb`, and `PCA_inspection.ipynb`

Dataset 1 is not included because it is governed by a data-sharing agreement. Its preprocessing notebook documents the 
expected workflow; use of that dataset requires the appropriate authorization. Saved models and results are provided 
for inspecting the reported analyses without retraining.

## Reference

U. Biccari, Q. Huang, and E. Zuazua, *Interior interpretability with attention rollout: contraction and propagation profiles in Transformers*, 2026. 
The manuscript is available in [arXiv:2607.22367](https://arxiv.org/abs/2607.22367).

## Funding

This project has received funding from the European Research Council (ERC) under the European Union's Horizon Europe research and innovation programme (grant agreement No. 101096251, CoDeFeL).
