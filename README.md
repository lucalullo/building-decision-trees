# Building Decision Trees

A progressive educational project that shows how to build **decision-tree learning from scratch**, starting from a one-split categorical stump and evolving step by step toward numerical thresholds, recursive tree growth, native missing-value routing and simple gradient boosting.

The project uses the **Kaggle Titanic dataset** as a constant practical laboratory. Keeping the same dataset across all versions makes each algorithmic change easy to inspect and compare.

The implementation intentionally avoids a ready-made decision-tree estimator. Split generation, error measurement, gain, leaf predictions, recursive growth, missing-value routing and boosting updates are implemented directly so that the mechanics remain visible.

Version 5 completes the original five-stage roadmap with a small gradient boosting model built on top of the missing-aware tree developed in the previous versions.

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white) ![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Kaggle Notebook](https://img.shields.io/badge/Kaggle-Notebook-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/lucalullo/building-decision-trees)

[Kaggle notebook](https://www.kaggle.com/code/lucalullo/building-decision-trees) · [GitHub repository](https://github.com/lucalullo/building-decision-trees) · [Technical report](project-report-en.pdf)

![Project roadmap and architecture](infographic.png)

## Overview

The project follows one main principle:

> **One version, one major concept.**

Version 1 introduces a **Categorical Decision Stump**:

1. learn a constant prediction from the target;
2. measure prediction error with mean squared error;
3. evaluate categorical equality rules;
4. compute weighted split error and gain;
5. select the best one-split rule;
6. learn one prediction for each branch;
7. convert the final values into binary Titanic predictions.

Version 2 adds **Numerical Split Search** while preserving the one-split architecture:

## Current capabilities

- load the Titanic training and test datasets;
- extract passenger titles from names as a simple engineered feature;
- distinguish numerical and categorical features explicitly;
- learn constant leaf predictions directly from target values;
- compute mean squared error from scratch;
- evaluate weighted split error;
- compute split gain as error reduction;
- evaluate categorical equality splits such as `Title == "mr"`;
- generate numerical threshold candidates from consecutive-value midpoints;
- compare categorical and numerical candidates with the same gain criterion;
- grow a recursive binary decision tree;
- reuse the same feature more than once in different parts of the tree;

## Architecture

```text
Titanic Data
    ↓
Feature Engineering
Title extracted from Name
    ↓
Feature Types
Age / Fare → numerical
Sex / Title / Pclass → categorical
    ↓
Candidate Split Generation
categorical equality rules
numerical midpoint thresholds
    ↓
Split Evaluation
weighted MSE + gain
    ↓
Missing-Aware Numerical Routing
missing → left or right
choose best gain
    ↓
Recursive Decision Tree
internal nodes + terminal leaves
    ↓
Shallow Tree Learner
(max depth = 2 in Version 5)
    ↓
Initial Constant Prediction
mean(y)
    ↓
Residuals
y - current prediction
    ↓
Train Tree on Residuals
    ↓
Scale Correction
learning_rate × tree prediction
    ↓
Update Current Prediction
    ↓
Repeat for 5 Trees
    ↓
Final Continuous Score
    ↓
Threshold 0.5
    ↓
Binary Titanic Prediction
    ↓
Kaggle Submission
```

The key transition in Version 5 is that a leaf no longer represents only a direct class estimate. Inside boosting, each shallow tree learns a **correction** to the current prediction by fitting the residuals left by the previous trees.

## Version guide

| Version | Focus | Notebook and resources |
|---:|---|---|
| 1 | Categorical Decision Stump | [`v1-categorical-decision-stump/`](v1-categorical-decision-stump/) |
| 2 | Mixed Decision Stump | [`v2-mixed-decision-stump/`](v2-mixed-decision-stump/) |
| 3 | Recursive Decision Tree | [`v3-recursive-decision-tree/`](v3-recursive-decision-tree/) |
| 4 | Missing-Aware Decision Tree | [`v4-missing-aware-decision-tree/`](v4-missing-aware-decision-tree/) |
| 5 | Simple Gradient Boosting | [`v5-simple-gradient-boosting/`](v5-simple-gradient-boosting/) |

### What each version introduces

<details>
<summary>Version 1 - Categorical Decision Stump</summary>

Version 1 establishes the complete learning loop with one split.

The model learns a constant prediction, evaluates categorical equality rules with mean squared error, computes the gain produced by each candidate and stores one prediction for the left branch and one for the right branch.

</details>

<details>
<summary>Version 2 - Mixed Decision Stump</summary>

Version 2 keeps the one-split architecture and adds correct handling of numerical features.

Numerical candidates use threshold rules such as:

</details>

<details>
<summary>Version 3 - Recursive Decision Tree</summary>

Version 3 introduces recursive tree growth.

Instead of stopping after the best root split, each branch becomes a new training problem. The same best-split logic is reused on the subset of rows that reaches each node.

- the node is pure;
- the maximum depth has been reached;
- no candidate produces positive gain.

</details>

<details>
<summary>Version 4 - Missing-Aware Decision Tree</summary>

Version 4 removes median imputation as a requirement for numerical missing values.

For each numerical threshold, the model evaluates two alternatives:

</details>

<details>
<summary>Version 5 - Simple Gradient Boosting</summary>

Version 5 builds a small gradient boosting model on top of the missing-aware recursive tree.

The model starts from the average Titanic target:

</details>

## Getting started

### Requirements

- Python 3
- Jupyter Notebook or JupyterLab
- pandas
- Titanic `train.csv` and `test.csv` data

The notebooks were developed for the Kaggle Titanic environment and currently read the competition files from:

```python
PATH = "/kaggle/input/competitions/titanic/"
```

For the simplest reproducible experience, use the Kaggle notebook linked above.

To run locally, clone the repository:

```bash
git clone https://github.com/lucalullo/building-decision-trees.git
cd building-decision-trees
```

Start Jupyter Notebook:

```bash
jupyter notebook
```

Download the Titanic `train.csv` and `test.csv` files, then update the `PATH` variable in the notebook to point to their local directory.

Open the version you want to study and run the cells in order.

For example, to study the final boosting implementation:

```text
v5-simple-gradient-boosting/building-decision-trees.ipynb
```

## Repository structure

Each version is kept in its own folder with the corresponding notebook and, where available, version-specific diagrams and reports.

## Documentation

- [Project report (English)](project-report-en.pdf)
- [Relazione del progetto (Italiano)](project-report-it.pdf)
- [Project infographic](infographic.png)

## Limitations

- the implementation is written for clarity rather than computational efficiency;
- numerical threshold search is exhaustive over midpoint candidates;
- categorical splits use one-category-versus-rest equality rules rather than arbitrary category subsets;
- tree growth is controlled only by simple stopping conditions and maximum depth;
- there is no pruning or cost-complexity regularization;
- native learned missing-value routing is implemented for numerical splits;
- the boosting objective uses squared-error residuals for a binary target rather than logistic loss;
- final continuous boosting scores are thresholded at `0.5` and should not be interpreted as calibrated probabilities;
- the final boosting configuration uses only five shallow trees with fixed hyperparameters;

## Project status

**Building Decision Trees is complete according to the V1-V5 structure originally planned for the project.**

The repository now contains the full educational path from one categorical rule to a recursive missing-aware decision tree and a small gradient boosting model built from the same tree implementation.

Version 1 established the first working Titanic baseline with a Kaggle public score of `0.75119`.

## License

Distributed under the [MIT License](LICENSE).

## Author

Created by [Luca Lullo](https://github.com/lucalullo).
