# Interpretable Physics-Guided Machine Learning Reveals Electronic Descriptor Influence on the HOMO–LUMO Gap of Functionalized Cellulose
Made by Jherby Kyle C. Teodoro

## Overview

This repository contains the complete implementation of a statistically validated and interpretable physics-guided machine learning framework for predicting the HOMO–LUMO energy gap (ΔE) of functionalized cellulose systems.

The study investigates how quantum chemically derived global reactivity descriptors influence electronic stability using interpretable ensemble learning.

The framework integrates:

- Physics-guided descriptor selection (conceptual DFT)
- Data augmentation (controlled ±2% noise)
- Gradient Boosting regression
- Leave-One-Out Cross-Validation (LOOCV)
- External validation
- Permutation feature importance
- SHAP interpretability
- Prediction interval estimation
- Residual diagnostics

---

## Scientific Objective

To determine how electronic structure descriptors influence the HOMO–LUMO gap (ΔE) in functionalized cellulose systems using interpretable machine learning.

Rather than redefining ΔE, the model quantifies descriptor contributions and nonlinear interactions governing electronic stability.

---

## Dataset

The dataset consists of eight functionalized cellulose structures with the following descriptors:

- LUMO energy
- HOMO energy
- Chemical potential (μ)
- Hardness (η)
- Softness (σ)
- Electrophilicity index (ω)
- Total dipole moment (TDM)
- HOMO–LUMO gap (ΔE)

Controlled augmentation (25× per structure) is applied using ±2% Gaussian perturbation to simulate realistic DFT uncertainty.

---

## Model Architecture

- Regressor: GradientBoostingRegressor
- n_estimators = 800
- learning_rate = 0.02
- max_depth = 2
- subsample = 0.9
- min_samples_leaf = 1
- random_state = 42

Validation strategies:
- Strict Leave-One-Out Cross-Validation (LOOCV)
- 20% independent external validation split

---

## Key Results

- LOOCV R² ≈ 0.948
- External R² ≈ 0.934
- Stable residual distribution
- Reliable prediction intervals
- Electrophilicity (ω) identified as dominant descriptor via SHAP and permutation importance

---


---

## Installation

### Requirements

- Python 3.9+
- numpy
- pandas
- matplotlib
- scikit-learn
- shap


---

## Interpretability

Global interpretability:
- Permutation importance
- SHAP summary plot

Local interpretability:
- SHAP dependence plots

Uncertainty quantification:
- 95% prediction intervals derived from ensemble variability

---

## Reproducibility

- Random seed fixed
- Deterministic augmentation
- Frozen model hyperparameters
- Explicit external validation split

---

## Citation

If this work is used in academic research, please cite:

*Interpretable Physics-Guided Machine Learning Reveals Electronic Descriptor Influence on the HOMO–LUMO Gap of Functionalized Cellulose.*

(Manuscript in preparation)

---

## License

MIT License
---

## Notes

This framework demonstrates descriptor–property relationships rather than redefining electronic structure equations. It is intended for interpretable electronic descriptor analysis and computational screening of functionalized cellulose systems.

