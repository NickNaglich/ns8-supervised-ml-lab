# AGENT-RULES.md
**Project:** NS8 Supervised Learning Showcase (scikit-learn)  
**Audience:** VSCode Codex / AI coding agents  
**Goal:** Build a recruiter-facing ML engineering project demonstrating supervised learning fundamentals using Naglich Squares (NS8) datasets.

---

## 0) Non-Negotiables (Read First)

### Primary objectives
1. Demonstrate **scikit-learn mastery**:
   - train/test splits
   - cross-validation
   - preprocessing
   - pipelines
   - hyperparameter tuning
   - evaluation and reporting
2. Make **NS8 central**:
   - datasets derived from NS8 grids parameterized by N, k, family, and orientation
3. Produce a **recruiter-ready repository**:
   - one-command reproducibility
   - clean structure
   - documented results
   - saved artifacts
   - tests

### Rules of engagement
- Prefer **simple + correct** over clever
- All experiments must be **reproducible**
- **Pipelines are mandatory** wherever preprocessing exists
- **Cross-validation** is required for model selection
- **Test sets are sacred** — never used during tuning
- All meaningful changes must be logged (see Section 12)

### Allowed stack
- Python 3.11+
- numpy, pandas
- scikit-learn
- matplotlib
- joblib

Optional (allowed but not required):
- typer or argparse (CLI)
- pydantic (configs)

Do **NOT** introduce deep learning frameworks.

---

## 1) Project Narrative (What We Are Building)

This project is a supervised-learning benchmark built on **synthetic but structured NS8 data**.

The agent will:
- implement NS8 grid generation from canonical formulas
- derive tabular features from grids
- construct multiple supervised tasks:
  - **Classification:** predict NS8 family/orientation
  - **Regression:** predict N (or another continuous structural proxy)
- train, tune, and evaluate classical ML models
- save reproducible artifacts and reports

This repo must read like **real ML engineering work**, not a tutorial.

---

## 2) Canonical NS8 Definition (Source of Truth)

All NS8 grids are 1-indexed and must follow these formulas exactly.

### Helper functions
