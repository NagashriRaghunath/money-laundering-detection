# Anti-Money Laundering Detection Using Machine Learning

A machine learning project that detects money-laundering transactions in the IBM Transactions for Anti-Money Laundering (AML) HI-Small dataset. The work follows the CRISP-DM lifecycle from data understanding through to a consolidated evaluation dashboard.

**Dataset:** [IBM Transactions for AML (HI-Small)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

## What's in here

| Notebook | Section | Description |
|----------|---------|-------------|
| `Group_Notebook_1_Data_Prep.ipynb` | 1 & 2 | Data loading, EDA, scaling, one-hot encoding, and SMOTE balancing |
| `Nagashri_Modeling.ipynb` | 3 & 4 | Individual modelling - Logistic Regression baseline + MLP |
| `Akash_Modelling_CW_F.ipynb` | 3 & 4 | Individual modelling |
| `Parth_Modelling_Notebook_final.ipynb` | 3 & 4 | Individual modelling |
| `Piyush_Modelling_Notebook_final.ipynb` | 3 & 4 | Individual modelling |
| `Group_Notebook_2_Visualisation.ipynb` | 5 | Aggregates all models into an evaluation dashboard |
| `cross_group_comparison_final.csv` | - | Small results file powering the dashboard |

## The approach

- **Sections 1 & 2 - Data prep:** loading the dataset, exploratory analysis (distributions, correlations), and pre-processing. Class imbalance is handled with SMOTE balancing at a 10% ratio.
- **Sections 3 & 4 - Modelling:** each group member trained a shared Logistic Regression (L2) baseline plus a unique model of their own, then tuned and evaluated it.
- **Section 5 - Evaluation:** all the individual models are pulled together into a single dashboard for a like-for-like comparison.

## A note on the data

The fully processed datasets (the balanced training files) come to over 1.3 GB, so they aren't included here. To regenerate them, run `Group_Notebook_1_Data_Prep.ipynb` top to bottom - it downloads the original Kaggle dataset and runs the complete preprocessing pipeline that the modelling notebooks depend on.

## Running it

\`\`\`bash
pip install
