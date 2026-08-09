# Basic Machine Learning - I: Practical Notebook

This repository contains the hands-on companion notebook for the **Basic Machine Learning - I**
session, covering data preprocessing, Linear Regression, Logistic Regression, K-Nearest Neighbors
(KNN), and Support Vector Machines (SVM).

## What's Inside

The notebook (`Basic_Machine_Learning_I_Practical.ipynb`) is built around two real datasets, both
loaded directly through `seaborn` (no manual download needed):

- **Titanic dataset** — used for the full data preprocessing walkthrough, then reused for Logistic
  Regression, KNN, and SVM
- **Auto MPG dataset** — used for Linear Regression

| Practical | Dataset | What's Covered |
|---|---|---|
| 1. Data Preprocessing | Titanic | Missing value handling, categorical encoding, EDA, standardization, train-test split |
| 2. Linear Regression | Toy example + Auto MPG | Manual scaling, **assumption checks (linearity, normality, independence, homoscedasticity, multicollinearity, autocorrelation) with full interpretation ranges**, evaluation (RMSE/R²), **before/after performance comparison when removing high-VIF and statistically insignificant features**, prediction visualization, feature weights **with p-values** (via `statsmodels`), a prediction function |
| 3. Logistic Regression | 10-record toy example + Titanic | Feature importance, accuracy/precision/recall/F1, confusion matrix |
| 4. KNN | Toy example + Titanic | Best K via cross-validation, permutation importance, confusion matrix, decision boundary |
| 5. SVM | Toy example + Titanic | Support vectors and margin, feature importance, confusion matrix, decision boundary, 3-way model comparison |

For Linear Regression, Logistic Regression, KNN, and SVM, each practical opens with a small,
fully-visible toy example (a handful of hand-built rows) before moving to the real dataset — so the
mechanics of each algorithm are clear before the data gets more complex.

### A Note on `Pipeline`

scikit-learn's `Pipeline` class bundles scaling and modeling into a single object. This notebook
deliberately avoids it in the main flow — every step (scale, fit, predict) is written out separately
so it's easy to see exactly what happens at each stage. A commented-out `Pipeline` equivalent is
included at a couple of points, so you can reveal it as a "here's the shortcut" moment once the
manual version is clear to your learners.

## Requirements

All dependencies are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

## Running the Notebook

1. Install the requirements (see above).
2. Launch Jupyter from this folder:
   ```bash
   jupyter notebook
   ```
3. Open `Basic_Machine_Learning_I_Practical.ipynb`.
4. Run all cells from top to bottom (**Kernel > Restart & Run All** is recommended for a clean run).

The notebook is fully self-contained: both datasets load directly from `seaborn`'s dataset
repository over the network, so an internet connection is required the first time each dataset is
loaded (results are cached locally by `seaborn` afterwards).

## Saved Models

Running the notebook creates a `models/` folder containing the trained models (and the Auto MPG
scaler) saved with `joblib`:

- `linear_regression_model.joblib` and `mpg_scaler.joblib`
- `logistic_regression_model.joblib`
- `knn_model.joblib`
- `svm_model.joblib`

Since the main flow doesn't use `Pipeline`, the scaler and model are saved as separate objects for
Auto MPG; the Titanic-based models don't need a separate saved scaler, since Practical 1 already
scaled the full Titanic feature set before the models were trained. To reuse a saved model:

```python
import joblib
model = joblib.load("models/knn_model.joblib")
model.predict(new_data)  # new_data must already match the training feature format
```

## Notes

- Random seeds are fixed throughout, so re-running the notebook reproduces the same results.
- Titanic is standardized *before* the train-test split (for simplicity in that walkthrough); Auto
  MPG is split *first*, then scaled on the training data only — the notebook calls out this
  difference explicitly, since scaling after splitting is the more rigorous approach in practice.
- Linear Regression's six assumptions are checked **after** the model is fit, since every diagnostic
  (residual plots, Shapiro-Wilk, Durbin-Watson, VIF) is computed from the model's own residuals. Each
  check states its possible value range, a full interpretation table where relevant (e.g. VIF and
  Durbin-Watson), and the range considered acceptable. On the real Auto MPG data, some of these
  checks genuinely fail (e.g. the engine-spec features show real multicollinearity), which is left
  as-is deliberately — it's a realistic example of assumptions not always holding.
- The notebook then measures the actual before/after impact of removing flagged features, rather
  than just asserting it: removing all four high-VIF features noticeably **hurts** performance (RMSE
  rises from 2.89 to 4.44), since those features still carry real predictive signal even though
  they're correlated with each other. Keeping just one representative feature (`weight`) recovers
  almost all of that lost performance (RMSE back down to 2.95) while still resolving the
  multicollinearity — a more realistic fix than dropping every correlated feature. Removing the
  statistically insignificant (p > 0.05) features, by contrast, leaves performance essentially
  unchanged, exactly as the p-values predicted.
- KNN's best K is chosen automatically via 5-fold cross-validation across K = 1 through 20, rather
  than hard-coded.
- Since KNN has no coefficients, its feature importance uses **permutation importance** instead — a
  model-agnostic technique that measures the accuracy drop when a feature's values are shuffled.
- The decision boundary plots for Logistic Regression, KNN, and SVM use a simplified model trained
  on only two features (Age and Fare), purely for visualization — the saved models use the full
  feature set.
