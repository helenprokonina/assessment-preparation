# Credit Default Risk Assessment

This project develops and evaluates binary classification models that estimate an applicant's probability of credit default. The analysis compares regularized logistic regression with XGBoost and explains the final tree model using SHAP and LIME.

The complete analysis is in [notebook.ipynb](notebook.ipynb). The source data is stored in `data/credit risk_data.csv`.

## Business objective

The target variable is `default`:

- `1`: the borrower defaulted.
- `0`: the credit was repaid as agreed.

A false negative approves a borrower who later defaults, while a false positive rejects or restricts a reliable borrower. Because an undetected default is assumed to be more costly, the analysis emphasizes recall for class 1 while monitoring precision and the number of false positives.

## Project structure

```text
CreditRisk/
|-- data/
|   `-- credit risk_data.csv
|-- .gitignore
|-- notebook.ipynb
|-- pyproject.toml
`-- README.md
```

## Setup

Python 3.13 or later is required by `pyproject.toml`.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e .
python -m pip install jupyter ipykernel
```

Open `notebook.ipynb` in VS Code or Jupyter and select the `.venv` interpreter as the kernel. Run the cells from top to bottom because later sections use fitted objects created earlier.

## Notebook workflow

### 1. Load and inspect the data

The notebook reads the CSV file with pandas and inspects its shape, data types, descriptive statistics, missing values, duplicate rows, unique values, and target distribution.

### 2. Exploratory data analysis

Categorical features are examined using category counts, observed default rates, and Cramer's V. Numerical features are examined with class-specific distributions and a Pearson correlation heatmap. These results describe associations and do not establish causal relationships.

### 3. Split the data

The data is divided into an 80% training set and a 20% test set with stratification so both sets retain approximately the same default rate. The test set remains untouched during model and threshold selection.

Five-fold stratified cross-validation is used within the training set.

### 4. Preprocess the features

A `ColumnTransformer` applies:

- `StandardScaler` to numerical features.
- `OneHotEncoder` to categorical features, with unknown categories ignored and one reference category dropped.

Preprocessing and modeling are combined in scikit-learn pipelines. This prevents validation data from influencing fitted preprocessing parameters.

The model therefore sees transformed features. Numerical variables are expressed in standard-deviation units, and each categorical variable is expanded into binary indicator columns.

### 5. Tune logistic regression

`GridSearchCV` compares combinations of:

- Regularization strength `C`.
- L1 and L2 penalties.
- Unweighted and balanced class weights.

Each combination is evaluated by average precision, recall, and the recall-weighted $F_2$ score. Average precision is the refit metric, so the configuration with the best cross-validated probability ranking is selected and retrained on the full training set.

### 6. Select a classification threshold

`cross_val_predict` generates an out-of-fold default probability for every training observation. No observation is predicted by a model that was trained on that same observation.

The threshold that maximizes $F_2$ is selected from these probabilities:

$$
F_2 = \frac{5 \cdot \text{precision} \cdot \text{recall}}
{4 \cdot \text{precision} + \text{recall}}
$$

This gives recall more influence than precision. Threshold selection uses training data only; the chosen threshold is then applied once to the test probabilities.

### 7. Tune XGBoost

Optuna searches XGBoost hyperparameters such as tree depth, learning rate, number of estimators, row and column sampling, regularization, and positive-class weight. Every trial is evaluated by mean cross-validated average precision on the training set.

The best XGBoost configuration receives the same out-of-fold $F_2$ threshold-selection procedure used for logistic regression.

### 8. Evaluate the models

The notebook reports:

- Average precision (a threshold-independent summary of the precision-recall curve).
- ROC-AUC.
- Precision, recall, $F_1$, and $F_2$ for default.
- Classification reports and confusion matrices.
- Precision-recall curves with the positive-class prevalence as the no-skill baseline.

The final choice should consider both ranking performance and the operational effect of the selected threshold. Accuracy alone is not appropriate for this imbalanced, cost-sensitive problem.

### 9. Explain XGBoost predictions

SHAP explains the fitted XGBoost model on transformed test features:

- The bar plot ranks features by mean absolute SHAP value.
- The beeswarm plot shows the direction and magnitude of each feature's contribution across applicants. Positive SHAP values push the model toward default; negative values push it toward no default.

LIME explains the test applicant with the highest predicted default probability. It generates nearby synthetic observations and fits a simple local surrogate model. Its bars show which feature conditions locally support default or no default.

Because explanations operate on preprocessed data, numerical LIME conditions are standardized values and categorical conditions refer to one-hot indicators. SHAP and LIME explain model behavior; they do not prove that changing a feature would cause default risk to change.

## Methodological limitations

- The dataset is small, so test-set differences have substantial uncertainty.
- The generic $F_2$ objective is only a proxy for business cost. A production threshold should use validated losses, review costs, approval revenue, and policy constraints.
- Probability calibration should be checked before predicted probabilities are used as absolute risk estimates.
- Temporal stability, subgroup fairness, data drift, and explanation stability require validation on recent out-of-time data.
- Model explanations are predictive attributions, not causal effects.
