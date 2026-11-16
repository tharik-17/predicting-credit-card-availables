# Predicting Credit Card Approvals 💳

A machine learning project that predicts whether a credit card application will be **approved or denied**, using Logistic Regression on the UCI Credit Card Approval dataset.

**Author:** Kishore R

---

## 📌 Project Overview

Commercial banks receive a large number of credit card applications every day, and checking each one by hand is slow and error-prone. This project automates the decision with a classification model.

The notebook covers the full workflow: loading and inspecting the data, handling missing values, encoding and scaling features, selecting features by correlation, training a Logistic Regression model, and tuning it with grid search.

## 📂 Dataset

- **Source:** [Credit Approval dataset, UCI Machine Learning Repository](http://archive.ics.uci.edu/ml/datasets/credit+approval)
- **Size:** 690 applications, 15 features plus 1 target (16 columns)
- **Target (column 15):** `+` = approved (307, 44.5%), `-` = denied (383, 55.5%)
- **Features:** anonymised to protect privacy. They are a mix of numeric (columns 2, 7, 10, 14) and categorical columns.
- **Missing values:** marked as `?` in the raw file (columns 0, 1, 3, 4, 5, 6 and 13)

Place the data file at `dataset/cc_approvals.data`.

## 🔄 Methodology

1. **Load and inspect** the data (summary statistics, data types, class balance pie chart).
2. **Handle missing values:** replace `?` with `NaN`, then fill the gaps with the most frequent value.
3. **Encode categorical features** to numbers with `LabelEncoder`.
4. **Correlation analysis:** Pearson correlation heatmap, then keep features whose absolute correlation with the target is above 0.15. This gives features **2, 3, 4, 7, 8, 9, 10, 14**.
5. **Split** the data 67% train / 33% test (stratified, `random_state=42`).
6. **Scale** features to the 0 to 1 range with `MinMaxScaler`.
7. **Train** a `LogisticRegression` classifier.
8. **Evaluate** with accuracy and a confusion matrix on the test set.
9. **Tune** `tol` and `max_iter` with `GridSearchCV` (5-fold cross-validation).

## 🛠️ Technologies Used

- Python 3
- Pandas, NumPy
- Matplotlib, Seaborn
- scikit-learn (`LabelEncoder`, `MinMaxScaler`, `LogisticRegression`, `GridSearchCV`)
- Jupyter Notebook

## 📊 Results

| Metric | Value |
|---|---|
| Test accuracy (Logistic Regression) | **84.65%** |
| Best cross-validation accuracy (GridSearchCV) | **84.06%** |
| Best parameters | `max_iter=100`, `tol=0.001` |

**Confusion matrix (test set, 228 applications):**

| | Predicted denied | Predicted approved |
|---|---|---|
| **Actual denied** | 88 | 13 |
| **Actual approved** | 22 | 105 |

From this matrix, precision for approvals is about 0.89 and recall is about 0.83. The model clearly beats the majority-class baseline of 55.5% (always predicting "denied").

## ⚙️ Installation and Usage

```bash
# 1. Clone the repository
git clone <your-repository-url>
cd <your-repository-folder>

# 2. Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn jupyter

# 3. Add the dataset
#    Save the UCI file as dataset/cc_approvals.data

# 4. Launch the notebook
jupyter notebook predicting_credit_card_approvals.ipynb
```

## 📁 Project Structure

```
├── dataset/
│   └── cc_approvals.data
├── predicting_credit_card_approvals.ipynb
└── README.md
```

## ⚠️ Known Limitations

- Missing values are filled with a single most-frequent value rather than per-column modes, so imputation could be improved.
- The scaler is fitted separately on the train and test sets, and again on the full data for grid search. Fitting it on the training set only and reusing it for the test set would avoid data leakage.
- Only `tol` and `max_iter` are tuned; `C` and `penalty` are not.
- The test accuracy is from a single train/test split.

## 🔮 Future Scope

- Try Random Forest and XGBoost.
- Combine models with majority voting (ensemble).
- Tune more hyperparameters (`C`, `penalty`) and report precision, recall and F1-score.
- Wrap the model in a scikit-learn `Pipeline` for cleaner preprocessing.
- Build a simple web interface for checking a new application.

## 📜 License

This project is created for educational purposes.
