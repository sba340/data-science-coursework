"""06. Validation strategies and model comparison on the breast cancer data set.

Part A compares evaluation schemes for one model (Gaussian Naive Bayes):
  * a single 50/50 holdout split
  * 5-, 10-, and 20-fold stratified cross-validation
Part B compares three classifiers under the same 10-fold cross-validation:
  * Gaussian Naive Bayes
  * Support Vector Machine, with and without feature scaling
  * Random Forest

Data: scikit-learn's built-in Wisconsin breast cancer set (569 samples, 30 numeric
features; target 0 = malignant, 1 = benign). No download is needed.

Usage:
    python 06_ml_validation.py
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

SEED = 54321


def holdout_accuracy(model, X, y, test_size: float = 0.5) -> float:
    """Accuracy of a model trained on one split and tested on the rest."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=SEED, stratify=y)
    return accuracy_score(y_test, model.fit(X_train, y_train).predict(X_test))


def cv_scores(model, X, y, folds: int):
    """Stratified k-fold accuracy scores. Shuffling with a fixed seed makes runs repeatable."""
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=SEED)
    return cross_val_score(model, X, y, cv=splitter, scoring="accuracy")


def main() -> None:
    X, y = load_breast_cancer(return_X_y=True, as_frame=True)
    print(f"{X.shape[0]} samples, {X.shape[1]} features; "
          f"{(y == 1).sum()} benign, {(y == 0).sum()} malignant\n")

    # Part A: validation schemes with a fixed model
    rows = [{"scheme": "50/50 holdout", "mean_accuracy": holdout_accuracy(GaussianNB(), X, y),
             "std": float("nan")}]
    for folds in (5, 10, 20):
        scores = cv_scores(GaussianNB(), X, y, folds)
        rows.append({"scheme": f"{folds}-fold CV", "mean_accuracy": scores.mean(), "std": scores.std()})
    print("Part A: GaussianNB under different validation schemes")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # Part B: model comparison under the same 10-fold splits
    models = {
        "GaussianNB": GaussianNB(),
        "SVC (raw features)": SVC(),
        "SVC (standardized)": make_pipeline(StandardScaler(), SVC()),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=SEED),
    }
    rows = []
    for name, model in models.items():
        scores = cv_scores(model, X, y, 10)
        rows.append({"model": name, "mean_accuracy": scores.mean(), "std": scores.std()})
    print("\nPart B: model comparison, 10-fold CV")
    print(pd.DataFrame(rows).round(4).to_string(index=False))

    # How to read these results:
    #  * A single holdout uses one split, so its score depends on which rows landed in the
    #    test half; it also trains on only half the data. Cross-validation tests every row
    #    once and trains on 80% to 95% of the data per fold, so its estimate is more stable.
    #    Here the two schemes agree closely (about 0.937 versus 0.940), a difference well
    #    inside one fold-to-fold standard deviation, so neither scheme is clearly "better".
    #  * Means that barely move between 5, 10, and 20 folds, with standard deviations that are
    #    small compared with the gaps between models, indicate stable estimates. A difference
    #    between two models smaller than about one standard deviation should not be over-read.
    #  * An SVM is sensitive to feature scale, and these 30 features span very different
    #    ranges. Its weak raw-feature score (about 0.914) reflects missing preprocessing, not
    #    a limit of the method. Standardizing inside a pipeline, so scaling is learned on
    #    training folds only and avoids leakage, lifts it to about 0.977, the best score here.
    #  * Random Forest handles unscaled, correlated features and nonlinear structure without
    #    tuning, which makes it a strong baseline on tabular data like this.


if __name__ == "__main__":
    main()
