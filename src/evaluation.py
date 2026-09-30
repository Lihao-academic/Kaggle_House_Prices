# 本文件保存执行交叉验证并统计数据的代码


import numpy as np

from sklearn.model_selection import KFold, cross_validate


SCORINGS = {
    "RMSE": "neg_root_mean_squared_error",
    "MAE": "neg_mean_absolute_error",
    "MSE": "neg_mean_squared_error",
    "R2": "r2"
}


def evaluate_cv(
        pipeline,
        X,
        y,
        n_splits=5,
        random_state=42,
        return_estimator=True,
        n_jobs=None
):

    cv = KFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    scores = cross_validate(
        pipeline,
        X,
        np.log(y),
        cv=cv,
        scoring=SCORINGS,
        return_train_score=True,
        return_estimator=return_estimator,
        n_jobs=n_jobs
    )

    return scores


import pandas as pd


def summarize_scores(scores):

    metrics = ["RMSE", "MAE", "MSE", "R2"]

    results = []

    for metric in metrics:

        train_values = scores[f"train_{metric}"]
        test_values = scores[f"test_{metric}"]

        if metric != "R2":
            train_values = -train_values
            test_values = -test_values

        results.append({
            "Metric": f"train_{metric}",
            "Mean": train_values.mean(),
            "Std": train_values.std()
        })

        results.append({
            "Metric": f"test_{metric}",
            "Mean": test_values.mean(),
            "Std": test_values.std()
        })

    return pd.DataFrame(results)


def extract_model_info(scores):

    information = []

    for fold, estimator in enumerate(
        scores["estimator"],
        start=1
    ):

        model = estimator.named_steps["LRmodel"]

        fold_info = {
            "fold": fold,
            "n_features": model.n_features_in_
        }

        if "selector" in estimator.named_steps:

            selector = estimator.named_steps["selector"]

            fold_info["selected_features"] = (
                selector.selected_columns_
            )

        information.append(fold_info)

    return information


