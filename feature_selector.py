# feature_selector.py

import numpy as np
import pandas as pd

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_regression,
    RFE
)
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline


# Utility functions
def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _r2(y_true, y_pred):
    return r2_score(y_true, y_pred)


def _evaluate_model(X_train, X_test, y_train, y_test, selected_idx):
    """
    Fit ordinary linear regression using selected features
    and return train/test RMSE and R2.
    """

    selected_idx = list(selected_idx)

    Xtrain = X_train.iloc[:, selected_idx]
    Xtest = X_test.iloc[:, selected_idx]

    model = LinearRegression()
    model.fit(Xtrain, y_train)

    train_pred = model.predict(Xtrain)
    test_pred = model.predict(Xtest)

    return {
        "train_rmse": _rmse(y_train, train_pred),
        "test_rmse": _rmse(y_test, test_pred),
        "train_r2": _r2(y_train, train_pred),
        "test_r2": _r2(y_test, test_pred),
    }


def _cross_validation_rmse(X, y, selected_idx, cv=5):
    """
    Calculate CV RMSE for the selected features.
    """

    selected_idx = list(selected_idx)
    X_selected = X.iloc[:, selected_idx]

    model = LinearRegression()
    kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_selected, y, cv=kfold, scoring="neg_root_mean_squared_error")

    return -scores.mean()

# Feature Selector
class FeatureSelector:
    def __init__(self, method, max_features=None, random_state=42):
        self.method = method.lower()
        self.max_features = max_features
        self.random_state = random_state

    # Preprocessing
    def _prepare_features(self, X_train, X_test):
        """
        Convert categorical columns into dummy variables.
        """

        X_train = X_train.copy()
        X_test = X_test.copy()

        numeric_cols = X_train.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = X_train.select_dtypes(exclude=["number"]).columns.tolist()

        # Numeric features
        if numeric_cols:
            num_imputer = SimpleImputer(strategy="median")
            X_train_num = pd.DataFrame(num_imputer.fit_transform(X_train[numeric_cols]), columns=numeric_cols, index=X_train.index)
            X_test_num = pd.DataFrame(num_imputer.transform(X_test[numeric_cols]), columns=numeric_cols, index=X_test.index)
        else:
            X_train_num = pd.DataFrame(index=X_train.index)
            X_test_num = pd.DataFrame(index=X_test.index)

        # Categorical features
        if categorical_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X_train_cat = cat_imputer.fit_transform(X_train[categorical_cols])
            X_test_cat = cat_imputer.transform(X_test[categorical_cols])

            encoder = OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False)
            X_train_encoded = encoder.fit_transform(X_train_cat)
            X_test_encoded = encoder.transform(X_test_cat)
        
            encoded_names = encoder.get_feature_names_out(categorical_cols)
            X_train_cat = pd.DataFrame(X_train_encoded, columns=encoded_names, index=X_train.index)
            X_test_cat = pd.DataFrame(X_test_encoded, columns=encoded_names, index=X_test.index)
        else:
            X_train_cat = pd.DataFrame(index=X_train.index)
            X_test_cat = pd.DataFrame(index=X_test.index)

        # Combine
        X_train_processed = pd.concat([X_train_num, X_train_cat], axis=1)
        X_test_processed = pd.concat([X_test_num, X_test_cat], axis=1)

        return X_train_processed, X_test_processed
    
    # Feature ranking
    def _rank_variance(self, X):
        """
        Rank features according to variance.
        """

        variances = X.var(axis=0)
        ranking = variances.sort_values(ascending=False).index.tolist()

        return ranking

    def _rank_mutual_information(self, X, y):
        """
        Rank features using Mutual Information.
        """

        scores = mutual_info_regression(X, y, random_state=self.random_state)
        ranking = pd.Series(scores, index=X.columns).sort_values(ascending=False).index.tolist()
        
        return ranking

    def _rank_forward(self, X, y):
        """
        Forward feature selection ranking.

        The first selected feature is the best single predictor.
        Then features are added one at a time.
        """

        remaining = list(X.columns)
        selected = []

        while remaining:

            best_feature = None
            best_score = np.inf

            for feature in remaining:

                current_features = selected + [feature]

                model = LinearRegression()
                model.fit(X[current_features], y)

                prediction = model.predict(X[current_features])
                score = _rmse(y, prediction)

                if score < best_score:
                    best_score = score
                    best_feature = feature

            selected.append(best_feature)
            remaining.remove(best_feature)

        return selected

    def _rank_backward(self, X, y):
        """
        Backward feature elimination ranking.

        Features are removed from the model one by one.
        """

        remaining = list(X.columns)
        removed = []

        while len(remaining) > 1:

            worst_feature = None
            best_score = np.inf

            for feature in remaining:

                candidate = [f for f in remaining if f != feature]

                model = LinearRegression()
                model.fit(X[candidate], y)
                prediction = model.predict(X[candidate])
                score = _rmse(y, prediction)

                if score < best_score:
                    best_score = score
                    worst_feature = feature

            removed.append(worst_feature)
            remaining.remove(worst_feature)

        ranking = remaining + removed[::-1]

        return ranking

    def _rank_rfe(self, X, y):
        """
        Recursive Feature Elimination ranking.
        """

        estimator = LinearRegression()

        rfe = RFE(estimator=estimator, n_features_to_select=1, step=1)
        rfe.fit(X, y)

        ranking_df = pd.DataFrame({
            "feature": X.columns,
            "rank": rfe.ranking_
        })

        ranking_df = ranking_df.sort_values("rank")

        return ranking_df["feature"].tolist()

    def select_from_processed(self, X, y):
        """
        Return features ranked from most important to least important.
        """

        if self.method == "variance_threshold":
            return self._rank_variance(X)

        elif self.method == "mutual_information":
            return self._rank_mutual_information(X, y)

        elif self.method == "forward":
            return self._rank_forward(X, y)

        elif self.method == "backward":
            return self._rank_backward(X, y)

        elif self.method == "rfe":
            return self._rank_rfe(X, y)

        else:
            raise ValueError(f"Unsupported method: {self.method}")

# Standard feature-count experiments
def feature_count_experiment(X_train, X_test, y_train, y_test, method, max_features=None, cv=5, random_state=42):
    """
    Run feature selection for k = 1 ... N.

    Used for:
        - Variance Threshold
        - Mutual Information
        - Forward
        - Backward
        - RFE
    """

    selector = FeatureSelector(method=method, max_features=max_features, random_state=random_state)
    X_train_processed, X_test_processed = selector._prepare_features(X_train, X_test)

    feature_names = X_train_processed.columns.tolist()
    n_features = len(feature_names)

    if max_features is None:
        max_features = n_features

    max_features = min(max_features, n_features)
    ranking = selector.select_from_processed(X_train_processed, y_train)

    results = []

    for k in range(1, max_features + 1):

        selected_features = ranking[:k]
        selected_idx = [feature_names.index(feature) for feature in selected_features]

        metrics = _evaluate_model(X_train_processed, X_test_processed, y_train, y_test, selected_idx)
        cv_rmse = _cross_validation_rmse(X_train_processed, y_train, selected_idx, cv=cv)

        results.append({
            "n_features": k,
            "selected_features": ", ".join(selected_features),
            "train_rmse": metrics["train_rmse"],
            "test_rmse": metrics["test_rmse"],
            "train_r2": metrics["train_r2"],
            "test_r2": metrics["test_r2"],
            "cv_rmse": cv_rmse
        })

    return pd.DataFrame(results)

# LASSO
def lasso_experiment(X_train, X_test, y_train, y_test, max_features=None, alpha=0.01, cv=5, random_state=42):
    """
    Lasso feature ranking.

    Process:

    1. Preprocess categorical features.
    2. Standardize features.
    3. Fit Lasso once.
    4. Rank features by absolute Lasso coefficient.
    5. For k = 1 ... max_features:
         - select top-k Lasso features
         - fit ordinary Linear Regression
         - calculate RMSE and R2

    This matches the feature-count approach used
    by the teacher's Multiple Linear Regression explorer.
    """

    method = "lasso"

    selector = FeatureSelector(method=method, max_features=max_features, random_state=random_state)
    X_train_processed, X_test_processed = selector._prepare_features(X_train, X_test)

    feature_names = X_train_processed.columns.tolist()
    n_features = len(feature_names)

    if max_features is None:
        max_features = n_features

    max_features = min(max_features, n_features)

    # Standardize for Lasso
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)

    model = Lasso(alpha=alpha, max_iter=100000, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    coefficients = np.abs(model.coef_)

    ranking_indices = np.argsort(coefficients)[::-1]
    ranking = [feature_names[i] for i in ranking_indices]

    # Test k = 1 ... max_features
    results = []

    for k in range(1, max_features + 1):

        selected_features = ranking[:k]
        selected_idx = [feature_names.index(feature) for feature in selected_features]

        metrics = _evaluate_model(X_train_processed, X_test_processed, y_train, y_test, selected_idx)
        cv_rmse = _cross_validation_rmse(X_train_processed, y_train, selected_idx, cv=cv)

        results.append({
            "n_features": k,
            "selected_features": ", ".join(selected_features),
            "train_rmse": metrics["train_rmse"],
            "test_rmse": metrics["test_rmse"],
            "train_r2": metrics["train_r2"],
            "test_r2": metrics["test_r2"],
            "cv_rmse": cv_rmse
        })

    return pd.DataFrame(results)

# ELASTIC NET
def elastic_net_experiment(X_train, X_test, y_train, y_test, max_features=None, alpha=0.01, l1_ratio=0.5, cv=5, random_state=42):
    """
    Elastic Net feature ranking.

    Process:

    1. Preprocess categorical features.
    2. Standardize features.
    3. Fit Elastic Net once.
    4. Rank features by absolute coefficient.
    5. For k = 1 ... max_features:
         - select top-k features
         - fit ordinary Linear Regression
         - calculate RMSE and R2
    """

    method = "elastic_net"

    selector = FeatureSelector(method=method, max_features=max_features, random_state=random_state)
    X_train_processed, X_test_processed = selector._prepare_features(X_train, X_test)

    feature_names = X_train_processed.columns.tolist()
    n_features = len(feature_names)

    if max_features is None:
        max_features = n_features

    max_features = min(max_features, n_features)

    # Standardize for Elastic Net
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_processed)

    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=100000, random_state=random_state)
    model.fit(X_train_scaled, y_train)

    coefficients = np.abs(model.coef_)
    ranking_indices = np.argsort(coefficients)[::-1]
    ranking = [feature_names[i] for i in ranking_indices]

    # Test k = 1 ... max_features
    results = []

    for k in range(1, max_features + 1):

        selected_features = ranking[:k]
        selected_idx = [feature_names.index(feature) for feature in selected_features]

        metrics = _evaluate_model(X_train_processed, X_test_processed, y_train, y_test, selected_idx)
        cv_rmse = _cross_validation_rmse(X_train_processed, y_train, selected_idx, cv=cv)

        results.append({
            "n_features": k,
            "selected_features": ", ".join(selected_features),
            "train_rmse": metrics["train_rmse"],
            "test_rmse": metrics["test_rmse"],
            "train_r2": metrics["train_r2"],
            "test_r2": metrics["test_r2"],
            "cv_rmse": cv_rmse
        })

    return pd.DataFrame(results)