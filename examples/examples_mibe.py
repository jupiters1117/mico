"""
Backward elimination approach for feature selection
===================================================

An introductory example that demonstrates how to perform feature selection using :class:`mico.MutualInformationBackwardElimination`.
"""
from mico import MutualInformationBackwardElimination
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes


def test_mibe_classification():

    print("=" * 80)
    print("Start classification example.")
    print("=" * 80)

    # Prepare data.
    data = load_breast_cancer()
    y = data.target
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # Perform feature selection.
    mibe = MutualInformationBackwardElimination(verbose=2, categorical=True, n_features=7)
    mibe.fit(X, y)

    print("-" * 80)
    print("Populate results.")
    # Populate selected features.
    print(" - Selected features: \n{}".format(mibe.get_support()))
    # Populate feature importance scores.
    print(" - Feature importance scores: \n{}".format(mibe.feature_importances_))
    # Call transform() on X.
    X_transformed = mibe.transform(X)
    print(" - X_transformed: \n{}".format(X_transformed))


def test_mibe_regression():

    print("=" * 80)
    print("Start regression example.")
    print("=" * 80)

    # Prepare data.
    data = load_diabetes()
    y = data.target
    X = pd.DataFrame(data.data, columns=data.feature_names)
    print(X)
    print(y)

    # Perform feature selection.
    mibe = MutualInformationBackwardElimination(verbose=2, num_bins=10, categorical=False, n_features=5)
    mibe.fit(X, y)

    print("-" * 80)
    print("Populate results.")
    # Populate selected features.
    print(" - Selected features: \n{}".format(mibe.get_support()))
    # Populate feature importance scores.
    print(" - Feature importance scores: \n{}".format(mibe.feature_importances_))
    # Call transform() on X.
    X_transformed = mibe.transform(X)
    print(" - X_transformed: \n{}".format(X_transformed))


if __name__ == '__main__':
    test_mibe_classification()
    test_mibe_regression()
