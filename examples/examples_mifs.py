"""
Forward selection approach for feature selection
================================================

An introductory example that demonstrates how to perform feature selection using :class:`mico.MutualInformationForwardSelection`.
"""
from mico import MutualInformationForwardSelection
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes


def test_mifs_classification():

    print("=" * 80)
    print("Start classification example.")
    print("=" * 80)

    # Prepare data.
    data = load_breast_cancer()
    y = data.target
    X = pd.DataFrame(data.data, columns=data.feature_names)

    # Perform feature selection.
    mifs = MutualInformationForwardSelection(verbose=2, categorical=True, n_features=7)
    mifs.fit(X, y)

    print("-" * 80)
    print("Populate results.")
    # Populate selected features.
    print(" - Selected features: \n{}".format(mifs.get_support()))
    # Populate feature importance scores.
    print(" - Feature importance scores: \n{}".format(mifs.feature_importances_))
    # Call transform() on X.
    X_transformed = mifs.transform(X)
    print(" - X_transformed: \n{}".format(X_transformed))


def test_mifs_regression():

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
    mifs = MutualInformationForwardSelection(verbose=2, num_bins=10, categorical=False, n_features=5)
    mifs.fit(X, y)

    print("-" * 80)
    print("Populate results.")
    # Populate selected features.
    print(" - Selected features: \n{}".format(mifs.get_support()))
    # Populate feature importance scores.
    print(" - Feature importance scores: \n{}".format(mifs.feature_importances_))
    # Call transform() on X.
    X_transformed = mifs.transform(X)
    print(" - X_transformed: \n{}".format(X_transformed))


if __name__ == '__main__':
    test_mifs_classification()
    test_mifs_regression()