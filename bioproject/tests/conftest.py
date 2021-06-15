import pandas as pd
import pytest
from epigenomic_dataset import active_enhancers_vs_inactive_enhancers
from sklearn.datasets import make_classification
from sklearn.impute import KNNImputer
from sklearn.model_selection import (
    StratifiedShuffleSplit, train_test_split
)
from ucsc_genomes_downloader import Genome


@pytest.fixture
def genome():
    return Genome("hg38")


@pytest.fixture
def window_size():
    return 256


@pytest.fixture
def cell_line():
    return "K562"


@pytest.fixture
def random_dataset():
    X, y = make_classification(50, 5)
    X_full, X_test, y_full, y_test = train_test_split(
        X, y,
        test_size=.2,
        random_state=42,
        stratify=y
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_full, y_full,
        test_size=.2,
        random_state=42,
        stratify=y_full
    )
    return pd.DataFrame(X_train), pd.DataFrame(X_valid), \
        pd.DataFrame(X_test), pd.DataFrame(y_train), \
        pd.DataFrame(y_valid), pd.DataFrame(y_test)


@pytest.fixture
def enhancers_data(window_size, cell_line):
    X, y = active_enhancers_vs_inactive_enhancers(
        cell_line=cell_line,
        window_size=window_size
    )
    y = pd.DataFrame(y[cell_line].values)
    bed = X.reset_index().rename_axis(None, axis=1)
    X_epigenomic = bed[bed.columns[4:]]
    return X_epigenomic, bed, y


def get_train_test(X, y):
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2
    ).split(X, y)
    train_indices, test_indices = next(splitter)

    X_train, y_train = X.iloc[train_indices], y.iloc[train_indices]
    X_test, y_test = X.iloc[test_indices], y.iloc[test_indices]
    return X_train, X_test, y_train, y_test


@pytest.fixture
def enhancers_train_test_epi(enhancers_data):
    X_epigenomic, bed, y = enhancers_data
    imputer = KNNImputer()
    X_epigenomic = pd.DataFrame(
        imputer.fit_transform(X_epigenomic),
        columns=X_epigenomic.columns
    )
    return get_train_test(X_epigenomic, y)


@pytest.fixture
def enhancers_train_test_bed(enhancers_data):
    X_epigenomic, bed, y = enhancers_data
    return get_train_test(bed, y)
