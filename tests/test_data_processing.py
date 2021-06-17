import pytest
import numpy as np
import pandas as pd
from epigenomic_dataset import (
    active_enhancers_vs_inactive_enhancers,
    active_promoters_vs_inactive_promoters
)
from pytest import approx

from bioproject.data_processing import (
    fill_data_dictionaries, flat_one_hot_encode, get_pos_neg, get_initial_output_bias, get_class_weights,
    missing_values_report, execute_preprocessing, execute_boruta_feature_selection, get_ffnn_sequence, get_cnn_sequence,
    get_mmnn_sequence,
)


@pytest.fixture
def data_dictionaries(genome, window_size, cell_line):
    beds, epigenomes = dict(), dict()
    labels, sequences = dict(), dict()

    for (task, region) in (
            (
                    (active_enhancers_vs_inactive_enhancers, "enhancers"),
                    (active_promoters_vs_inactive_promoters, "promoters")
            )
    ):
        beds, epigenomes, labels, sequences = fill_data_dictionaries(
            task=task,
            region=region,
            genome=genome,
            beds=beds,
            epigenomes=epigenomes,
            labels=labels,
            sequences=sequences,
            cell_line=cell_line,
            window_size=window_size
        )
    return beds, epigenomes, labels, sequences


def assert_first_batch(seq):
    assert np.equal(seq, np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0,
                                   1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0,
                                   0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
                                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                   0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
                                   0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,
                                   0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
                                   0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])).all()


def test_flat_one_hot_encode(genome, enhancers_data, window_size):
    _, bed, _ = enhancers_data
    fohe = flat_one_hot_encode(
        genome=genome,
        bed=bed,
        window_size=window_size
    )
    assert (fohe.values.all() == 0 or fohe.values.all() == 1)


def test_fill_data_dictionaries(genome, cell_line, window_size):
    region = "enhancers"

    beds, epigenomes = dict(), dict()
    labels, sequences = dict(), dict()

    beds, epigenomes, labels, sequences = fill_data_dictionaries(
        task=active_enhancers_vs_inactive_enhancers,
        region=region,
        genome=genome,
        beds=beds,
        epigenomes=epigenomes,
        labels=labels,
        sequences=sequences,
        cell_line=cell_line,
        window_size=window_size,
    )

    assert (len(beds) > 0)
    assert (len(epigenomes) > 0)
    assert (len(labels) > 0)
    assert (len(sequences) > 0)


def test_get_pos_neg():
    df = pd.DataFrame([0, 0, 0, 1, 1])
    pos, neg = get_pos_neg(df)
    assert (pos == 2)
    assert (neg == 3)


def test_get_initial_output_bias():
    df = pd.DataFrame([0, 0, 0, 1, 1])
    initializer = get_initial_output_bias(df)
    assert initializer.value == approx(np.array([[-0.40546511]]))


def test_get_class_weights():
    df = pd.DataFrame([0, 0, 0, 1, 1])
    dd1 = get_class_weights(df)
    dd2 = {0: np.array([0.83333333]), 1: np.array([1.25])}
    assert dd1.keys() == dd2.keys()
    for v1, v2 in zip(dd1.values(), dd2.values()):
        assert v1 == approx(v2)


def test_missing_values_report(genome, data_dictionaries, capsys):
    beds, epigenomes, _, _ = data_dictionaries
    missing_values_report(epigenomes=epigenomes, beds=beds, genome=genome)
    captured = capsys.readouterr()
    cps = captured.out.split("\n")
    comp = """=========================
    ENHANCERS EPIGENOMIC DATA
    =========================
    The dataset contains 102 NaN values out of 27149265 total values.
    Sample (row) number 63272 has the most NaN values, amounting to 7 NaN values out of 429 row values.
    Feature (column) number 140 (whole-genome shotgun bisulfite sequencing) has the most NaN values, amounting to 93 NaN values out of 63285 column values.

    =========================
    PROMOTERS EPIGENOMIC DATA
    =========================
    The dataset contains 496 NaN values out of 42848949 total values.
    Sample (row) number 92631 has the most NaN values, amounting to 20 NaN values out of 429 row values.
    Feature (column) number 140 (whole-genome shotgun bisulfite sequencing) has the most NaN values, amounting to 250 NaN values out of 99881 column values.

    =======================
    ENHANCERS SEQUENCE DATA
    =======================
    The dataset contains 1 unidentified nucleotides.
    =======================
    PROMOTERS SEQUENCE DATA
    =======================
    The dataset contains 4 unidentified nucleotides.""".split("\n")
    for c1, c2 in zip(cps, comp):
        assert c1.strip() == c2.strip()


def test_execute_preprocessing(random_dataset):
    X_train, X_valid, X_test, _, _, _ = random_dataset
    new_X_train, _, _ = execute_preprocessing(
        X_train=X_train, X_valid=X_valid, X_test=X_test
    )
    assert new_X_train.values != approx(X_train)


def test_execute_preprocessing_imputer_exc(random_dataset):
    X_train, X_valid, X_test, _, _, _ = random_dataset

    with pytest.raises(ValueError):
        new_X_train, _, _ = execute_preprocessing(
            X_train=X_train, X_valid=X_valid, X_test=X_test,
            imputer_type="imputer"
        )


def test_execute_preprocessing_scaler_exc(random_dataset):
    X_train, X_valid, X_test, _, _, _ = random_dataset

    with pytest.raises(ValueError):
        new_X_train, _, _ = execute_preprocessing(
            X_train=X_train, X_valid=X_valid, X_test=X_test,
            scaler_type="scaler"
        )


def test_execute_boruta_feature_selection(random_dataset):
    X_train, _, _, y_train, _, _ = random_dataset
    kept, _ = execute_boruta_feature_selection(
        X_train, y_train.values.ravel(), 20, False, 256, "enhancers"
    )
    assert len(kept) > 0


def test_get_ffnn_sequence(random_dataset):
    X_train, _, _, y_train, _, _ = random_dataset
    seq = get_ffnn_sequence(
        X_train.values, y_train.values.ravel()
    )
    assert np.equal(np.array(seq)[0][1],
                    np.array([1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1,
                              0, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0,
                              0, 1, 1, 0, 1, 0])).all()


def test_get_cnn_sequence(data_dictionaries, genome):
    beds, _, labels, _ = data_dictionaries
    seq = get_cnn_sequence(
        genome, beds["enhancers"], labels["enhancers"].values.ravel()
    )
    assert (np.array(seq)[0][1][0] == 0) or (np.array(seq)[0][1][0] == 1)


def test_get_mmnn_sequence(data_dictionaries, genome):
    beds, epigenomes, labels, _ = data_dictionaries
    seq = get_mmnn_sequence(
        genome,
        beds["enhancers"],
        epigenomes["enhancers"].values,
        labels["enhancers"].values.ravel()
    )
    assert (np.array(seq)[0][1][0] == 0) or (np.array(seq)[0][1][0] == 1)
