import pytest
from sklearn.svm import SVC

from bioproject.data_processing import get_ffnn_sequence, get_cnn_sequence
from bioproject.models import build_cnn1d, build_ffnn
from bioproject.model_evaluation import *


def test_evaluate_model_cnn(enhancers_train_test_bed, window_size, genome):
    X_train, X_test, y_train, y_test = enhancers_train_test_bed

    train_seq = get_cnn_sequence(genome, X_train, y_train.values.ravel())
    test_seq = get_cnn_sequence(genome, X_test, y_test.values.ravel())

    models = build_cnn1d(window_size)

    history, evaluations = evaluate_model(
        model=models["model"],
        model_name="test_model_cnn",
        region="enhancers",
        train_sequence=train_seq,
        valid_sequence=test_seq,
        test_sequence=test_seq,
        holdout_number=20,
        use_feature_selection=False,
        use_validation_set=False,
        window_size=window_size,
        epochs=1,
        patience=1
    )
    assert(len(evaluations) > 0)


def test_evaluate_model_ffnn(enhancers_train_test_epi, window_size):
    X_train, X_test, y_train, y_test = enhancers_train_test_epi

    train_seq = get_ffnn_sequence(X_train.values, y_train.values.ravel())
    test_seq = get_ffnn_sequence(X_test.values, y_test.values.ravel())

    models = build_ffnn(X_train.values)

    history, evaluations = evaluate_model(
        model=models["model"],
        model_name="test_model_ffnn",
        region="enhancers",
        train_sequence=train_seq,
        valid_sequence=test_seq,
        test_sequence=test_seq,
        holdout_number=20,
        use_feature_selection=False,
        use_validation_set=False,
        window_size=window_size,
        epochs=1,
        patience=1
    )
    assert(len(evaluations) > 0)


def test_evaluate_model_exception(enhancers_train_test_epi, window_size):
    X_train, X_test, y_train, y_test = enhancers_train_test_epi

    train_seq = get_ffnn_sequence(X_train.values, y_train.values.ravel())
    test_seq = get_ffnn_sequence(X_test.values, y_test.values.ravel())

    models = build_ffnn(X_train.values)

    with pytest.raises(ValueError):
        _, _ = evaluate_model(
            model=models["model"],
            model_name="test_model_ffnn",
            region="None",
            train_sequence=train_seq,
            valid_sequence=test_seq,
            test_sequence=test_seq,
            holdout_number=20,
            use_feature_selection=False,
            use_validation_set=False,
            window_size=window_size,
            epochs=1,
            patience=1
        )


def test_repeated_holdout_evaluation(
        enhancers_data, window_size, genome
):
    X_epigenomic, bed, y = enhancers_data

    models = {
        "ffnn": (
            lambda _X_train, _: build_ffnn(_X_train),
            lambda _genome, _bed, _X, _y: get_ffnn_sequence(_X, _y),
            {}
        )
    }

    _, evaluations, _, _ = repeated_holdout_evaluation(
        X_epi=X_epigenomic,
        bed=bed,
        y=y,
        genome=genome,
        models=models,
        region="enhancers",
        window_size=window_size,
        n_splits=1,
        no_feature_selection=True
    )
    assert(len(evaluations) > 0)


def test_repeated_holdout_evaluation_exception(
        enhancers_data, window_size, genome
):
    X_epigenomic, bed, y = enhancers_data

    models = {
        "ffnn": (
            lambda _X_train, _: build_ffnn(_X_train),
            lambda _genome, _bed, _X, _y: get_ffnn_sequence(_X, _y),
            {}
        )
    }

    with pytest.raises(ValueError):
        _, evaluations, _, _ = repeated_holdout_evaluation(
            X_epi=X_epigenomic,
            bed=bed,
            y=y,
            genome=genome,
            models=models,
            region="None",
            window_size=window_size,
            n_splits=1,
            no_feature_selection=True
        )


def test_evaluate_sklearn_model(
    enhancers_data, window_size, genome
):
    X_epigenomic, bed, y = enhancers_data

    def get_sklearn_sequence(X, y):
        return {"X": X, "y": y}

    models = {
        "svc": (
            lambda _X_train, _y_train: {"model": SVC()},
            lambda _genome, _bed, _X, _y: get_sklearn_sequence(_X, _y),
            {}
        )
    }

    _, evaluations, _, _ = repeated_holdout_evaluation(
        X_epi=X_epigenomic,
        bed=bed,
        y=y,
        genome=genome,
        models=models,
        region="enhancers",
        window_size=window_size,
        n_splits=1,
        no_feature_selection=True
    )

    assert len(evaluations) > 0

