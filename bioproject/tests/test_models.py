import kerastuner as kt
from kerastuner import Hyperband

from bioproject.data_processing import get_cnn_sequence
from bioproject.models import *


def test_build_ffnn_iwbbio2020(random_dataset, capsys):
    X_train, _, _, _, _, _ = random_dataset
    build_ffnn_iwbbio2020(X_train)["model"].summary()

    captured = capsys.readouterr()
    assert "ffnn_IWBBIO2020" in captured.out


def test_build_ffnn(random_dataset, capsys):
    X_train, _, _, _, _, _ = random_dataset
    build_ffnn(X_train)["model"].summary()

    captured = capsys.readouterr()
    assert "ffnn" in captured.out


def test_build_cnn_iwbbio2020(window_size, capsys):
    build_cnn_iwbbio2020(window_size)["model"].summary()

    captured = capsys.readouterr()
    assert "cnn_IWBBIO2020" in captured.out


def test_build_deep_enhancer(window_size, capsys):
    build_deep_enhancer(window_size)["model"].summary()

    captured = capsys.readouterr()
    assert "deep_enhancer" in captured.out


def test_build_deepcape(window_size, capsys):
    build_deepcape(window_size)["model"].summary()

    captured = capsys.readouterr()
    assert "deepcape" in captured.out


def test_build_decode(enhancers_data, capsys):
    X_epigenomic, _, _ = enhancers_data
    build_decode(X_epigenomic.values)["model"].summary()

    captured = capsys.readouterr()
    assert "decode" in captured.out


def test_build_cnn1d(window_size, capsys):
    build_cnn1d(window_size)["model"].summary()

    captured = capsys.readouterr()
    assert "cnn1d" in captured.out


def test_build_mmnn_not_pretrained(enhancers_data, window_size, capsys):
    X_epigenomic, _, _ = enhancers_data
    build_mmnn(X_epigenomic, window_size)["model"].summary()

    captured = capsys.readouterr()
    assert "mmnn" in captured.out


def test_build_mmnn_pretrained(enhancers_data, window_size, capsys):
    X_epigenomic, _, _ = enhancers_data
    models = {
        "ffnn": build_ffnn(X_epigenomic),
        "cnn1d": build_cnn1d(window_size)
    }
    build_mmnn(mmnn_models=models, pretrained=True)["model"].summary()

    captured = capsys.readouterr()
    assert "mmnn" in captured.out


def test_build_cnn1d_hp_support(window_size, capsys):
    build_cnn1d_hp_support(window_size)["model"].summary()

    captured = capsys.readouterr()
    assert "cnn1d" in captured.out


def test_build_ffnn_hp(random_dataset):
    X_train, X_valid, _, y_train, y_valid, _ = random_dataset

    model_builder = lambda hp: build_ffnn_hp(X_train, hp)

    tuner = Hyperband(
        model_builder,
        objective=kt.Objective("val_AUPRC", direction="max"),
        max_epochs=1,
        factor=3,
        directory='tuner',
        project_name='ffnn_test',
        overwrite=True
    )

    tuner.search(
        X_train, y_train,
        validation_data=(X_valid, y_valid),
        epochs=1,
        batch_size=256
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    assert best_hp["n_neurons0"] > 0


def test_build_cnn1d_hp(enhancers_data, window_size, genome):
    X_epigenomic, bed, y = enhancers_data
    sequence = get_cnn_sequence(
        genome, bed, y.values.ravel()
    )

    model_builder = lambda hp: build_cnn1d_hp(window_size, hp)

    tuner = Hyperband(
        model_builder,
        objective=kt.Objective("AUPRC", direction="max"),
        max_epochs=1,
        factor=3,
        directory='tuner',
        project_name='cnn1d_test',
        overwrite=True
    )

    tuner.search(
        sequence,
        epochs=1,
        batch_size=256
    )

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    assert best_hp["n_neurons0"] > 0
