import numpy as np
import pandas as pd
from cache_decorator import Cache
from keras_mixed_sequence import MixedSequence
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    accuracy_score, average_precision_score, roc_auc_score
)
from sklearn.model_selection import (
    StratifiedShuffleSplit
)
from tensorflow.keras import Model
from tensorflow.keras.callbacks import EarlyStopping
from tqdm.keras import TqdmCallback
from tqdm.notebook import tqdm_notebook
from typing import Dict, List, Tuple
from ucsc_genomes_downloader import Genome

from .data_processing import (
    execute_boruta_feature_selection,
    execute_preprocessing
)


@Cache(
    cache_path=[
        "cache/{function_name}/{region}/{model_name}/"
        + "history_{_hash}.csv.xz",
        "cache/{function_name}/{region}/{model_name}/"
        + "evaluations_{_hash}.csv.xz",
    ],
    args_to_ignore=[
        "model", "train_sequence", "valid_sequence", "test_sequence"
    ]
)
def evaluate_model(
        model: Model,
        model_name: str,
        region: str,
        train_sequence: MixedSequence,
        valid_sequence: MixedSequence,
        test_sequence: MixedSequence,
        holdout_number: int,
        use_feature_selection: bool,
        use_validation_set: bool,
        window_size: int,
        resampling_strategy: str = None,
        patience: int = 3,
        epochs: int = 1000,
        batch_size: int = 256,
        class_weight: Dict[int, float] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Train and evaluate a given model.
    
    Parameters
    ----------
    model : Model
        Model to be trained and evaluated.
    
    model_name : str
        Name of the model.

    region : str
        The kind of regulatory region considered (enhancers or
        promoters).

    train_sequence : MixedSequence
        Training set that will be used to train the model.

    valid_sequence : MixedSequence
        Validation set that will be used during training.

    test_sequence : MixedSequence
        Test set that will be used for the final evaluation
        of the model.

    holdout_number : int
        Number of the current holdout iteration.

    use_feature_selection : bool
        Whether a feature selection algorithm has been applied
        to the data.

    use_validation_set : bool
        Whether the validation set has been extracted from the
        training set (True) or the test set is being used also
        as validation set (False).

    window_size : int
        Size of the window used to sample the data. The parameter 
        is needed to allow the cache decorator to store different 
        values when using different window sizes.

    resampling_strategy : str or None
        The kind of resampling strategy (e.g. random 
        under-sampling, SMOTE over-sampling, etc.), if any, 
        applied to the data.

    patience : int
        Number of epochs to be used for early stopping.

    epochs : int
        Maximum number of epochs to be used for training.

    batch_size : int
        Size of the batches to be fed to the model.

    class_weight : dict of str -> float or None
        Optional dictionary mapping class indices (integers) 
        to a weight (float) value, used for weighting the loss 
        function during training only. May be beneficial for
        imbalanced datasets.

    Raises
    ------
    ValueError
        If the region string is neither "enhancers" or "promoters".

    Returns
    ------- 
    A dictionary containing the training history and 
    a DataFrame containing the evaluation scores of the model 
    on both the training set and test set.
    """
    supported_regions = ["enhancers", "promoters"]

    if region not in supported_regions:
        raise ValueError(
            f"Parameter 'region' should be one "
            f"of {supported_regions}."
            f"Got '{region}' instead."
        )

    history = pd.DataFrame(model.fit(
        train_sequence,
        validation_data=valid_sequence,
        epochs=epochs,
        batch_size=batch_size,
        verbose=False,
        class_weight=class_weight,
        callbacks=[
            EarlyStopping("val_loss", patience=patience),
            TqdmCallback(verbose=1, tqdm_class=tqdm_notebook)
        ]
    ).history)

    train_evaluation = dict(
        zip(
            model.metrics_names,
            model.evaluate(
                train_sequence,
                verbose=False,
                batch_size=batch_size
            )
        )
    )
    test_evaluation = dict(
        zip(
            model.metrics_names,
            model.evaluate(
                test_sequence,
                verbose=False,
                batch_size=batch_size
            )
        )
    )
    train_evaluation["run_type"] = "train"
    test_evaluation["run_type"] = "test"
    for evaluation in (train_evaluation, test_evaluation):
        evaluation["model_name"] = model_name
        evaluation["region"] = region
        evaluation["holdout_number"] = holdout_number
        evaluation["use_feature_selection"] = use_feature_selection
        evaluation["use_validation_set"] = use_validation_set
        evaluation["patience"] = patience
        evaluation["max_epochs"] = epochs
        evaluation["resampling_strategy"] = resampling_strategy
        evaluation["class_weights"] = class_weight is not None
        evaluation["window_size"] = window_size

    evaluations = pd.DataFrame([
        train_evaluation,
        test_evaluation
    ])

    return history, evaluations


def repeated_holdout_evaluation(
        X_epi: pd.DataFrame,
        bed: pd.DataFrame,
        y: pd.DataFrame,
        genome: Genome,
        models: Dict[str, Tuple],
        region: str,
        window_size: int,
        n_splits: int = 10,
        test_size: float = 0.2,
        random_state: int = 42,
        use_validation_set: bool = True,
        resampling_strategy: str = None,
        no_feature_selection: bool = False
) -> Tuple[List, List, List, List]:
    """Evaluate models according to a repeated holdout strategy.

    Parameters
    ----------
    X_epi : pd.DataFrame
        DataFrame containing the epigenomic data.

    bed : pd.DataFrame
        BED representation of the data that will be used
        by the Genome object to retrieve the sequence data.

    y : pd.DataFrame
        DataFrame of binary class labels annotated as 0 and 1.

    genome : Genome
        Genome object to retrieve the sequence data.

    models : dict of str -> Tuple
        Dictionary having the name of models as keys and as values
        tuples containing: 
            1) the corresponding models;
            2) a tuple containing a function to generate data for
            the models and optional parameters to use during 
            training of the models.

    region : str
        Type of regulatory region under consideration ("enhancers" or
        "promoters").

    window_size : int
        Size of the window used to sample the data.

    n_splits : n
        Number of splits for the repeated holdout.

    test_size : float
        Float number representing the ratio of instances from the
        dataset to be used as test set (and validation set).

    random_state : int
        Number used to initialize the holdout generator random seed.

    use_validation_set : bool
        Whether the validation set has been extracted from the
        training set (True) or the test set is being used also
        as validation set (False).

    resampling_strategy : str or None
        The kind of resampling strategy (e.g. random 
        under-sampling, SMOTE over-sampling, etc.), if any, 
        applied to the data.

    no_feature_selection : bool
        Skip feature selection step.

    Raises
    ------
    ValueError
        If the region string is neither "enhancers" or "promoters".

    Returns
    -------
    Tuple containing the aggregated training histories of the models,
    their performances on the training and test sets, the list of 
    features kept by the feature selection algorithm and the list
    of features discarded by the same feature selection algorithm.
    """
    supported_regions = ["enhancers", "promoters"]

    if region not in supported_regions:
        raise ValueError(
            f"Parameter 'region' should be one "
            f"of {supported_regions}."
            f"Got '{region}' instead."
        )

    all_histories = []
    all_performances = []
    all_kept_features = []
    all_discarded_features = []

    holdouts_generator = StratifiedShuffleSplit(
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state
    )

    for holdout_n, (train_full_idx, test_idx) in tqdm_notebook(
            enumerate(holdouts_generator.split(X_epi, y)),
            total=n_splits,
            desc="Computing holdouts",
            leave=False
    ):
        # In order to extract a validation set from the training set, 
        # we consider the first set of indices as "train_full_idx". 
        # The actual training set indices will be "train_idx" and 
        # the validation set indices will be "valid_idx".
        X_epi_train_full = X_epi.iloc[train_full_idx]
        y_train_full = y.iloc[train_full_idx]

        # The test set is considered "unfiltered" ("unf") because we 
        # have yet to perform feature selection. The same notation 
        # will be used for the actual training set and validation 
        # set below.
        unf_X_epi_test = X_epi.iloc[test_idx]
        y_test = y.iloc[test_idx].values.ravel()

        if use_validation_set:
            validation_data_generator = StratifiedShuffleSplit(
                n_splits=1,
                test_size=test_size,
                random_state=random_state
            ).split(X_epi_train_full, y_train_full)

            train_idx, valid_idx = next(validation_data_generator)

            unf_X_epi_train = X_epi_train_full.iloc[train_idx]
            y_train = y_train_full.iloc[train_idx].values.ravel()
            unf_X_epi_valid = X_epi_train_full.iloc[valid_idx]
            y_valid = y_train_full.iloc[valid_idx].values.ravel()
        else:
            train_idx, valid_idx = train_full_idx, test_idx
            unf_X_epi_train, y_train = X_epi_train_full, y_train_full.values.ravel()
            unf_X_epi_valid, y_valid = unf_X_epi_test, y_test

        # Preprocessing the epigenomic data with imputation 
        # and normalization.
        unf_X_epi_train, unf_X_epi_valid, unf_X_epi_test = \
            execute_preprocessing(
                unf_X_epi_train, unf_X_epi_valid, unf_X_epi_test
            )

        for use_feature_selection in tqdm_notebook(
                (True, False),
                desc="Running feature selection",
                leave=False
        ):
            if use_feature_selection & no_feature_selection:
                continue

            if use_feature_selection:
                kept_features, discarded_features = \
                    execute_boruta_feature_selection(
                        X_train=unf_X_epi_train,
                        y_train=y_train,
                        holdout_number=holdout_n,
                        use_validation_set=use_validation_set,
                        window_size=window_size,
                        region=region
                    )
                all_kept_features.append(kept_features)
                all_discarded_features.append(discarded_features)

                X_epi_train = unf_X_epi_train[kept_features].values
                X_epi_valid = unf_X_epi_valid[kept_features].values
                X_epi_test = unf_X_epi_test[kept_features].values
            else:
                X_epi_train = unf_X_epi_train.values
                X_epi_valid = unf_X_epi_valid.values
                X_epi_test = unf_X_epi_test.values

            for (
                    model_name,
                    (build_model, get_model_train_sequence, get_model_test_sequence, params)
            ) in tqdm_notebook(
                models.items(),
                desc="Training models",
                leave=False
            ):
                kwargs = {
                    "holdout_number": holdout_n,
                    "genome": genome,
                    "bed_train": bed.iloc[train_idx],
                    "bed_valid": bed.iloc[valid_idx],
                    "X_train": X_epi_train,
                    "X_valid": X_epi_valid,
                    "y_train": y_train,
                    "y_valid": y_valid,
                    "region": region
                }
                model_dict = build_model(X_epi_train, window_size, kwargs)
                model = model_dict["model"]

                if issubclass(type(model), BaseEstimator):
                    history, performance = evaluate_sklearn_model(
                        model=model,
                        model_name=model_name,
                        train_sequence=get_model_train_sequence(
                            genome,
                            bed.iloc[train_idx],
                            X_epi_train,
                            y_train
                        ),
                        valid_sequence=get_model_test_sequence(
                            genome,
                            bed.iloc[valid_idx],
                            X_epi_valid,
                            y_valid
                        ),
                        test_sequence=get_model_test_sequence(
                            genome,
                            bed.iloc[test_idx],
                            X_epi_test,
                            y_test
                        ),
                        holdout_number=holdout_n,
                        use_feature_selection=use_feature_selection,
                        use_validation_set=use_validation_set,
                        window_size=window_size,
                        resampling_strategy=params[region]["resampling"] \
                            if "resampling" in params[region] else resampling_strategy,
                        region=region,
                        **params[region]
                    )
                else:
                    history, performance = evaluate_model(
                        model=model,
                        model_name=model_name,
                        train_sequence=get_model_train_sequence(
                            genome,
                            bed.iloc[train_idx],
                            X_epi_train,
                            y_train
                        ),
                        valid_sequence=get_model_test_sequence(
                            genome,
                            bed.iloc[valid_idx],
                            X_epi_valid,
                            y_valid
                        ),
                        test_sequence=get_model_test_sequence(
                            genome,
                            bed.iloc[test_idx],
                            X_epi_test,
                            y_test
                        ),
                        holdout_number=holdout_n,
                        use_feature_selection=use_feature_selection,
                        use_validation_set=use_validation_set,
                        window_size=window_size,
                        resampling_strategy=params[region]["resampling"] \
                            if "resampling" in params[region] else resampling_strategy,
                        region=region,
                        **params[region]
                    )

                all_performances.append(performance)
                all_histories.append(history)

    return all_histories, all_performances, \
           all_kept_features, all_discarded_features


@Cache(
    cache_path=[
        "cache/{function_name}/{region}/{model_name}/"
        + "history_{_hash}.csv.xz",
        "cache/{function_name}/{region}/{model_name}/"
        + "evaluations_{_hash}.csv.xz",
    ],
    args_to_ignore=[
        "model", "train_sequence", "valid_sequence", "test_sequence"
    ]
)
def evaluate_sklearn_model(
    model: BaseEstimator,
    model_name: str,
    region: str,
    train_sequence: Dict[str, np.ndarray],
    valid_sequence: Dict[str, np.ndarray],
    test_sequence: Dict[str, np.ndarray],
    holdout_number: int,
    use_feature_selection: bool,
    use_validation_set: bool,
    window_size: int,
    resampling_strategy: str = None,
    batch_size: int = 256,
):
    """
    Train and evaluate a given scikit-learn estimator model.

    Parameters
    ----------
    model : BaseEstimator
        Model to be trained and evaluated.

    model_name : str
        Name of the model.

    region : str
        The kind of regulatory region considered (enhancers or
        promoters).

    train_sequence : dict of str -> np.ndarray
        Dictionary containing data (key "X") and labels
        (key "y") of the training set.

    valid_sequence : dict of str -> np.ndarray
        Dictionary containing data (key "X") and labels
        (key "y") of the validation set.

    test_sequence : dict of str -> np.ndarray
        Dictionary containing data (key "X") and labels
        (key "y") of the test set.

    holdout_number : int
        Number of the current holdout iteration.

    use_feature_selection : bool
        Whether a feature selection algorithm has been applied
        to the data.

    use_validation_set : bool
        Whether the validation set has been extracted from the
        training set (True) or the test set is being used also
        as validation set (False).

    window_size : int
        Size of the window used to sample the data. The parameter
        is needed to allow the cache decorator to store different
        values when using different window sizes.

    resampling_strategy : str or None
        The kind of resampling strategy (e.g. random
        under-sampling, SMOTE over-sampling, etc.), if any,
        applied to the data.

    batch_size : int
        Size of the batches to be fed to the model.

    Returns
    -------
    An empty dictionary, for compatibility purposes, and
    a DataFrame containing the evaluation scores of the model
    on both the training set and test set.
    """
    model.fit(train_sequence["X"], train_sequence["y"])
    y_train_pred = model.predict(train_sequence["X"])
    y_test_pred = model.predict(test_sequence["X"])
    y_train_true = train_sequence["y"]
    y_test_true = test_sequence["y"]
    # print("finished training")

    train_evaluation, test_evaluation = dict(), dict()

    for metric_name, metric_func in (
        ("accuracy", accuracy_score),
        ("AUROC", roc_auc_score),
        ("AUPRC", average_precision_score)
    ):
        train_evaluation[metric_name] = metric_func(y_train_true, y_train_pred)
        test_evaluation[metric_name] = metric_func(y_test_true, y_test_pred)

    train_evaluation["run_type"] = "train"
    test_evaluation["run_type"] = "test"
    for evaluation in (train_evaluation, test_evaluation):
        evaluation["model_name"] = model_name
        evaluation["region"] = region
        evaluation["holdout_number"] = holdout_number
        evaluation["use_feature_selection"] = use_feature_selection
        evaluation["use_validation_set"] = use_validation_set
        evaluation["resampling_strategy"] = resampling_strategy
        evaluation["window_size"] = window_size

    evaluations = pd.DataFrame([
        train_evaluation,
        test_evaluation
    ])
    history = pd.DataFrame({"hist": [0], "test": ["test"]})

    return history, evaluations
