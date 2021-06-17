import numpy as np
import pandas as pd
from boruta import BorutaPy
from cache_decorator import Cache
from keras_bed_sequence import BedSequence
from keras_mixed_sequence import MixedSequence, VectorSequence
from multiprocessing import cpu_count
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from tensorflow.keras import initializers
from typing import Callable, Dict, List, Tuple
from ucsc_genomes_downloader import Genome

from bioproject.resampling.resampled_sequence import (
    ResampledBedSequence
)
from bioproject.resampling.resampling_utils import resample_data


def flat_one_hot_encode(
    genome: Genome, 
    bed: pd.DataFrame,
    window_size: int, 
    nucleotides: str = "actg"
) -> pd.DataFrame:
    """Return one-hot encoded nucleotides flat sequence.

    Parameters
    ----------
    genome : Genome
        Genome object to retrieve the sequence data.

    bed : pd.DataFrame
        BED representation of the data that will be used
        by the Genome object to retrieve the sequence data.

    window_size : int
        Size of the window used to sample the data.

    nucleotides : str = "actg"
        Nucleotides to consider when performing one-hot encoding.        

    Returns
    -------
    DataFrame containing the flattened numpy array of the one-hot 
    encoded nucleotides sequence.
    """
    flattened_array = np.array(
        BedSequence(
            genome=genome,
            bed=bed,
            nucleotides=nucleotides,
            batch_size=1
        )
    ).reshape(-1, window_size*4).astype(int)

    return pd.DataFrame(
        flattened_array,
        columns = [
            f"{i}{nucleotide}"
            for i in range(window_size)
            for nucleotide in nucleotides
        ]
    )


def fill_data_dictionaries(
    task: Callable, 
    region: str,
    genome: Genome, 
    beds: Dict, 
    epigenomes: Dict, 
    labels: Dict,
    sequences: Dict,
    cell_line: str,
    window_size: int,
    nucleotides: str = "actg"
) -> Tuple[Dict, Dict, Dict, Dict]:
    """Dictionaries filled with epigenomic and sequence data for
    the given task.

    Parameters
    ----------
    task : Callable
        Function to be called to get data for a given task (e.g.
        "active enhancers vs inactive enhancers").

    region : str
        Type of regulatory region under consideration ("enhancers" or
        "promoters").

    genome : Genome
        Genome object to retrieve the sequence data.

    beds : dict of str -> pd.DataFrame
        Dictionary mapping a region to the corresponding BED data,
        to be filled by the present function.

    epigenomes : dict of str -> pd.DataFrame
        Dictionary mapping a region to the corresponding epigenomic 
        data, to be filled by the present function.

    labels : dict of str -> pd.DataFrame
        Dictionary mapping a region to the corresponding class label 
        data, to be filled by the present function.

    sequences : dict of str -> pd.DataFrame
        Dictionary mapping a region to the corresponding sequence 
        data, to be filled by the present function.

    cell_line : str
        Name of the cell line under consideration.

    window_size : int
        Size of the window used to sample the data.

    nucleotides : str = "actg"
        Nucleotides to consider when performing one-hot encoding.

    Returns
    -------
    Four dictionaries, respectively containing the BED, epigenomic,
    class labels and sequence data for the given task.
    """
    X, y = task(
        cell_line=cell_line,
        window_size=window_size,
        binarize=True
    )
    y = pd.DataFrame(y.astype(int).values)
    bed = X.reset_index().rename_axis(None, axis=1)
    X_epigenomic = bed[bed.columns[4:]]

    beds[region] = bed
    epigenomes[region] = X_epigenomic
    labels[region] = y
    
    sequences[region] = flat_one_hot_encode(
        genome=genome,
        bed=bed,
        window_size=window_size,
        nucleotides=nucleotides
    )
    
    return beds, epigenomes, labels, sequences


def get_pos_neg(
    y: pd.DataFrame
) -> Tuple[int, int]:
    """Return the number of positive and negative samples.

    Parameters
    ----------
    y : pd.DataFrame
        DataFrame of binary class labels annotated as 0 and 1.

    Returns
    ------- 
    Tuple providing the number of positive and negative samples 
    in the binary class labels array. 
    """
    pos = sum(y.values)
    neg = len(y) - pos
    return pos, neg


def get_initial_output_bias(
    y: pd.DataFrame
) -> initializers.Constant:
    """Return bias initializer for imbalanced data.

    Parameters
    ----------
    y : pd.DataFrame
        DataFrame of binary class labels annotated as 0 and 1.
 
    Returns
    -------
    Bias initializer to be used in the output layer of a neural
    network that handles imbalanced data. The bias initial value
    is computed according to the formula provided in:
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#optional_set_the_correct_initial_bias
    """
    pos, neg = get_pos_neg(y)
    return initializers.Constant(np.log([pos/neg]))


def get_class_weights(
    y: pd.DataFrame
) -> Dict[int, float]:
    """Return class weight for loss function.

    Parameters
    ----------
    y : pd.DataFrame
        DataFrame of binary class labels annotated as 0 and 1.

    Returns
    -------
    Dictionary mapping class indices (integers) to a weight (float) 
    value, used for weighting the loss function during training 
    only. May be beneficial for imbalanced datasets. The weights 
    are computed according to the formula provided in:
    https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
    """
    pos, neg = get_pos_neg(y)
    total = len(y)
    weight_for_0 = (1 / neg) * (total / 2.0)
    weight_for_1 = (1 / pos) * (total / 2.0)

    return {0: weight_for_0, 1: weight_for_1}


def missing_values_report(
    epigenomes: Dict[str, pd.DataFrame],
    beds: Dict[str, pd.DataFrame],
    genome: Genome
) -> None:
    """Print a report on missing values in the dataset.

    Parameters
    ----------
    epigenomes : dict of str -> pd.DataFrame
        Dictionary mapping each region to the corresponding
        dataset containing epigenomic data.

    beds : dict of str -> pd.DataFrame
        Dictionary mapping each region to the corresponding 
        BED representation of the data that will be used
        by the Genome object to retrieve the sequence data.

    genome : Genome
        Genome object to retrieve the sequence data.

    Returns
    -------
    The function prints the report without returning anything.
    """
    # Epigenomic data analysis
    for region, epigenomic_data in epigenomes.items():

        total_nans = epigenomic_data.isna().values.sum()
        max_row_index = epigenomic_data.isna().sum(axis=1).argmax()
        max_row_sum = epigenomic_data.iloc[max_row_index].isna().sum()

        max_column_number = epigenomic_data.isna().sum(axis=0).argmax()
        max_column_name = epigenomic_data.columns[max_column_number]
        max_column_sum = sum(
            epigenomic_data.loc[:, max_column_name].isna()
        )

        title = f"{region.upper()} EPIGENOMIC DATA"

        print("\n".join((
                "=" * len(title),
                title,
                "=" * len(title),
                f"The dataset contains {total_nans} NaN values " 
                + f"out of {epigenomic_data.values.size} total values.",
                f"Sample (row) number {max_row_index} has the most "
                + f"NaN values, amounting to {max_row_sum} NaN values "
                + f"out of {epigenomic_data.shape[1]} row values.",
                f"Feature (column) number {max_column_number} "
                + f"({max_column_name}) has the most NaN values, "
                + f"amounting to {max_column_sum} NaN values out of "
                + f"{epigenomic_data.shape[0]} column values.\n"
            ))
        )

    # Sequence data analysis
    for region, bed in beds.items():

        total_unk = sum([
            nucleotide.lower() == "n"
            for sequence in genome.bed_to_sequence(bed)
            for nucleotide in sequence
        ])

        title = f"{region.upper()} SEQUENCE DATA"

        print("\n".join((
                "=" * len(title),
                title,
                "=" * len(title),
                f"The dataset contains {total_unk} unidentified "
                + f"nucleotides."
            ))
        )


def execute_preprocessing(
    X_train: pd.DataFrame,
    X_valid: pd.DataFrame,
    X_test: pd.DataFrame,
    imputer_type: str = "knn_imputer",
    scaler_type: str = "robust_scaler"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform imputation and scaling on the data.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training set that will be used to fit the imputer and the
        scaler and then will be transformed by said imputer and
        scaler.

    X_valid : pd.DataFrame
        Validation set that will be transformed by the imputer
        and scaler.

    X_test : pd.DataFrame
        Test set that will be transformed by the imputer and 
        scaler.

    imputer_type : str or None
        Name of the imputer that will be used, if any.
        The only supported imputer type is "knn_imputer". 

    scaler_type : str or None
        Name of the scaler that will be used, if any.
        The only supported scaler types are "minmax_scaler"
        and "robust_scaler".

    Raises
    ------
    ValueError
        If the imputer type is not one of the supported imputer
        types.

        If the scaler type is not one of the supported scaler
        types.

    Returns
    -------
    Training, validation and test sets transformed according to the
    specified imputer and scaler, if any. Both the imputer and the
    scaler are fitted only on the training set.
    """
    supported_imputer_types = ["knn_imputer"]
    supported_scaler_types = ["minmax_scaler", "robust_scaler"]

    if imputer_type is not None:
        if imputer_type == "knn_imputer":
            imputer = KNNImputer()
        else:
            raise ValueError(
                    f"When 'imputer_type' is a string, it "
                    f"should be one of {supported_imputer_types}."
                    f"Got '{imputer_type}' instead."
                )
        imputer.fit(X_train)
        X_train = pd.DataFrame(
            imputer.transform(X_train),
            columns=X_train.columns
        )
        X_valid = pd.DataFrame(
            imputer.transform(X_valid),
            columns=X_valid.columns
        )
        X_test = pd.DataFrame(
            imputer.transform(X_test),
            columns=X_test.columns
        )
        
    if scaler_type is not None:
        if scaler_type == "minmax_scaler":
            scaler = MinMaxScaler()
        elif scaler_type == "robust_scaler":
            scaler = RobustScaler()
        else:
            raise ValueError(
                    f"When 'scaler_type' is a string, it "
                    f"should be one of {supported_scaler_types}."
                    f"Got '{scaler_type}' instead."
                )
        
        scaler.fit(X_train)
        X_train = pd.DataFrame(
            scaler.transform(X_train),
            columns=X_train.columns
        )
        X_valid = pd.DataFrame(
            scaler.transform(X_valid),
            columns=X_valid.columns
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns
        )
        
    return X_train, X_valid, X_test


@Cache(
    cache_path=[
        "cache/{function_name}/kept_features_{_hash}.json",
        "cache/{function_name}/discarded_features_{_hash}.json"
    ],
    args_to_ignore=[
        "X_train", "y_train"
    ]
)
def execute_boruta_feature_selection(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    holdout_number: int,
    use_validation_set: bool,
    window_size: int,
    region: str,
    imputer_type: str = "knn_imputer",
    scaler_type: str = "robust_scaler",
    max_iter: int = 100
) -> Tuple[List[str], List[str]]:
    """Returns tuple with lists of kept and discarded features.
    
    Parameters
    ----------
    X_train : pd.DataFrame
        The data reserved for the input of the training 
        of the Boruta model.

    y_train : np.ndarray
        The data reserved for the output of the training 
        of the Boruta model.

    holdout_number : int
        The number of the current holdout.

    use_validation_set : bool
        Whether the validation set has been extracted from the
        training set (True) or the test set is being used also
        as validation set (False). The parameter is needed to
        allow the cache decorator to store different features
        in case the validation set is also the test set.

    window_size : int
        Size of the window used to sample the data. The parameter 
        is needed to allow the cache decorator to store different 
        features when using different window sizes.

    region : str
        The kind of regulatory region considered (enhancers or
        promoters).

    imputer_type : str or None
        Name of the imputer that will be used, if any.
        Similarly to the use_validation_set parameter,
        imputer_type is needed for the cache decorator to
        work properly when different imputers are used. 

    scaler_type : str or None
        Name of the scaler that will be used, if any.
        Similarly to the use_validation_set parameter,
        scaler_type is needed for the cache decorator to
        work properly when different scalers are used.

    max_iter : int
        Number of iterations to run Boruta for.

    Returns
    -------
    A tuple a of lists. The first list contains the names of the 
    columns that the Boruta algorithm identified as features to
    be kept. The second list contains the names of the columns
    that should be discarded. 
    """
    
    boruta_selector = BorutaPy(
        RandomForestClassifier(
            n_jobs=cpu_count(), 
            class_weight='balanced_subsample', 
            max_depth=5
        ),
        n_estimators='auto',
        verbose=False,
        alpha=0.05,  # p_value
        max_iter=max_iter,
        random_state=42
    )

    boruta_selector.fit(X_train.values, y_train)
    
    kept_features = list(
        X_train.columns[boruta_selector.support_]
    )
    discarded_features = list(
        X_train.columns[~boruta_selector.support_]
    )
    
    return kept_features, discarded_features


def get_ffnn_sequence(
    X: np.ndarray, 
    y: np.ndarray,
    resampling_strategy: str = None,
    batch_size: int = 256
) -> MixedSequence:
    """Return an iterable MixedSequence providing batch-sized 
    data suitable for a Feed-Forward Neural Network working
    on epigenomic data.

    Parameters
    ----------
    X : np.ndarray
        Numpy array of epigenomic data.

    y : np.ndarray
        Numpy array of binary class labels annotated as 0 and 1.

    resampling_strategy : str or None
        String representing the desired resampling strategy, if any.
        The supported strategies are "over_sample", performed
        through the SMOTE algorithm, and "under_sample",
        performed through the random under-sample algorithm.

    batch_size : int
        Size of the batches to be fed to the neural network.

    Returns
    -------
    A MixedSequence object ready to be passed as input to a
    Feed-Forward Neural Network working on epigenomic data.
    """
    if resampling_strategy is not None:
        X, y = resample_data(X, y, resampling_strategy)

    return MixedSequence(
        x={
            "input_epigenomic_data": VectorSequence(
                X,
                batch_size=batch_size
            )
        },
        y=VectorSequence(
            y,
            batch_size=batch_size
        )
    )


def get_cnn_sequence(
    genome: Genome, 
    bed: pd.DataFrame,
    y: np.ndarray,
    resampling_strategy: str = None,
    batch_size: int = 256
) -> MixedSequence:
    """Return an iterable MixedSequence providing batch-sized 
    data suitable for a Convolutional Neural Network working
    on sequence data.

    Parameters
    ----------
    genome : Genome
        Genome object needed to retrieve the nucleotide sequences
        from the DataFrame representing a BED object.

    bed : pd.DataFrame
        BED representation of data.

    y : np.ndarray
        Numpy array of binary class labels annotated as 0 and 1.

    resampling_strategy : str or None
        String representing the desired resampling strategy, if any.
        The supported strategies are "over_sample", performed
        through the SMOTE algorithm, and "under_sample",
        performed through the random under-sample algorithm.

    batch_size : int
        Size of the batches to be fed to the neural network.

    Returns
    -------
    A MixedSequence object ready to be passed as input to a
    Convolutional Neural Network working on sequence data.
    """
    if resampling_strategy is not None:
        X, y = resample_data(bed, y, resampling_strategy, genome)
        return MixedSequence(
            x={
                "input_sequence_data": ResampledBedSequence(
                    X,
                    batch_size=batch_size
                )
            },
            y=VectorSequence(
                y,
                batch_size=batch_size
            )
        )
    else:
        return MixedSequence(
            x={
                "input_sequence_data": BedSequence(
                    genome,
                    bed,
                    batch_size=batch_size
                )
            },
            y=VectorSequence(
                y,
                batch_size=batch_size
            )
        )


def get_mmnn_sequence(
    genome: Genome, 
    bed: pd.DataFrame, 
    X: np.ndarray, 
    y: np.ndarray,
    resampling_strategy: str = None,
    batch_size=256
) -> MixedSequence:
    """Return an iterable MixedSequence providing batch-sized 
    data suitable for a Multi-Modal Neural Network working
    on both sequence data and epigenomic data.

    Parameters
    ----------
    genome : Genome
        Genome object needed to retrieve the nucleotide sequences
        from the DataFrame representing a BED object.

    bed : pd.DataFrame
        BED representation of data.

    X : np.ndarray
        Numpy array of epigenomic data.

    y : np.ndarray
        Numpy array of binary class labels annotated as 0 and 1.

    resampling_strategy : str or None
        String representing the desired resampling strategy, if any.
        The supported strategies are "over_sample", performed
        through the SMOTE algorithm, and "under_sample",
        performed through the random under-sample algorithm.

    batch_size : int
        Size of the batches to be fed to the neural network.

    Returns
    -------
    A MixedSequence object ready to be passed as input to a
    Multi-Modal Neural Network working on both sequence data
    and epigenomic data.
    """
    if resampling_strategy is not None:
        # y should be the same for both types
        # of data. Need a way to enforce that.
        raise ValueError(
            "Resampling not yet supported for MMNN"
        )
        """
        X_seq, y = resample_data(
            bed, y, resampling_strategy, genome
        )
        
        X_epi, _ = resample_data(
            X, y, resampling_strategy
        )
        return MixedSequence(
            x={
                "input_sequence_data": ResampledBedSequence(
                    X_seq,
                    batch_size=batch_size
                ),
                "input_epigenomic_data": VectorSequence(
                    X_epi,
                    batch_size
                )
            },
            y=VectorSequence(
                y,
                batch_size=batch_size
            )
        )
        """
    else:
        return MixedSequence(
            x={
                "input_sequence_data": BedSequence(
                    genome,
                    bed,
                    batch_size=batch_size
                ),
                "input_epigenomic_data": VectorSequence(
                    X,
                    batch_size
                )
            },
            y=VectorSequence(
                y,
                batch_size=batch_size
            )
        )
