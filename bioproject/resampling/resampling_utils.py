import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from keras_bed_sequence.utils import nucleotides_to_numbers
from typing import Tuple, Union
from ucsc_genomes_downloader import Genome


def resample_data(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
    resampling_strategy: str,
    genome: Genome = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Return data resampled according to a resampling strategy.

    Parameters
    ----------
    X : np.ndarray or pd.DataFrame
        Data values to be resampled. This should corresponds to
        epigenomic data (np.ndarray) when genome is None and to
        a bed sequence (pd.DataFrame) when genome is not None.

    y : np.ndarray
        Numpy array of binary class labels annotated as 0 and 1.

    resampling_strategy : str
        String representing the desired resampling strategy.
        The supported strategies are "over_sample", performed
        through the SMOTE algorithm, and "under_sample",
        performed through the random under-sample algorithm.

    genome : Genome or None
        Genome object needed to retrieve the sequences of
        nucleotides when dealing with a bed sequence.

    Raises
    ------
    ValueError
        If the resampling strategy is not one of the supported 
        resampling strategies.

    Returns
    -------
    Tuple containing the resampled X and y.
    """
    supported_resampling_strategies = [
        "over_sample", "under_sample"
    ]

    if resampling_strategy == "over_sample":
        resampler = SMOTE()
    elif resampling_strategy == "under_sample":
        resampler = RandomUnderSampler()
    else:
        raise ValueError(
            f"Parameter 'resampling_strategy' should be one "
            f"of {supported_resampling_strategies}."
            f"Got '{resampling_strategy}' instead."
        )

    # When genome is provided, X is expected to be a bed sequence
    # instead of epigenomic data.
    if genome is not None:
        dna_seq = np.array(genome.bed_to_sequence(X), dtype=str)
        X = nucleotides_to_numbers(
            nucleotides="actg",
            sequences=dna_seq
        )
    
    X, y = resampler.fit_resample(X, y)

    return X, y
