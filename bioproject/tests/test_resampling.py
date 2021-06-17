import numpy as np
import pytest
from sklearn.impute import KNNImputer

from bioproject.resampling.resampled_sequence import ResampledBedSequence
from bioproject.resampling.resampling_utils import resample_data


def test_resample_data(enhancers_data):
    X_epigenomic, bed, y = enhancers_data
    imputer = KNNImputer()
    X_epigenomic = imputer.fit_transform(X_epigenomic)
    _, y = resample_data(X_epigenomic, y.values.ravel(), "over_sample")

    # The number of samples in each class should now be equal.
    assert (sum(y) == (len(y) - sum(y)))


def test_resample_data_exception(enhancers_data):
    X_epigenomic, _, y = enhancers_data
    with pytest.raises(ValueError):
        resample_data(X_epigenomic, y.values.ravel(), "None")


def test_resampled_bed_sequence(enhancers_data, genome):
    X_epigenomic, bed, y = enhancers_data
    X, _ = resample_data(bed, y, "over_sample", genome)

    seq = ResampledBedSequence(X, batch_size=256)
    assert len(np.array(seq)[0]) == 256
