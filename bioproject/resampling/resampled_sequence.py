import numpy as np
from keras_mixed_sequence import VectorSequence
from keras_bed_sequence.utils import fast_to_categorical
from typing import Union


class ResampledBedSequence(VectorSequence):
    """Keras Sequence to lazily one-hot encode sequences from 
    a given bed file. The code of the class is the same as that of 
    the BedSequence class in:

    https://github.com/LucaCappelletti94/keras_bed_sequence

    The only change to the BedSequence class is that during the 
    object initialization the nucleotide sequences are not 
    retrieved with a Genome object. It is assumed that a vector,
    supposedly resampled from nucleotide sequences, is already 
    provided at the time of the object initialization. In this 
    way, it is now possible to handle resampled sequences.
    """

    def __init__(
        self,
        vector: np.ndarray,
        batch_size: int,
        window_length: int = 256,
        nucleotides: str = "actg",
        unknown_nucleotide_value: float = 0.25,
        random_state: int = 42,
        elapsed_epochs: int = 0,
        shuffle: bool = True
    ):
        """Return new ResampledBedSequence object.

        Parameters
        ----------
        vector: np.ndarray
            Numpy array with data to be split into batches.

        batch_size : int
            Batch size to be returned for each request. 

        nucleotides : str = "actg"
            Nucleotides to consider when one-hot encoding.

        unknown_nucleotide_value : float = 0.25
            The default value to use for encoding unknown 
            nucleotides.

        random_state : int = 42
            Starting random_state to use if shuffling the dataset.

        elapsed_epochs : int = 0
            Number of elapsed epochs to init state of generator.

        shuffle : bool = True
            Whether to shuffle or not the sequence.
        """
        self._window_length = window_length
        self._nucleotides = nucleotides
        self._nucleotides_number = len(nucleotides)
        self._unknown_nucleotide_value = unknown_nucleotide_value

        super().__init__(
            vector,
            batch_size,
            random_state=random_state,
            elapsed_epochs=elapsed_epochs,
            shuffle=shuffle
        )

    @property
    def window_length(self) -> int:
        """Return number of nucleotides in a window."""
        return self._window_length

    @property
    def nucleotides(self) -> str:
        """Return number of nucleotides considered."""
        return self._nucleotides

    @property
    def nucleotides_number(self) -> int:
        """Return number of nucleotides considered."""
        return self._nucleotides_number

    def __getitem__(self, idx: int) -> Union[np.ndarray, np.ndarray]:
        """Return batch corresponding to given index.
        Parameters
        ---------------
        idx: int,
            Index corresponding to batch to be rendered.
        Returns
        ---------------
        Return Tuple containing X and Y numpy arrays corresponding 
        to given batch index.
        """
        return fast_to_categorical(
            super().__getitem__(idx),
            num_classes=self.nucleotides_number,
            unknown_nucleotide_value=self._unknown_nucleotide_value
        )
