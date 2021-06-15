import numpy as np
from extra_keras_metrics import get_complete_binary_metrics
from kerastuner.engine.hyperparameters import HyperParameters
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.layers import (
    BatchNormalization, Concatenate, Conv1D, Conv2D, Dense,
    Dropout, Flatten, GlobalAveragePooling1D, Input, InputLayer,
    MaxPool1D, MaxPool2D, Multiply, ReLU, Reshape
)
from tensorflow.keras.models import Model, Sequential
from typing import Dict, Union

# ===========
# FFNN MODELS
# ===========
from tensorflow.python.layers.base import Layer


def build_ffnn_iwbbio2020(
    X_train: np.ndarray
) -> Dict[str, Sequential]:
    """Build the Feed-Forward Neural Network IWBBIO2020.
    This is the neural network described as Bayesian-FFNN in [1].
    The paper mentions a batch size of 100 and a maximum of 1000
    epochs to train the network.

    Parameters
    ----------
    X_train : np.ndarray
        Numpy array of input data representing the training set.

    Returns
    -------
    The compiled FFNN.

    References
    ----------
        [1] L. Cappelletti et al., "Bayesian Optimization Improves 
        Tissue-Specific Prediction of Active Regulatory Regions 
        with Deep Neural Networks", in I. Rojas et al. (Eds.), 
        Bioinformatics and Biomedical Engineering - 8th International 
        Work-Conference, IWBBIO 2020, Proceedings, pp. 600-612, 
        Springer, 2020. https://doi.org/10.1007/978-3-030-45385-5_54
    """
    model = Sequential(
        layers=[
            InputLayer((X_train.shape[1],)),
            Dense(
                256,
                activation="relu",
                kernel_regularizer=regularizers.L2(0.001)
            ),
            Dense(
                128,
                activation="relu",
                kernel_regularizer=regularizers.L2(0.001)
            ),
            Dense(
                64,
                activation="relu",
                kernel_regularizer=regularizers.L2(0.001)
            ),
            Dense(1, activation="sigmoid")
        ],
        name="ffnn_IWBBIO2020"
    )

    initial_learning_rate = 0.1
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.01
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.SGD(learning_rate=lr_schedule),
        metrics=get_complete_binary_metrics()
    )

    return {"model": model}


def build_ffnn(
    X_train: np.ndarray,
    num_layers: int = 4,
    n_neurons0: int = 224,
    n_neurons1: int = 32,
    l2_reg: float = 0.005,
    learning_rate: float = 0.0001
) -> Dict[str, Union[Model, Layer]]:
    """Build a custom Feed-Forward Neural Network.

    Parameters
    ----------
    X_train : np.ndarray
        Numpy array of input data representing the training set.

    num_layers : int
        Number of hidden layers in the network.

    n_neurons0 : int
        Number of neurons in hidden layers number 0 and 1, 
        if present.

    n_neurons1 : int
        Number of neurons in hidden layers number 2 and 3, 
        if present.

    l2_reg : float
        Amount of L2 regularization in all the hidden layers.

    learning_rate : float
        Learning rate of the Nadam optimizer.

    Returns
    -------
    The compiled FFNN.
    """
    input_epigenomic_data = Input(
        (X_train.shape[1], ), name="input_epigenomic_data"
    )
    last_hidden_ffnn = hidden = input_epigenomic_data

    for layer in range(num_layers):
        if layer == (num_layers - 1):
            name = "last_hidden_ffnn"
        else:
            name = None
        if layer >= 2:
            hidden = Dense(
                n_neurons1,
                activation="relu",
                kernel_regularizer=regularizers.L2(l2_reg),
                name=name
            )(hidden)
            last_hidden_ffnn = hidden
        else:
            hidden = Dense(
                n_neurons0,
                activation="relu",
                kernel_regularizer=regularizers.L2(l2_reg),
                name=name
            )(hidden)
            last_hidden_ffnn = hidden

    model = finalize_ffnn(input_epigenomic_data, last_hidden_ffnn, learning_rate)

    return {
        "model": model,
        "input_epigenomic_data": input_epigenomic_data,
        "last_hidden_ffnn": last_hidden_ffnn
    }


def build_ffnn_hp(
    X_train: np.ndarray,
    hp: HyperParameters
) -> Model:
    """Build a custom Feed-Forward Neural Network for 
    hyperparameter optimization.

    Parameters
    ----------
    X_train : np.ndarray
        Numpy array of input data representing the training set.

    hp : HyperParameters
        Instance of HyperParameters representing the hyperparameters 
        search space.

    Returns
    -------
    The compiled FFNN for hyperparameter optimization.
    """
    num_layers = hp.Int(name="num_layers", min_value=2, max_value=6)
    n_neurons0 = hp.Int(
        name="n_neurons0", min_value=32, max_value=256, step=32
    )
    l2_reg = hp.Float(name="l2_reg", min_value=0.0, max_value=0.1)
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-4])

    input_epigenomic_data = Input(
        (X_train.shape[1],), name="input_epigenomic_data"
    )
    last_hidden_ffnn = hidden = input_epigenomic_data

    for layer in range(num_layers):
        if layer == (num_layers - 1):
            name = "last_hidden_ffnn"
        else:
            name = None
        if layer >= 2:
            with hp.conditional_scope("num_layers", [3, 4, 5, 6]):
                n_neurons1 = hp.Int(
                    name="n_neurons1",
                    min_value=16,
                    max_value=128,
                    step=16
                )
                hidden = Dense(
                    n_neurons1,
                    activation="relu",
                    kernel_regularizer=regularizers.L2(l2_reg),
                    name=name
                )(hidden)
                last_hidden_ffnn = hidden
        else:
            hidden = Dense(
                n_neurons0,
                activation="relu",
                kernel_regularizer=regularizers.L2(l2_reg),
                name=name
            )(hidden)
            last_hidden_ffnn = hidden

    model = finalize_ffnn(input_epigenomic_data, last_hidden_ffnn, learning_rate)

    return model


def finalize_ffnn(
    input_epigenomic_data,
    last_hidden_ffnn,
    learning_rate
):
    output_ffnn = Dense(1, activation="sigmoid")(last_hidden_ffnn)
    model = Model(
        inputs=input_epigenomic_data, outputs=output_ffnn, name="ffnn"
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Nadam(learning_rate=learning_rate),
        metrics=get_complete_binary_metrics()
    )
    return model


# ==========
# CNN MODELS
# ==========
def build_cnn_iwbbio2020(
    window_size: int
) -> Dict[str, Sequential]:
    """Build the Convolutional Neural Network IWBBIO2020.
    This is the neural network described as Bayesian-CNN in [1].
    The paper mentions a batch size of 100 to train the network.

    Parameters
    ----------
    window_size : int
        Size of the window used to sample the data.

    Returns
    -------
    The compiled CNN.

    References
    ----------
        [1] L. Cappelletti et al., "Bayesian Optimization Improves 
        Tissue-Specific Prediction of Active Regulatory Regions 
        with Deep Neural Networks", in I. Rojas et al. (Eds.), 
        Bioinformatics and Biomedical Engineering - 8th International 
        Work-Conference, IWBBIO 2020, Proceedings, pp. 600-612, 
        Springer, 2020. https://doi.org/10.1007/978-3-030-45385-5_54
    """
    model = Sequential(
        layers=[
            InputLayer((window_size, 4)),
            Conv1D(64, kernel_size=5, activation="linear"),
            BatchNormalization(),
            ReLU(),
            Conv1D(64, kernel_size=5, activation="linear"),
            BatchNormalization(),
            ReLU(),
            Conv1D(64, kernel_size=5, activation="linear"),
            BatchNormalization(),
            ReLU(),
            MaxPool1D(pool_size=2),
            Conv1D(64, kernel_size=10, activation="linear"),
            BatchNormalization(),
            ReLU(),
            MaxPool1D(pool_size=2),
            Flatten(),
            Dense(
                64,
                activation="relu"
            ),
            Dropout(0.1),
            Dense(
                64,
                activation="relu"
            ),
            Dropout(0.1),
            Dense(1, activation="sigmoid")
        ],
        name="cnn_IWBBIO2020"
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Nadam(learning_rate=0.002),
        metrics=get_complete_binary_metrics()
    )

    return {"model": model}


def build_deep_enhancer(
    window_size: int
) -> Dict[str, Sequential]:
    """Build the Convolutional Neural Network DeepEnhancer.
    This is the CNN described in [1] as 4conv2pool4norm, adapted
    to the different shape of the input in our dataset. 
    In the original paper a softmax activation function was 
    employed in the output layer, composed by 2 neurons. Here we
    adopt the sigmoid activation function to perform classification.
    The maximum number of epochs was set to 30. The Authors applied 
    an unspecified learning rate decay schedule to the Adam 
    optimizer.

    An alternative implementation is available at:

    https://github.com/zommiommy/mendelian_snv_prediction/blob/master/mendelian_snv_prediction/deep_enhancer.py 
    
    Parameters
    ----------
    window_size : int
        Size of the window used to sample the data.

    Returns
    -------
    The compiled CNN.

    References
    ----------
        [1] Xu Min, Ning Chen, Ting Chen and Rui Jiang, 
        "DeepEnhancer: Predicting enhancers by convolutional 
        neural networks," 2016 IEEE International Conference 
        on Bioinformatics and Biomedicine (BIBM), 2016, 
        pp. 637-644. https://doi.org/10.1186/s12859-017-1878-3
    """
    model = Sequential(
        layers=[
            InputLayer((window_size, 4)),
            Conv1D(128, kernel_size=8),
            BatchNormalization(),
            ReLU(),
            Conv1D(128, kernel_size=8),
            BatchNormalization(),
            ReLU(),
            MaxPool1D(pool_size=2),
            Conv1D(64, kernel_size=3),
            BatchNormalization(),
            ReLU(),
            Conv1D(64, kernel_size=3),
            BatchNormalization(),
            ReLU(),
            MaxPool1D(pool_size=2),
            Flatten(),
            Dense(
                256,
                activation="relu"
            ),
            Dropout(0.5),
            Dense(
                128,
                activation="relu"
            ),
            Dense(
                1,
                activation="sigmoid"
            )
        ],
        name="deep_enhancer"
    )

    initial_learning_rate = 1e-4
    lr_schedule = optimizers.schedules.ExponentialDecay(
        initial_learning_rate,
        decay_steps=100000,
        decay_rate=0.01
    )

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(learning_rate=lr_schedule),
        metrics=get_complete_binary_metrics()
    )

    return {"model": model}


def build_deepcape(
    window_size: int
) -> Dict[str, Model]:
    """Build the Convolutional Neural Network DeepCAPE.
    This is the DNA module of the CNN described in [1], 
    slightly modified to match the input shape of our dataset. 
    The Authors mentioned using the Adam optimizer.

    The Authors provided a Python implementation of the original
    CNN DeepCAPE, available at:

    https://github.com/ShengquanChen/DeepCAPE/blob/master/train_net.py

    In the implementation, early stopping with patience 3 was
    employed. Batch size was 128, number of epochs was 30 and
    validation split was 0.1.
    
    Parameters
    ----------
    window_size : int
        Size of the window used to sample the data.

    Returns
    -------
    The compiled CNN.

    References
    ----------
        [1] S. Chen, M. Gan, H. Lv, R. Jiang, "DeepCAPE: A Deep 
        Convolutional Neural Network for the Accurate Prediction 
        of Enhancers, Genomics, Proteomics & Bioinformatics (2021), 
        doi: https://doi.org/10.1016/j.gpb.2019.04.006
    """
    input_seq = Input(shape=(window_size, 4), name="input_sequence_data")
    reshape = Reshape((256, 4, 1))(input_seq)
    seq_conv1_ = Conv2D(
        128,
        kernel_size=(8, 4),
        activation="relu",
        padding="valid"
    )
    seq_conv1 = seq_conv1_(reshape)
    seq_conv2_ = Conv2D(
        64,
        kernel_size=(1, 1),
        activation="relu",
        padding="same"
    )
    seq_conv2 = seq_conv2_(seq_conv1)
    seq_conv3_ = Conv2D(
        64,
        kernel_size=(3, 1),
        activation="relu",
        padding="same"
    )
    seq_conv3 = seq_conv3_(seq_conv2)
    seq_conv4_ = Conv2D(
        128,
        kernel_size=(1, 1),
        activation="relu",
        padding="same"
    )
    seq_conv4 = seq_conv4_(seq_conv3)
    seq_pool1 = MaxPool2D(pool_size=(2, 1))(seq_conv4)
    seq_conv5_ = Conv2D(
        64,
        kernel_size=(3, 1),
        activation="relu",
        padding="same"
    )
    seq_conv5 = seq_conv5_(seq_pool1)
    seq_conv6_ = Conv2D(
        64,
        kernel_size=(3, 1),
        activation="relu",
        padding="same"
    )
    seq_conv6 = seq_conv6_(seq_conv5)

    seq_conv7_ = Conv2D(
        128,
        kernel_size=(1, 1),
        activation="relu",
        padding="same"
    )
    seq_conv7 = seq_conv7_(seq_conv6)

    seq_pool2 = MaxPool2D(pool_size=(2, 1))(seq_conv7)
    merge_seq_conv2_conv3 = Concatenate(axis=-1)(
        [seq_conv2, seq_conv3]
    )
    merge_seq_conv5_conv6 = Concatenate(axis=-1)(
        [seq_conv5, seq_conv6]
    )
    x = Concatenate(axis=1)(
        [
            seq_conv1,
            merge_seq_conv2_conv3,
            merge_seq_conv5_conv6,
            seq_pool2
        ]
    )
    x = Flatten()(x)
    dense1_ = Dense(512, activation='relu')
    dense1 = dense1_(x)
    dense2 = Dense(256, activation='relu')(dense1)
    x = Dropout(0.5)(dense2)
    dense3 = Dense(128, activation='relu')(x)
    pred_output = Dense(1, activation='sigmoid')(dense3)
    model = Model(
        inputs=[input_seq],
        outputs=[pred_output],
        name="deepcape"
    )

    model.compile(
        optimizer=optimizers.Adam(
            learning_rate=1e-4,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=1e-6
        ),
        loss="binary_crossentropy",
        metrics=get_complete_binary_metrics()
    )

    return {"model": model}


def build_decode(
    X_train: np.ndarray
) -> Dict[str, Model]:
    """Build the Convolutional Neural Network DECODE.
    This is the CNN described in [1], modified to match the input 
    shape of our dataset and to simplify the Architecture. 
    The Authors mentioned using the Adam optimizer with a 5e-5 
    learning rate and a maximum of 100 epochs of training. Dropout 
    layers are also mentioned.

    Contrary to the other CNNs found in literature, the DECODE
    CNN is meant to be trained on epigenetic data, not on
    sequence data. 

    The Authors provided a Python implementation of the original
    CNN DECODE, available at:

    http://decode.gersteinlab.org/download.html

    In the implementation, a batch size of 32 was used, together
    with early stopping with patience equal to 2, a validation
    split of 0.1 and class weights for the loss function.

    Parameters
    ----------
    X_train : np.ndarray
        Numpy array of input data representing the training set.

    Returns
    -------
    The compiled CNN.

    References
    ----------
        [1] Chen, Z., Zhang, J., Liu, J., Dai, Y., Lee, D., 
        Min, M.R., Xu, M., & Gerstein, M. (2021). "DECODE: A 
        Deep-learning Framework for Condensing Enhancers and 
        Refining Boundaries with Large-scale Functional Assays". 
        DOI: 10.1101/2021.01.27.428477
    """
    import keras.backend as K

    def squeeze_excite(tensor, ratio=16):
        nb_channel = K.int_shape(tensor)[-1]

        hid = GlobalAveragePooling1D()(tensor)
        hid = Dense(nb_channel // ratio, activation='relu')(hid)
        hid = Dense(nb_channel, activation='sigmoid')(hid)

        hid = Multiply()([tensor, hid])
        return hid

    input_size = Input((X_train.shape[1],), name="input_epigenomic_data")
    reshape = Reshape((X_train.shape[1], 1))(input_size)
    conv1_ = Conv1D(
        128, 10, padding="same", activation="relu"
    )(reshape)
    conv1 = squeeze_excite(conv1_)
    conv2_ = Conv1D(
        64, 10, padding="same", activation="relu"
    )(conv1)
    conv2 = squeeze_excite(conv2_)
    conv3_ = Conv1D(
        64, 10, padding="same", activation="relu"
    )(conv2)
    conv3 = squeeze_excite(conv3_)
    conv4_ = Conv1D(
        128, 10, padding="valid", activation="relu"
    )(conv3)
    conv4 = squeeze_excite(conv4_)
    pool1 = MaxPool1D(pool_size=2)(conv4)
    conv5_ = Conv1D(
        64, 4, padding="same", activation="relu"
    )(pool1)
    conv5 = squeeze_excite(conv5_)
    conv6_ = Conv1D(
        64, 4, padding="same", activation="relu"
    )(conv5)
    conv6 = squeeze_excite(conv6_)
    conv7_ = Conv1D(
        128, 4, padding="same", activation="relu"
    )(conv6)
    conv7 = squeeze_excite(conv7_)
    pool2 = MaxPool1D(pool_size=2)(conv7)

    x = Flatten()(pool2)
    dense1 = Dense(256, activation='relu')(x)
    x = Dropout(0.4)(dense1)
    pred_output = Dense(1, activation='sigmoid')(x)
    model = Model(inputs=[input_size], outputs=[pred_output], name="decode")

    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Adam(
            learning_rate=5e-5,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-08,
            decay=9e-5
        ),
        metrics=get_complete_binary_metrics()
    )

    return {"model": model}


def build_cnn1d(
    window_size: int,
    num_conv_layers: int = 4,
    n_neurons0: int = 128,
    n_neurons1: int = 64,
    kernel_size0: int = 10,
    kernel_size1: int = 8,
    drop_rate: float = 0.6,
    learning_rate: float = 0.0001,
    l2_reg: float = 0.1
) -> Dict[str, Union[Model, Layer]]:
    """Build a custom 1D Convolutional Neural Network.

    Parameters
    ----------
    window_size : int
        Size of the window used to sample the data.

    num_conv_layers : int
        Number of hidden convolutional layers in the network.

    n_neurons0 : int
        Number of neurons in convolutional hidden layers number 0 
        and 1, if present.

    n_neurons1 : int
        Number of neurons in convolutional hidden layers from 
        number 2 onward, if present.

    kernel_size0 : int
        Kernel size for convolutional hidden layers number 0 
        and 1, if present.

    kernel_size1 : int
        Kernel size for convolutional hidden layers from 
        number 2 onward, if present.

    drop_rate : float
        Rate for the dropout layers.

    learning_rate : float
        Learning rate of the Nadam optimizer.

    l2_reg : float
        Amount of L2 regularization in the dense hidden layers.

    Returns
    -------
    The compiled 1D CNN.
    """
    input_sequence_data = Input((window_size, 4), name="input_sequence_data")
    hidden = input_sequence_data

    for num_conv_layer in range(num_conv_layers):
        if num_conv_layer >= 2:
            hidden = Conv1D(n_neurons1, kernel_size=kernel_size1)(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = ReLU()(hidden)
        else:
            hidden = Conv1D(n_neurons0, kernel_size=kernel_size0)(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = ReLU()(hidden)

        if num_conv_layer % 2 != 0:
            hidden = MaxPool1D(pool_size=2)(hidden)

    hidden = Flatten()(hidden)
    hidden = Dense(
        128,
        activation="relu",
        kernel_regularizer=regularizers.L2(l2_reg)
    )(hidden)
    hidden = Dropout(drop_rate)(hidden)
    last_hidden_cnn = Dense(
        64,
        activation="relu",
        kernel_regularizer=regularizers.L2(l2_reg)
    )(hidden)
    output_cnn = Dense(1, activation="sigmoid")(last_hidden_cnn)
    model = Model(
        inputs=input_sequence_data, outputs=output_cnn, name="cnn1d"
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Nadam(learning_rate=learning_rate),
        metrics=get_complete_binary_metrics()
    )

    return {
        "model": model,
        "input_sequence_data": input_sequence_data,
        "last_hidden_cnn": last_hidden_cnn
    }


def build_cnn1d_hp(
    window_size: int,
    hp: HyperParameters
) -> Model:
    """Build a custom 1D Convolutional Neural Network for 
    hyperparameter optimization.

    Parameters
    ----------
    window_size : int
        Size of the window used to sample the data.

    hp : HyperParameters
        Instance of HyperParameters representing the hyperparameters 
        search space.

    Returns
    -------
    The compiled 1D CNN for hyperparameter optimization.
    """
    num_conv_layers = hp.Int(
        name="num_conv_layers", min_value=2, max_value=8
    )
    n_neurons0 = hp.Int(
        name="n_neurons0", min_value=32, max_value=256, step=32
    )
    kernel_size0 = hp.Int(
        name="kernel_size0", min_value=5, max_value=10
    )

    drop_rate = hp.Float(
        name="drop_rate", min_value=0.0, max_value=0.5
    )

    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-4])

    input_sequence_data = Input((window_size, 4), name="input_sequence_data")
    hidden = input_sequence_data

    for num_conv_layer in range(num_conv_layers):
        if num_conv_layer >= 2:
            with hp.conditional_scope(
                    "num_conv_layers", [3, 4, 5, 6, 7, 8]
            ):
                n_neurons1 = hp.Int(
                    name="n_neurons1",
                    min_value=16,
                    max_value=128,
                    step=16
                )
                kernel_size1 = hp.Int(
                    name="kernel_size1",
                    min_value=2,
                    max_value=5
                )

                hidden = Conv1D(
                    n_neurons1, kernel_size=kernel_size1
                )(hidden)
                hidden = BatchNormalization()(hidden)
                hidden = ReLU()(hidden)
        else:
            hidden = Conv1D(
                n_neurons0, kernel_size=kernel_size0
            )(hidden)
            hidden = BatchNormalization()(hidden)
            hidden = ReLU()(hidden)

        if num_conv_layer % 2 != 0:
            hidden = MaxPool1D(pool_size=2)(hidden)

    last_hidden_cnn, model = finalize_cnn1d(
        drop_rate,
        hidden,
        input_sequence_data,
        learning_rate,
        "cnn1d_hp"
    )

    return model


def finalize_cnn1d(
    drop_rate,
    hidden,
    input_sequence_data,
    learning_rate,
    name
):
    hidden = Flatten()(hidden)
    hidden = Dense(
        512,
        activation="relu"
    )(hidden)
    hidden = Dropout(drop_rate)(hidden)
    hidden = Dense(
        256,
        activation="relu"
    )(hidden)
    hidden = Dropout(drop_rate)(hidden)
    last_hidden_cnn = Dense(
        128,
        activation="relu"
    )(hidden)
    output_cnn = Dense(1, activation="sigmoid")(last_hidden_cnn)
    model = Model(
        inputs=input_sequence_data, outputs=output_cnn, name=name
    )
    model.compile(
        loss="binary_crossentropy",
        optimizer=optimizers.Nadam(learning_rate=learning_rate),
        metrics=get_complete_binary_metrics()
    )
    return last_hidden_cnn, model


# ===========
# MMNN MODELS
# ===========
def build_mmnn(
    X_train: np.ndarray = None,
    window_size: int = None,
    models: Dict[str, Dict] = None,
    pretrained: bool = False,
    ffnn_model_name: str = "ffnn",
    cnn_model_name: str = "cnn1d"
) -> Dict[str, Model]:
    """Build a Multi-Modal Neural Network composed by a Feed-Forward
    Neural Network learning from epigenomic data and by a
    Convolutional Neural Network learning from sequence data.

    Parameters
    ----------
    X_train : np.ndarray or None
        Numpy array of input data representing the training set.

    window_size : int or None
        Size of the window used to sample the data.

    models : dict of str -> Dict or None
        Dictionary having the name of models as keys and as values
        dictionaries containing, at the key "model", the corresponding
        models. When the "pretrained" is True, the FFNN and CNN models
        will be retrieved from the dictionary according to the
        "ffnn_model_name" and "cnn_model_name" parameters.

    pretrained : bool
        Whether the input and last hidden layers of the FFNN and CNN
        models will be retrieved from already trained models in the
        models dictionary (True) or from models created anew (False).

    ffnn_model_name: str
        Name to use as key value in the models dictionary to retrieve
        the FFNN model.

    cnn_model_name: str
        Name to use as key value in the models dictionary to retrieve
        the CNN model.    

    Returns
    -------
    The compiled MMNN.
    """
    if not pretrained:
        ffnn_dict = build_ffnn(
            X_train
        )
        cnn_dict = build_cnn1d(
            window_size
        )
    else:
        ffnn_dict = models[ffnn_model_name]
        cnn_dict = models[cnn_model_name]

    input_epigenomic_data = ffnn_dict["input_epigenomic_data"]
    last_hidden_ffnn = ffnn_dict["last_hidden_ffnn"]
    input_sequence_data = cnn_dict["input_sequence_data"]
    last_hidden_cnn = cnn_dict["last_hidden_cnn"]

    concatenation_layer = Concatenate()([
        last_hidden_ffnn,
        last_hidden_cnn
    ])

    hidden_mmnn = Dense(128, activation="relu")(
        concatenation_layer
    )
    last_hidden_mmnn = Dense(64, activation="relu")(hidden_mmnn)
    output_mmnn = Dense(1, activation="sigmoid")(last_hidden_mmnn)

    model = Model(
        inputs=[input_epigenomic_data, input_sequence_data],
        outputs=output_mmnn,
        name="mmnn"
    )

    model.compile(
        optimizer="nadam",
        loss="binary_crossentropy",
        metrics=get_complete_binary_metrics()
    )

    return {"model": model}
