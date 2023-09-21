from tensorflow.keras.layers import (
    Input,
    Dropout,
    Flatten,
    Dense,
    Conv2D,
    Conv3D,
    AveragePooling2D,
    AvgPool2D,
    BatchNormalization,
    MaxPooling2D,
    MaxPooling3D,
    Masking,
)
from tensorflow.keras.models import Model

from tensorflow.keras.layers import Activation

# from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import SeparableConv2D, DepthwiseConv2D

# from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import SpatialDropout2D

# from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.constraints import max_norm
from tensorflow.keras import backend as K

from shared.utilities import MASKING_VALUE


def SAT1Start(n_channels, n_samples, n_classes):
    input = Input(shape=(n_channels, n_samples, 1))
    x = Conv2D(filters=16, kernel_size=(1, 5), activation="relu")(input)
    # x = AvgPool2D((1, 2))(x)
    # x = Dropout(0.25)(x)
    x = Conv2D(filters=32, kernel_size=(5, 5), activation="relu")(x)
    # x = AvgPool2D((1, 2))(x)
    # x = Dropout(0.25)(x)
    x = Conv2D(filters=64, kernel_size=(5, 1), activation="relu")(x)
    x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = AvgPool2D(pool_size=(1, 5), strides=(1, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    x = Dense(n_classes, activation="softmax")(x)
    model = Model(inputs=input, outputs=x)

    return model


def SAT1Base(n_channels, n_samples, n_classes):
    input = Input(shape=(n_channels, n_samples, 1))
    x = Masking(MASKING_VALUE)(input)
    x = Conv2D(filters=64, kernel_size=(1, 5), activation="relu")(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    # x = Dropout(0.25)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Conv2D(filters=128, kernel_size=(1, 3), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Conv2D(filters=256, kernel_size=(1, 3), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=x)

    return model


def SAT1Topological(n_x, n_y, n_samples, n_classes):
    input = Input(shape=(n_x, n_y, n_samples, 1))
    x = Masking(MASKING_VALUE)(input)
    x = Conv3D(filters=64, kernel_size=(5, 3, 5), activation="relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)
    x = Conv3D(filters=256, kernel_size=(1, 1, 3), activation="relu")(x)
    x = Dropout(0.25)(x)
    x = MaxPooling3D(pool_size=(1, 1, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=x)

    return model


def square(x):
    return K.square(x)


def log(x):
    return K.log(K.clip(x, min_value=1e-7, max_value=10000))


def ShallowConvNet(n_channels, n_samples, n_classes):
    input_main = Input((n_channels, n_samples, 1))
    block1 = Conv2D(
        40,
        (1, 13),
        input_shape=(n_channels, n_samples, 1),
        kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
    )(input_main)
    block1 = Conv2D(
        40,
        (n_channels, 1),
        use_bias=False,
        kernel_constraint=max_norm(2.0, axis=(0, 1, 2)),
    )(block1)
    block1 = BatchNormalization(epsilon=1e-05, momentum=0.9)(block1)
    block1 = Activation(square)(block1)
    block1 = AveragePooling2D(pool_size=(1, 30), strides=(1, 5))(block1)
    block1 = Activation(log)(block1)
    block1 = Dropout(0.5)(block1)
    flatten = Flatten()(block1)
    dense = Dense(n_classes, kernel_constraint=max_norm(0.5))(flatten)
    softmax = Activation("softmax")(dense)

    return Model(inputs=input_main, outputs=softmax)


def EEGNet(
    nb_classes,
    Chans=30,
    Samples=210,
    dropoutRate=0.5,
    kernLength=64,
    F1=8,
    D=2,
    F2=16,
    norm_rate=0.25,
    dropoutType="Dropout",
):
    """Keras Implementation of EEGNet
    http://iopscience.iop.org/article/10.1088/1741-2552/aace8c/meta

    Note that this implements the newest version of EEGNet and NOT the earlier
    version (version v1 and v2 on arxiv). We strongly recommend using this
    architecture as it performs much better and has nicer properties than
    our earlier version. For example:

        1. Depthwise Convolutions to learn spatial filters within a
        temporal convolution. The use of the depth_multiplier option maps
        exactly to the number of spatial filters learned within a temporal
        filter. This matches the setup of algorithms like FBCSP which learn
        spatial filters within each filter in a filter-bank. This also limits
        the number of free parameters to fit when compared to a fully-connected
        convolution.

        2. Separable Convolutions to learn how to optimally combine spatial
        filters across temporal bands. Separable Convolutions are Depthwise
        Convolutions followed by (1x1) Pointwise Convolutions.


    While the original paper used Dropout, we found that SpatialDropout2D
    sometimes produced slightly better results for classification of ERP
    signals. However, SpatialDropout2D significantly reduced performance
    on the Oscillatory dataset (SMR, BCI-IV Dataset 2A). We recommend using
    the default Dropout in most cases.

    Assumes the input signal is sampled at 128Hz. If you want to use this model
    for any other sampling rate you will need to modify the lengths of temporal
    kernels and average pooling size in blocks 1 and 2 as needed (double the
    kernel lengths for double the sampling rate, etc). Note that we haven't
    tested the model performance with this rule so this may not work well.

    The model with default parameters gives the EEGNet-8,2 model as discussed
    in the paper. This model should do pretty well in general, although it is
        advised to do some model searching to get optimal performance on your
        particular dataset.

    We set F2 = F1 * D (number of input filters = number of output filters) for
    the SeparableConv2D layer. We haven't extensively tested other values of this
    parameter (say, F2 < F1 * D for compressed learning, and F2 > F1 * D for
    overcomplete). We believe the main parameters to focus on are F1 and D.

    Inputs:

      nb_classes      : int, number of classes to classify
      Chans, Samples  : number of channels and time points in the EEG data
      dropoutRate     : dropout fraction
      kernLength      : length of temporal convolution in first layer. We found
                        that setting this to be half the sampling rate worked
                        well in practice. For the SMR dataset in particular
                        since the data was high-passed at 4Hz we used a kernel
                        length of 32.
      F1, F2          : number of temporal filters (F1) and number of pointwise
                        filters (F2) to learn. Default: F1 = 8, F2 = F1 * D.
      D               : number of spatial filters to learn within each temporal
                        convolution. Default: D = 2
      dropoutType     : Either SpatialDropout2D or Dropout, passed as a string.

    """

    if dropoutType == "SpatialDropout2D":
        dropoutType = SpatialDropout2D
    elif dropoutType == "Dropout":
        dropoutType = Dropout
    else:
        raise ValueError(
            "dropoutType must be one of SpatialDropout2D "
            "or Dropout, passed as a string."
        )

    input1 = Input(shape=(Chans, Samples, 1))

    ##################################################################
    block1 = Conv2D(
        F1,
        (1, kernLength),
        padding="same",
        input_shape=(Chans, Samples, 1),
        use_bias=False,
    )(input1)
    block1 = BatchNormalization()(block1)
    block1 = DepthwiseConv2D(
        (Chans, 1),
        use_bias=False,
        depth_multiplier=D,
        depthwise_constraint=max_norm(1.0),
    )(block1)
    block1 = BatchNormalization()(block1)
    block1 = Activation("elu")(block1)
    block1 = AveragePooling2D((1, 4))(block1)
    block1 = dropoutType(dropoutRate)(block1)

    block2 = SeparableConv2D(F2, (1, 16), use_bias=False, padding="same")(block1)
    block2 = BatchNormalization()(block2)
    block2 = Activation("elu")(block2)
    block2 = AveragePooling2D((1, 8))(block2)
    block2 = dropoutType(dropoutRate)(block2)

    flatten = Flatten(name="flatten")(block2)

    dense = Dense(nb_classes, name="dense", kernel_constraint=max_norm(norm_rate))(
        flatten
    )
    softmax = Activation("softmax", name="softmax")(dense)

    return Model(inputs=input1, outputs=softmax)
