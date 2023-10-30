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
    LSTM,
    SimpleRNN,
    GRU,
    Bidirectional,
    TimeDistributed,
)
from tensorflow.keras import Model
from pkg.hmpai.tensorflow.utilities import MASKING_VALUE


def SAT1Base(n_channels, n_samples, n_classes):
    input = Input(shape=(n_samples, n_channels, 1))
    x = Masking(MASKING_VALUE)(input)
    x = Conv2D(filters=64, kernel_size=(5, 1), activation="relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(filters=128, kernel_size=(3, 1), activation="relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(filters=256, kernel_size=(3, 1), activation="relu")(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=x)
    return model


# Model that is the same as SAT1Base except made for 3D data with a topological formulation
def SAT1Topological(n_x, n_y, n_samples, n_classes):
    input = Input(shape=(n_samples, n_x, n_y, 1))
    x = Masking(MASKING_VALUE)(input)
    x = Conv3D(filters=64, kernel_size=(5, 1, 1), activation="relu")(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    # x = Dropout(0.25)(x)
    x = MaxPooling3D(pool_size=(2, 1, 1))(x)
    x = Conv3D(filters=128, kernel_size=(3, 1, 1), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling3D(pool_size=(2, 1, 1))(x)
    x = Conv3D(filters=256, kernel_size=(3, 1, 1), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling3D(pool_size=(2, 1, 1))(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=x)
    return model


# Model for data with topological formulation, including convolution over the x and y dimensions
def SAT1TopologicalConv(n_x, n_y, n_samples, n_classes):
    input = Input(shape=(n_samples, n_x, n_y, 1))
    x = Masking(MASKING_VALUE)(input)
    x = Conv3D(filters=64, kernel_size=(5, 3, 3), activation="relu")(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    # x = Dropout(0.25)(x)
    # x = MaxPooling3D(pool_size=(1, 1, 2))(x)
    x = Conv3D(filters=128, kernel_size=(3, 3, 3), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling3D(pool_size=(2, 1, 1))(x)
    x = Conv3D(filters=256, kernel_size=(3, 1, 1), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    # x = MaxPooling3D(pool_size=(1, 1, 2))(x)
    # x = Conv3D(filters=512, kernel_size=(1, 1, 3), activation="relu")(x)
    x = Flatten()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=x)
    return model


def SAT1Deep(n_channels, n_samples, n_classes):
    input = Input(shape=(n_samples, n_channels, 1))
    x = Masking(MASKING_VALUE)(input)
    x = Conv2D(filters=64, kernel_size=(25, 1), activation="relu")(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    # x = Dropout(0.25)(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(filters=128, kernel_size=(17, 1), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(filters=256, kernel_size=(11, 1), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(filters=512, kernel_size=(5, 1), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Conv2D(filters=1024, kernel_size=(3, 1), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling2D(pool_size=(2, 1))(x)
    x = Flatten()(x)
    x = Dense(512, activation="relu")(x)
    x = Dropout(0.5)(x)
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=x)
    return model


def SAT1LSTM(n_channels, n_samples, n_classes):
    input = Input(shape=(n_samples, n_channels))
    x = Masking(MASKING_VALUE)(input)
    x = LSTM(256, dropout=0.25)(x)
    x = Dense(128)(x)
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=x)
    return model


def SAT1GRU(n_channels, n_samples, n_classes):
    input = Input(shape=(n_samples, n_channels))
    x = Masking(MASKING_VALUE)(input)
    x = GRU(256, dropout=0.25)(x)
    x = Dense(128)(x)
    x = Dense(n_classes, activation="softmax")(x)

    model = Model(inputs=input, outputs=x)
    return model


def SAT1seq2seqGRU(n_channels, n_samples, n_classes):
    input = Input(shape=(n_samples, n_channels))
    x = Masking(MASKING_VALUE)(input)
    x = GRU(128, return_sequences=True)(x)
    x = GRU(128, return_sequences=True)(x)
    # Predict for each timestep
    # x = TimeDistributed(Dense(n_classes, activation='softmax'))(x)
    x = Dense(n_classes, activation='softmax')(x)

    model = Model(inputs=input, outputs=x)
    return model
