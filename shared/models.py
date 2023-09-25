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
from tensorflow.keras import Model
from shared.utilities import MASKING_VALUE


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


def SAT1Deep(n_channels, n_samples, n_classes):
    input = Input(shape=(n_channels, n_samples, 1))
    x = Masking(MASKING_VALUE)(input)
    x = Conv2D(filters=64, kernel_size=(1, 25), activation="relu")(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    # x = Dropout(0.25)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Conv2D(filters=128, kernel_size=(1, 17), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Conv2D(filters=256, kernel_size=(1, 11), activation="relu")(x)
    # x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Conv2D(filters=512, kernel_size=(1, 5), activation="relu")(x)
    x = Dropout(0.25)(x)
    # x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = MaxPooling2D(pool_size=(1, 2))(x)
    x = Conv2D(filters=1024, kernel_size=(1, 3), activation="relu")(x)
    x = Dropout(0.25)(x)
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
