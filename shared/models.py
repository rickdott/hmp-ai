from tensorflow.keras.layers import Input, Conv1D, MaxPool1D, Dropout, Flatten, Dense, Conv2D, MaxPool2D, Conv3D, MaxPool3D, AvgPool2D, BatchNormalization
from tensorflow.keras.models import Model

def SAT1Start(n_channels, n_samples, n_classes):
    input = Input(shape=(n_channels, n_samples, 1))
    x = Conv2D(filters=16, kernel_size=(1, 5), activation='relu')(input)
    x = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(x)
    # x = AvgPool2D(pool_size=(1, 2))(x)
    x = Dropout(0.25)(x)
    x = Conv2D(filters=64, kernel_size=(5, 1), activation='relu')(x)
    x = BatchNormalization(epsilon=1e-05, momentum=0.9)(x)
    x = AvgPool2D(pool_size=(1, 5), strides =(1, 2))(x)
    x = Dropout(0.25)(x)
    x = Flatten()(x)
    # x = Dense(128, activation='relu')(x)
    x = Dense(n_classes, activation='softmax')(x)
    model = Model(inputs=input, outputs=x)

    return model

