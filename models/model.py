import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

def build_complex_model(input_shape):
    model = models.Sequential()

    model.add(layers.Conv1D(64, 3, activation='relu',
                            input_shape=input_shape,
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(128, 3, activation='relu',
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Conv1D(256, 3, activation='relu',
                            kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.BatchNormalization())
    model.add(layers.MaxPooling1D(2))

    model.add(layers.Dropout(0.5))

    model.add(layers.Bidirectional(
        layers.LSTM(128, return_sequences=True, dropout=0.3)
    ))
    model.add(layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.3)
    ))

    model.add(layers.Dense(128, activation='relu',
                           kernel_regularizer=regularizers.l2(0.01)))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu',
                           kernel_regularizer=regularizers.l2(0.01)))

    model.add(layers.Dense(3, activation='softmax'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.0005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
